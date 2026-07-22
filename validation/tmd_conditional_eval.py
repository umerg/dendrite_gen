"""Matched, TMD-conditioned pairwise fidelity metrics (live-eval companion to the
distributional suite in ``validation/dist_metrics.py``).

The distributional metrics pool statistics over the whole generated set vs the whole GT
set -- they answer "does the population look right?" but are blind to *per-conditioning*
failures: a generator can match every population marginal while mapping individual TMD
conditions to the wrong tree. When generation is conditioned on a per-GT-graph TMD, each
generated tree is produced *for a specific target*, and the rollout keeps them index-aligned
(``Trainer._evaluate_rollout`` reorders predictions back to GT order), so
``gen_graphs[i]`` was conditioned on ``gt_graphs[i]``. This module compares each such matched
pair directly:

  - PD Wasserstein (optionally bottleneck) between the two persistence diagrams, per
    filtration -- "did the generated tree realize the specific barcode we fed it?"
  - Per-pair geometric scalar diffs (|gt - pred|) of height / span_xy / bbox-diagonal.
  - Per-pair Wasserstein-1 between the two trees' branch-length and bifurcation-angle value
    sets -- the per-tree analog of the pooled ``branch_length_w1`` / ``bifurcation_angle_w1``.

``compute_conditional_pairwise_metrics`` is a pure function (like
``teacher_forced_eval.evaluate_teacher_forced``): it returns a flat ``{str: float}`` dict
(``{}`` if there are no usable pairs), never raises on a bad pair (that pair contributes nan
and is dropped by the nan-aware aggregation), and degrades to nan PD distances -- not a crash
-- when ``persim`` is unavailable. Every value is a Python ``float`` so it flows through
``Trainer.log`` to wandb under ``validation/ema_{beta}/tmd_cond/*``.
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from validation.dist_metrics import _w1
from validation.geometric_metric import (
    bbox_diag_length,
    height_z_range,
    span_xy_diameter,
)
from validation.structural_metrics import (
    _diagram_pairs,
    _pos_to_xyz,
    bifurcation_angle_values,
    branch_length_values,
)
from utils.tmd import compute_tmd_barcode_diagram

try:  # optional dependency; PD distances degrade to nan when absent (never a crash)
    import persim as _persim
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _persim = None


# --------------------------------------------------------------------------- helpers
def _finite_mean_median(vals: list[float]) -> tuple[float, float]:
    """(mean, median) over the finite entries; (nan, nan) if none are finite."""
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(np.median(a))


def _nan_frac(vals: list[float]) -> float:
    """Fraction of entries that are non-finite (nan/inf); nan if the list is empty."""
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(a)))


def _node_points(G: nx.Graph) -> np.ndarray:
    """All node positions as an (N,3) float array (empty (0,3) for a node-less graph)."""
    if G.number_of_nodes() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) for n in G.nodes()], axis=0)


def _pd_pairs(G: nx.Graph, filtration: str, normalize_mode: str, uhat) -> np.ndarray | None:
    """(M,2) canonical (birth<=death) diagram pairs for one filtration; None on any failure.

    ``compute_tmd_barcode_diagram`` requires a rooted tree graph with 3D ``pos``; a
    non-tree/unrooted/degenerate pred graph raises -- we catch and return None so the pair
    contributes nan rather than aborting the sweep.
    """
    try:
        _barcode, diagram = compute_tmd_barcode_diagram(
            G,
            filtration=filtration,
            normalize_mode=normalize_mode,
            weight_edges_by_euclidean=True,
            simplify_to_critical_tree=True,
            uhat=uhat,
        )
        return _diagram_pairs(diagram)
    except Exception:
        return None


def _pd_distance(pairs_a: np.ndarray, pairs_b: np.ndarray, *, kind: str) -> float:
    """persim Wasserstein/bottleneck between two (M,2) diagrams; nan if persim missing or on error.

    Both-empty diagrams are distance 0. persim matches a lone non-empty diagram to the
    diagonal on its own; any error (incl. an empty-diagram edge case) falls back to nan.
    """
    if _persim is None:
        return float("nan")
    if pairs_a.size == 0 and pairs_b.size == 0:
        return 0.0
    try:
        fn = _persim.wasserstein if kind == "wasserstein" else _persim.bottleneck
        return float(fn(pairs_a, pairs_b))
    except Exception:
        return float("nan")


def _safe_bifurcation(G: nx.Graph) -> np.ndarray:
    """Bifurcation-angle values, or an empty array if the graph has no usable root."""
    try:
        return bifurcation_angle_values(G, root=G.graph.get("root"))
    except Exception:
        return np.zeros((0,), dtype=np.float64)


def _pairwise_w1(a: np.ndarray, b: np.ndarray) -> float:
    """W1 between two per-tree value sets; nan if either side is empty (``_w1`` handles finiteness)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    return _w1(a, b)


# --------------------------------------------------------------------------- GT-side cache
def build_gt_pairwise_cache(gt_graphs: list[nx.Graph], *, uhat, pd_filtrations, normalize_mode: str) -> dict:
    """Precompute the per-GT objects that are identical across training steps.

    The GT eval set is fixed, so its diagrams, geometric scalars, and value arrays only need
    building once (mirrors ``Trainer._tf_batches_for`` / ``build_gt_cache``): halves per-pair
    work and keeps the pairwise curve comparable step-to-step. Returns a list aligned to
    ``gt_graphs``; each entry is a dict with ``diag`` (filtration -> pairs|None), ``height``,
    ``span_xy``, ``bbox``, ``bl`` (branch lengths), ``ba`` (bifurcation angles).
    """
    uhat = np.asarray(uhat, dtype=np.float64).reshape(3)
    filts = tuple(pd_filtrations)
    per: list[dict] = []
    for G in gt_graphs:
        pts = _node_points(G)
        per.append(
            {
                "diag": {f: _pd_pairs(G, f, normalize_mode, uhat) for f in filts},
                "height": height_z_range(pts, uhat),
                "span_xy": span_xy_diameter(pts, uhat),
                "bbox": bbox_diag_length(pts),
                "bl": branch_length_values(G),
                "ba": _safe_bifurcation(G),
            }
        )
    return {"uhat": tuple(uhat.tolist()), "filtrations": filts,
            "normalize_mode": normalize_mode, "per_gt": per}


# --------------------------------------------------------------------------- main entry point
def compute_conditional_pairwise_metrics(
    gen_graphs: list[nx.Graph],
    gt_graphs: list[nx.Graph],
    *,
    uhat=(0.0, 0.0, 1.0),
    pd_filtrations=("path", "radial_root"),
    max_pairs: int | None = 64,
    enable_wasserstein: bool = True,
    enable_bottleneck: bool = False,
    normalize_mode: str = "minmax",
    gt_cache: dict | None = None,
) -> dict[str, float]:
    """Index-matched per-pair fidelity of ``gen_graphs[i]`` vs its conditioning ``gt_graphs[i]``.

    Returns a flat dict of Python floats (``{}`` if no usable pairs). Never raises on a bad
    pair -- that pair contributes nan for the affected metric and is dropped by the nan-aware
    aggregation; structurally-unusable pred graphs are counted in ``n_pairs_skipped``.
    """
    uhat = np.asarray(uhat, dtype=np.float64).reshape(3)
    filts = tuple(pd_filtrations)

    n = min(len(gen_graphs), len(gt_graphs))
    if max_pairs is not None and int(max_pairs) > 0:
        n = min(n, int(max_pairs))
    if n == 0:
        return {}

    # Reuse the prebuilt GT cache when it matches this call's config; else build a local one.
    if (
        gt_cache is not None
        and tuple(gt_cache.get("filtrations", ())) == filts
        and gt_cache.get("normalize_mode") == normalize_mode
        and len(gt_cache.get("per_gt", [])) >= n
    ):
        gt_per = gt_cache["per_gt"]
    else:
        gt_per = build_gt_pairwise_cache(
            gt_graphs[:n], uhat=uhat, pd_filtrations=filts, normalize_mode=normalize_mode
        )["per_gt"]

    pd_w: dict[str, list[float]] = {f: [] for f in filts}
    pd_bn: dict[str, list[float]] = {f: [] for f in filts}
    height_d: list[float] = []
    span_d: list[float] = []
    bbox_d: list[float] = []
    bl_w1: list[float] = []
    ba_w1: list[float] = []
    n_skipped = 0

    for i in range(n):
        gp = gen_graphs[i]
        gt = gt_per[i]

        # Geometric scalar diffs (|gt - pred|); each extractor is nan-safe on empty input.
        pts_p = _node_points(gp)
        height_d.append(abs(gt["height"] - height_z_range(pts_p, uhat)))
        span_d.append(abs(gt["span_xy"] - span_xy_diameter(pts_p, uhat)))
        bbox_d.append(abs(gt["bbox"] - bbox_diag_length(pts_p)))

        # Per-pair W1 on the two trees' value sets.
        bl_w1.append(_pairwise_w1(branch_length_values(gp), gt["bl"]))
        ba_w1.append(_pairwise_w1(_safe_bifurcation(gp), gt["ba"]))

        # Persistence-diagram distances, per filtration.
        gp_bad = False
        for f in filts:
            pairs_g = _pd_pairs(gp, f, normalize_mode, uhat)
            pairs_t = gt["diag"].get(f)
            if pairs_g is None:
                gp_bad = True
            if pairs_g is None or pairs_t is None:
                if enable_wasserstein:
                    pd_w[f].append(float("nan"))
                if enable_bottleneck:
                    pd_bn[f].append(float("nan"))
                continue
            if enable_wasserstein:
                pd_w[f].append(_pd_distance(pairs_g, pairs_t, kind="wasserstein"))
            if enable_bottleneck:
                pd_bn[f].append(_pd_distance(pairs_g, pairs_t, kind="bottleneck"))
        if gp_bad:
            n_skipped += 1

    out: dict[str, float] = {}
    for f in filts:
        if enable_wasserstein:
            m, md = _finite_mean_median(pd_w[f])
            out[f"pd_wasserstein_{f}_mean"] = m
            out[f"pd_wasserstein_{f}_median"] = md
            out[f"pd_nan_frac_{f}"] = _nan_frac(pd_w[f])
        if enable_bottleneck:
            m, md = _finite_mean_median(pd_bn[f])
            out[f"pd_bottleneck_{f}_mean"] = m
            out[f"pd_bottleneck_{f}_median"] = md
            if not enable_wasserstein:
                out[f"pd_nan_frac_{f}"] = _nan_frac(pd_bn[f])

    for name, vals in (("height", height_d), ("span_xy", span_d), ("bbox_diag", bbox_d)):
        m, md = _finite_mean_median(vals)
        out[f"{name}_absdiff_mean"] = m
        out[f"{name}_absdiff_median"] = md

    for name, vals in (("branch_length", bl_w1), ("bifurcation_angle", ba_w1)):
        m, md = _finite_mean_median(vals)
        out[f"{name}_w1_pairwise_mean"] = m
        out[f"{name}_w1_pairwise_median"] = md

    out["n_pairs"] = float(n)
    out["n_pairs_skipped"] = float(n_skipped)
    return out
