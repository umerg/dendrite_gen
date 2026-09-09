"""Build the provisional main-paper table for unconditional neuron generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

from dendrite_gen.metrics.distributions import (
    CRITICAL_BRANCH_CABLE_LENGTH,
    CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
)
from dendrite_gen.visualization.paper.common import (
    DensitySamples,
    scalar_density_samples,
    tree_balanced_distribution_samples,
)
from dendrite_gen.visualization.paper.fig2_unconditional import paper_tree_scalar_stats
from dendrite_gen.visualization.utils.io import (
    load_gt_file_graphs,
    load_pred_graphs_from_pickle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "dendrite_gen"
if str(PACKAGE_ROOT) not in sys.path:
    # The training-time validation package still uses package-root imports such
    # as ``from validation...``. Keep that compatibility local to this runner.
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.tmd import compute_tmd_embedding  # noqa: E402
from validation.dist_metrics import (  # noqa: E402
    _apply_pca,
    _embed_matrix,
    assemble_morpho_vector,
    build_gt_cache,
    joint_metrics_from_vectors,
    standardize_vectors,
)


DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "val_extended"
DEFAULT_OURS_PKL = PROJECT_ROOT / "data" / "ours_best_val.pkl"
DEFAULT_SEMLA_PKL = PROJECT_ROOT / "data" / "semla_flow_unconditional_converted.pkl"
DEFAULT_TEX_OUTPUT = (
    PROJECT_ROOT / "iclr27-writeup" / "Tables" / "table1_unconditional.tex"
)
DEFAULT_JSON_OUTPUT = (
    PROJECT_ROOT
    / "dendrite_gen"
    / "outputs"
    / "paper_tables"
    / "table1_unconditional_metrics.json"
)
SO2_AXIS = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class MethodGraphs:
    label: str
    all_graphs: list[nx.Graph]
    valid_graphs: list[nx.Graph]
    validity_pct: float
    connectivity_pct: float
    sampling_ms: float | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--ours-pkl", type=Path, default=DEFAULT_OURS_PKL)
    parser.add_argument("--ours-ema-key", default="ema_1")
    parser.add_argument("--semla-pkl", type=Path, default=DEFAULT_SEMLA_PKL)
    parser.add_argument("--semla-ema-key", default="ema_1")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dc-k", type=int, default=5)
    parser.add_argument("--tmd-pca-ncomp", type=int, default=32)
    parser.add_argument("--ours-sampling-ms", type=float, default=None)
    parser.add_argument("--semla-sampling-ms", type=float, default=None)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def _raw_structure_flags(graph: nx.Graph) -> tuple[bool, bool]:
    """Return raw connected/tree flags, preferring conversion-time metadata."""
    raw_connected = graph.graph.get("conversion_raw_connected")
    raw_tree = graph.graph.get("conversion_raw_tree")
    if raw_connected is not None and raw_tree is not None:
        return bool(raw_connected), bool(raw_tree)
    if graph.number_of_nodes() == 0:
        return False, False
    return bool(nx.is_connected(graph)), bool(nx.is_tree(graph))


def _is_metric_ready(graph: nx.Graph) -> bool:
    """Check the rooted finite-position tree contract used by paper metrics."""
    if graph.number_of_nodes() == 0 or not nx.is_tree(graph):
        return False
    if graph.graph.get("root") not in graph:
        return False
    for node in graph.nodes:
        position = np.asarray(graph.nodes[node].get("pos"), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return False
    return True


def _prepare_method(
    label: str,
    graphs: list[nx.Graph],
    *,
    sampling_ms: float | None,
) -> MethodGraphs:
    flags = [_raw_structure_flags(graph) for graph in graphs]
    connected = np.asarray([flag[0] for flag in flags], dtype=bool)
    valid_tree = np.asarray([flag[1] for flag in flags], dtype=bool)
    valid_graphs = [
        graph
        for graph, is_raw_tree in zip(graphs, valid_tree)
        if is_raw_tree and _is_metric_ready(graph)
    ]
    count = max(len(graphs), 1)
    return MethodGraphs(
        label=label,
        all_graphs=graphs,
        valid_graphs=valid_graphs,
        validity_pct=100.0 * float(valid_tree.sum()) / count,
        connectivity_pct=100.0 * float(connected.sum()) / count,
        sampling_ms=sampling_ms,
    )


def _sample_graphs(
    graphs: list[nx.Graph],
    count: int,
    rng: np.random.Generator,
) -> list[nx.Graph]:
    if count > len(graphs):
        raise ValueError(f"Requested {count} graphs from a set of {len(graphs)}.")
    indices = rng.choice(len(graphs), size=count, replace=False)
    return [graphs[int(index)] for index in indices]


def _compute_joint_metrics(
    graphs: list[nx.Graph],
    gt_cache: dict[str, Any],
    *,
    dc_k: int,
) -> dict[str, float]:
    morpho = np.stack(
        [
            assemble_morpho_vector(
                graph,
                uhat=SO2_AXIS,
                radii=gt_cache["sholl_radii"],
            )
            for graph in graphs
        ],
        axis=0,
    )
    morpho_z = standardize_vectors(
        morpho,
        mean=gt_cache["morpho_mean"],
        std=gt_cache["morpho_std"],
    )
    metrics = joint_metrics_from_vectors(
        morpho_z,
        gt_cache["morpho_z"],
        prefix="morpho",
        sigma=gt_cache["morpho_sigma"],
        k=dc_k,
    )

    tmd_raw = _embed_matrix(graphs, gt_cache["embed_fn"])
    if tmd_raw.shape[0] != len(graphs):
        raise ValueError(
            f"TMD embedding succeeded for {tmd_raw.shape[0]}/{len(graphs)} graphs; "
            "refusing to hide an effective-sample-size mismatch."
        )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tmd_reduced = _apply_pca(tmd_raw, gt_cache["pca"])
    if not np.all(np.isfinite(tmd_reduced)):
        raise ValueError("TMD PCA projection produced non-finite values.")
    metrics.update(
        joint_metrics_from_vectors(
            tmd_reduced,
            gt_cache["tmd_reduced"],
            prefix="tmd",
            sigma=gt_cache["tmd_sigma"],
            k=dc_k,
        )
    )
    return metrics


def _weighted_std(samples: DensitySamples) -> float:
    weights = samples.weights / samples.weights.sum()
    mean = float(np.sum(weights * samples.values))
    return float(np.sqrt(np.sum(weights * (samples.values - mean) ** 2)))


def _normalized_w1(candidate: DensitySamples, reference: DensitySamples) -> float:
    if candidate.values.size == 0 or reference.values.size == 0:
        return float("nan")
    scale = _weighted_std(reference)
    if not np.isfinite(scale) or scale <= 1e-12:
        return float("nan")
    value = wasserstein_distance(
        candidate.values,
        reference.values,
        u_weights=candidate.weights,
        v_weights=reference.weights,
    )
    return float(value / scale)


def _mean_tree_balanced_marginal_w1(
    candidate_graphs: list[nx.Graph],
    reference_graphs: list[nx.Graph],
) -> float:
    """Average normalized W1 over the six marginals displayed in Figure 2."""
    candidate_rows = [paper_tree_scalar_stats(graph) for graph in candidate_graphs]
    reference_rows = [paper_tree_scalar_stats(graph) for graph in reference_graphs]
    values: list[float] = []
    for key in (
        "axial_extent",
        "lateral_span",
        "max_path_length",
        "total_cable_length",
    ):
        candidate = scalar_density_samples([row[key] for row in candidate_rows])
        reference = scalar_density_samples([row[key] for row in reference_rows])
        values.append(_normalized_w1(candidate, reference))

    for distribution in (
        CRITICAL_BRANCH_CABLE_LENGTH,
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
    ):
        candidate = tree_balanced_distribution_samples(candidate_graphs, distribution)
        reference = tree_balanced_distribution_samples(reference_graphs, distribution)
        values.append(_normalized_w1(candidate, reference))

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _best_labels(
    rows: dict[str, dict[str, float | None]],
    key: str,
    *,
    mode: str,
    reference_value: float | None = None,
) -> set[str]:
    values = {
        label: float(row[key])
        for label, row in rows.items()
        if label != "Real--real"
        and row.get(key) is not None
        and np.isfinite(float(row[key]))
    }
    if not values:
        return set()
    if mode == "min":
        score = values
    elif mode == "max":
        score = {label: -value for label, value in values.items()}
    elif mode == "zero":
        score = {label: abs(value) for label, value in values.items()}
    elif mode == "reference":
        if reference_value is None:
            raise ValueError("reference mode requires reference_value")
        score = {label: abs(value - reference_value) for label, value in values.items()}
    else:
        raise ValueError(f"Unknown best-value mode: {mode!r}")
    optimum = min(score.values())
    return {
        label for label, value in score.items() if np.isclose(value, optimum, rtol=1e-9)
    }


def _format_value(
    value: float | None,
    *,
    digits: int,
    bold: bool,
) -> str:
    if value is None or not np.isfinite(float(value)):
        return r"\textemdash"
    rendered = f"{float(value):.{digits}f}"
    return rf"\textbf{{{rendered}}}" if bold else rendered


def _render_latex(
    rows: dict[str, dict[str, float | None]],
    *,
    sample_size: int,
    seed: int,
    provenance: dict[str, str],
) -> str:
    real = rows["Real--real"]
    best = {
        "validity_pct": _best_labels(rows, "validity_pct", mode="max"),
        "connectivity_pct": _best_labels(rows, "connectivity_pct", mode="max"),
        "delta_mmd_morpho": _best_labels(rows, "delta_mmd_morpho", mode="zero"),
        "delta_mmd_tmd": _best_labels(rows, "delta_mmd_tmd", mode="zero"),
        "coverage_morpho": _best_labels(rows, "coverage_morpho", mode="max"),
        "density_morpho": _best_labels(
            rows,
            "density_morpho",
            mode="reference",
            reference_value=float(real["density_morpho"]),
        ),
        "marginal_w1": _best_labels(rows, "marginal_w1", mode="min"),
        "sampling_ms": _best_labels(rows, "sampling_ms", mode="min"),
    }

    def cell(label: str, key: str, digits: int) -> str:
        return _format_value(
            rows[label][key],
            digits=digits,
            bold=label in best[key],
        )

    table_rows = []
    for label in ("Real--real", "SemlaFlow", "Ours"):
        table_rows.append(
            " & ".join(
                [
                    label,
                    cell(label, "validity_pct", 1),
                    cell(label, "connectivity_pct", 1),
                    cell(label, "delta_mmd_morpho", 4),
                    cell(label, "delta_mmd_tmd", 4),
                    cell(label, "coverage_morpho", 3),
                    cell(label, "density_morpho", 3),
                    cell(label, "marginal_w1", 3),
                    cell(label, "sampling_ms", 1),
                ]
            )
            + r" \\"
        )

    comments = [
        "% AUTO-GENERATED by dendrite_gen.visualization.paper.table1_unconditional",
        "% PROVISIONAL: common legacy-validation plumbing, not final held-out-test results.",
        f"% Evaluation sample size per distribution row: {sample_size}; seed: {seed}.",
    ]
    comments.extend(f"% {key}: {value}" for key, value in provenance.items())
    body = [
        r"\begin{table*}[t]",
        r"    \centering",
        (
            r"    \caption{\saku{TEMP:} \textit{Provisional} unconditional neuron-generation "
            rf"results on the common legacy validation data ($N={sample_size}$ valid "
            r"trees per distributional comparison).}"
        ),
        r"    \label{tab:unconditional-main}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lcccccccc}",
        r"        \toprule",
        (
            r"        & \multicolumn{2}{c}{Structure} "
            r"& \multicolumn{5}{c}{Distributional agreement} "
            r"& \multicolumn{1}{c}{Efficiency} \\"
        ),
        r"        \cmidrule(lr){2-3}\cmidrule(lr){4-8}\cmidrule(lr){9-9}",
        (
            r"        Method "
            r"& \makecell{Valid tree\\(\%) $\uparrow$} "
            r"& \makecell{Connected\\(\%) $\uparrow$} "
            r"& \makecell{Morph.\\$\Delta\mathrm{MMD}^2\!\to\!0$} "
            r"& \makecell{TMD\\$\Delta\mathrm{MMD}^2\!\to\!0$} "
            r"& \makecell{Coverage\\$\uparrow$} "
            r"& \makecell{Density\\$\to$ ref.} "
            r"& \makecell{Mean norm.\\marginal $W_1$ $\downarrow$} "
            r"& \makecell{Sampling\\(ms/tree) $\downarrow$} \\"
        ),
        r"        \midrule",
        *(f"        {row}" for row in table_rows),
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"    \vspace{2pt}",
        (
            r"    \parbox{\textwidth}{\footnotesize "
            r"$\Delta\mathrm{MMD}^2$ is the additive excess over the matched "
            r"real--real reference. Coverage and density use the standardized "
            r"morphometric embedding ($k=5$). Mean normalized marginal $W_1$ "
            r"averages the six Figure~2 statistics and gives every tree equal total "
            r"weight. Structure is measured before repair; distributional metrics "
            r"use equal-sized valid-only subsets. Sampling times await a controlled "
            r"same-hardware benchmark. The current Ours evaluation inherits each "
            r"reference root degree, whereas SemlaFlow is unconditional; these rows "
            r"are therefore layout/plumbing results rather than final claims.}"
        ),
        r"\end{table*}",
        "",
    ]
    return "\n".join(comments + body)


def make_table(args: argparse.Namespace) -> tuple[Path, Path]:
    _, reference_all = load_gt_file_graphs(args.reference_dir)
    reference_valid = [graph for graph in reference_all if _is_metric_ready(graph)]

    ours = _prepare_method(
        "Ours",
        load_pred_graphs_from_pickle(args.ours_pkl, ema_key=args.ours_ema_key),
        sampling_ms=args.ours_sampling_ms,
    )
    semla = _prepare_method(
        "SemlaFlow",
        load_pred_graphs_from_pickle(args.semla_pkl, ema_key=args.semla_ema_key),
        sampling_ms=args.semla_sampling_ms,
    )

    maximum = min(len(reference_valid) // 2, len(ours.valid_graphs), len(semla.valid_graphs))
    sample_size = maximum if args.sample_size is None else int(args.sample_size)
    if sample_size < 2 or sample_size > maximum:
        raise ValueError(f"sample-size must be between 2 and {maximum}, got {sample_size}.")

    rng = np.random.default_rng(args.seed)
    reference_order = rng.permutation(len(reference_valid))
    reference = [reference_valid[int(index)] for index in reference_order[:sample_size]]
    real_peer = [
        reference_valid[int(index)]
        for index in reference_order[sample_size : 2 * sample_size]
    ]
    method_samples = {
        "SemlaFlow": _sample_graphs(semla.valid_graphs, sample_size, rng),
        "Ours": _sample_graphs(ours.valid_graphs, sample_size, rng),
    }

    embed_fn = lambda graph: compute_tmd_embedding(  # noqa: E731
        graph,
        filtration="radial_root",
        n_bins=16,
        uhat=SO2_AXIS,
    )
    print(f"Building real-reference cache from {sample_size} trees...")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gt_cache = build_gt_cache(
            reference,
            uhat=SO2_AXIS,
            embed_fn=embed_fn,
            tmd_pca_ncomp=args.tmd_pca_ncomp,
        )
    for key in ("morpho_z", "tmd_reduced"):
        if not np.all(np.isfinite(gt_cache[key])):
            raise ValueError(f"Reference cache contains non-finite values in {key!r}.")

    print("Computing matched real--real reference...")
    floor_joint = _compute_joint_metrics(real_peer, gt_cache, dc_k=args.dc_k)
    floor_w1 = _mean_tree_balanced_marginal_w1(real_peer, reference)

    method_lookup = {"SemlaFlow": semla, "Ours": ours}
    rows: dict[str, dict[str, float | None]] = {
        "Real--real": {
            "validity_pct": 100.0,
            "connectivity_pct": 100.0,
            "delta_mmd_morpho": 0.0,
            "delta_mmd_tmd": 0.0,
            "coverage_morpho": floor_joint["coverage_morpho"],
            "density_morpho": floor_joint["density_morpho"],
            "marginal_w1": floor_w1,
            "sampling_ms": None,
        }
    }
    raw_joint: dict[str, dict[str, float]] = {"Real--real": floor_joint}

    for label in ("SemlaFlow", "Ours"):
        print(f"Computing {label} metrics...")
        joint = _compute_joint_metrics(method_samples[label], gt_cache, dc_k=args.dc_k)
        marginal_w1 = _mean_tree_balanced_marginal_w1(
            method_samples[label], reference
        )
        method = method_lookup[label]
        rows[label] = {
            "validity_pct": method.validity_pct,
            "connectivity_pct": method.connectivity_pct,
            "delta_mmd_morpho": joint["mmd_morpho"] - floor_joint["mmd_morpho"],
            "delta_mmd_tmd": joint["mmd_tmd"] - floor_joint["mmd_tmd"],
            "coverage_morpho": joint["coverage_morpho"],
            "density_morpho": joint["density_morpho"],
            "marginal_w1": marginal_w1,
            "sampling_ms": method.sampling_ms,
        }
        raw_joint[label] = joint

    provenance = {
        "reference": str(args.reference_dir),
        "ours": f"{args.ours_pkl} [{args.ours_ema_key}]",
        "semlaflow": f"{args.semla_pkl} [{args.semla_ema_key}]",
    }
    latex = _render_latex(
        rows,
        sample_size=sample_size,
        seed=args.seed,
        provenance=provenance,
    )
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(latex, encoding="utf-8")

    payload = {
        "status": "provisional_legacy_validation",
        "sample_size": sample_size,
        "seed": args.seed,
        "protocol": {
            "reference_split": "one deterministic half-split of legacy val_extended",
            "distribution_policy": "valid raw outputs only; equal-sized subsets",
            "mmd": "unbiased RBF MMD^2; additive excess over real-real",
            "coverage_density_embedding": "standardized 16-D morphometric vector",
            "density_coverage_k": args.dc_k,
            "tmd": f"radial_root PI 16x16; PCA {args.tmd_pca_ncomp}",
            "marginal_w1": "mean normalized tree-balanced W1 over Figure 2 metrics",
            "sampling_time": "not yet benchmarked comparably",
            "conditioning_fairness": (
                "legacy Ours inherits each reference root degree; SemlaFlow is unconditional"
            ),
            "semlaflow_rooting": (
                "raw-valid converted samples use the converter's maximum-degree root "
                "because the SMOL artifact has no soma/root metadata"
            ),
        },
        "provenance": provenance,
        "available_counts": {
            "reference_all": len(reference_all),
            "reference_metric_ready": len(reference_valid),
            "ours_all": len(ours.all_graphs),
            "ours_raw_valid": len(ours.valid_graphs),
            "semlaflow_all": len(semla.all_graphs),
            "semlaflow_raw_valid": len(semla.valid_graphs),
        },
        "rows": rows,
        "raw_joint_metrics": raw_joint,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return args.tex_output, args.json_output


def main() -> None:
    args = _parse_args()
    tex_path, json_path = make_table(args)
    print(f"Wrote {tex_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
