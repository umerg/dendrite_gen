"""FROZEN ONE-OFF EXPERIMENT: do not extend this runner.

This script was used for a quick sweep of the September 2026 botanical-tree
generation artifacts. It is deliberately not part of the ongoing paper-figure
workflow. Run the existing paper figure plotters individually for future final
runs instead of generalizing or building on this file.

The attempted neuron generalization was stopped before use and may be
incomplete. Do not rely on ``--include-neurons``.

Generate separate population figures and paper-table results for generation runs.

The runner discovers the tree artifacts in ``data/trees_test_set_generations``,
matches each one to ``data/trees_genus_d{10,15,20}/test``, and writes one output
directory per artifact under ``dendrite_gen/outputs``.  With ``--include-neurons``
it also discovers the neuron SWC archives and uses ``neurons_conditional/test``.
All comparisons are set-level comparisons; prediction/reference indices are
never paired.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path, PurePosixPath
import pickle
import re
import tarfile
import tempfile
import traceback
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from dendrite_gen.metrics.distributions import (  # noqa: E402
    CRITICAL_BRANCH_CABLE_LENGTH,
    CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
)
from dendrite_gen.validation.structural_metrics import (  # noqa: E402
    contraction_ratio_values,
)
from dendrite_gen.visualization.convert_smol_to_pred_pkl import (  # noqa: E402
    _convert_one_sample,
    _format_summary,
)
from dendrite_gen.visualization.paper import table1_unconditional as table1  # noqa: E402
from dendrite_gen.visualization.paper.common import (  # noqa: E402
    configure_paper_style,
    density_bandwidth,
    density_domain,
    plot_density_comparison,
    save_png_pdf,
    scalar_density_samples,
    tree_balanced_distribution_samples,
)
from dendrite_gen.visualization.paper.colors import (  # noqa: E402
    OURS_COLOR,
    REFERENCE_COLOR,
    SEMLAFLOW_COLOR,
)
from dendrite_gen.visualization.paper.fig2_unconditional import (  # noqa: E402
    _tree_balanced_value_samples,
    paper_tree_scalar_stats,
)
from dendrite_gen.visualization.utils.io import (  # noqa: E402
    load_gt_file_graphs,
    load_tree_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dendrite_gen" / "outputs"
TREE_AXIS_PERMUTATION = (0, 2, 1)
DEPTH_PATTERN = re.compile(r"(?:^|_)d(10|15|20)(?:_|\.|$)")
PREDICTION_INDEX_PATTERN = re.compile(r"prediction_(\d+)\.swc$")


@dataclass(frozen=True)
class GenerationSpec:
    name: str
    path: Path
    domain: str
    depth: int | None
    kind: str

    @property
    def label(self) -> str:
        if self.name.startswith("semlaflow_"):
            return f"SemlaFlow d{self.depth}"
        words = self.name.replace("_trees_", " ").replace("_", " ").split()
        return " ".join(word.upper() if word == "tmd" else word for word in words)

    @property
    def output_name(self) -> str:
        return f"paper_tables_{self.name}"

    @property
    def reference_key(self) -> str:
        return f"trees_d{self.depth}" if self.domain == "trees" else "neurons"

    def reference_dir(self, data_root: Path) -> Path:
        if self.domain == "trees":
            return data_root / f"trees_genus_d{self.depth}" / "test"
        return data_root / "neurons_conditional" / "test"

    @property
    def reference_description(self) -> str:
        if self.domain == "trees":
            return f"depth-{self.depth} botanical-tree"
        return "held-out neuronal-tree"


@dataclass
class LoadedGeneration:
    graphs: list[nx.Graph]
    load_metadata: dict[str, Any]


@dataclass
class ReferenceContext:
    sample_size: int
    reference_graphs: list[nx.Graph]
    peer_graphs: list[nx.Graph]
    gt_cache: dict[str, Any]
    floor_joint: dict[str, float]
    floor_w1: float


def discover_generation_specs(
    data_root: Path, *, include_smol: bool = False, include_neurons: bool = False
) -> list[GenerationSpec]:
    """Discover supported tree artifacts, excluding similarly named neuron runs."""
    generation_dir = data_root / "trees_test_set_generations"
    if not generation_dir.is_dir():
        raise NotADirectoryError(f"Generation directory does not exist: {generation_dir}")

    paths = sorted(generation_dir.glob("*_trees_d*_test_swc.tar.gz"))
    if include_smol:
        paths.extend(sorted(generation_dir.glob("semlaflow_trees_d*.smol")))
    if include_neurons:
        paths.extend(sorted(generation_dir.glob("parity_neurons_*_test_swc.tar.gz")))
    specs: list[GenerationSpec] = []
    for path in paths:
        is_neuron = "neurons" in path.name
        match = DEPTH_PATTERN.search(path.name)
        if not is_neuron and match is None:
            continue
        if path.name.endswith("_test_swc.tar.gz"):
            name = path.name.removesuffix("_test_swc.tar.gz")
            kind = "swc_tar"
        elif path.suffix == ".smol":
            name = path.stem
            kind = "smol"
        else:
            continue
        specs.append(
            GenerationSpec(
                name=name,
                path=path,
                domain="neurons" if is_neuron else "trees",
                depth=None if is_neuron else int(match.group(1)),
                kind=kind,
            )
        )
    return sorted(specs, key=lambda spec: spec.name)


def _selected_specs(specs: Sequence[GenerationSpec], patterns: Sequence[str]) -> list[GenerationSpec]:
    if not patterns:
        return list(specs)
    selected = [
        spec
        for spec in specs
        if any(fnmatch.fnmatch(spec.name, pattern) for pattern in patterns)
    ]
    missing = [pattern for pattern in patterns if not any(fnmatch.fnmatch(s.name, pattern) for s in specs)]
    if missing:
        raise ValueError(f"No generation matched: {', '.join(missing)}")
    return selected


def _safe_swc_members(archive: tarfile.TarFile) -> list[tuple[int, tarfile.TarInfo]]:
    indexed: list[tuple[int, tarfile.TarInfo]] = []
    for member in archive.getmembers():
        if member.isdir():
            continue
        pure = PurePosixPath(member.name)
        if member.issym() or member.islnk() or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if not member.isfile() or not member.name.endswith(".swc"):
            raise ValueError(f"Unexpected archive member: {member.name}")
        match = PREDICTION_INDEX_PATTERN.search(pure.name)
        if match is None:
            raise ValueError(f"Prediction filename lacks a numeric index: {member.name}")
        indexed.append((int(match.group(1)), member))
    indexed.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed]
    if indices != list(range(len(indices))):
        raise ValueError(f"Prediction indices are not contiguous from zero: {indices[:5]}...{indices[-5:]}")
    return indexed


def _load_swc_tar(path: Path) -> LoadedGeneration:
    graphs: list[nx.Graph] = []
    with tarfile.open(path, "r:gz") as archive, tempfile.TemporaryDirectory(
        prefix="tree-generation-swc-"
    ) as temp_dir:
        members = _safe_swc_members(archive)
        temp_root = Path(temp_dir)
        for graph_index, member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            temp_path = temp_root / f"prediction_{graph_index:05d}.swc"
            temp_path.write_bytes(stream.read())
            graph = load_tree_graph(temp_path)
            graph.graph["graph_index"] = graph_index
            graphs.append(graph)
    return LoadedGeneration(
        graphs=graphs,
        load_metadata={
            "format": "tar.gz containing generated SWCs",
            "ordering": "numeric prediction index",
            "conversion": "none",
        },
    )


def _load_smol(path: Path) -> LoadedGeneration:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list in {path}, got {type(payload).__name__}")

    graphs: list[nx.Graph] = []
    stats: Counter[str] = Counter()
    for sample_index, item in enumerate(payload):
        sample = pickle.loads(item) if isinstance(item, (bytes, bytearray, memoryview)) else item
        if not isinstance(sample, dict):
            raise TypeError(f"SMOL sample {sample_index} is {type(sample).__name__}, expected dict")
        graph, sample_stats = _convert_one_sample(
            sample,
            sample_index=sample_index,
            component_mode="largest",
            root_mode="degree",
        )
        graphs.append(graph)
        stats.update(sample_stats)
    return LoadedGeneration(
        graphs=graphs,
        load_metadata={
            "format": "SemlaFlow .smol",
            "ordering": "list order; no reference IDs are present",
            "conversion": {
                "component_mode": "largest",
                "root_mode": "degree",
                "cycle_policy": "BFS spanning tree",
                "summary": list(_format_summary(stats)),
                "counts": dict(stats),
            },
        },
    )


def load_generation(spec: GenerationSpec) -> LoadedGeneration:
    if spec.kind == "swc_tar":
        return _load_swc_tar(spec.path)
    if spec.kind == "smol":
        return _load_smol(spec.path)
    raise ValueError(f"Unsupported generation kind: {spec.kind}")


def _map_tree_z_to_metric_y(graphs: Sequence[nx.Graph]) -> None:
    """Map botanical z-up coordinates to the legacy paper metric's y-up convention."""
    for graph in graphs:
        for node in graph.nodes:
            position = np.asarray(graph.nodes[node]["pos"], dtype=float)
            if position.shape != (3,):
                raise ValueError(f"Node {node!r} has invalid position shape {position.shape}")
            graph.nodes[node]["pos"] = position[list(TREE_AXIS_PERMUTATION)].copy()


def _cap(items: list[Any], maximum: int | None) -> list[Any]:
    return items if maximum is None else items[: max(0, maximum)]


def make_population_figure(
    reference_graphs: list[nx.Graph],
    candidate_graphs: list[nx.Graph],
    *,
    candidate_label: str,
    candidate_color: str,
    output_stem: Path,
    max_trees: int | None,
) -> tuple[Path, Path]:
    """Render the paper's eight distribution panels for one candidate."""
    configure_paper_style()
    sources = {
        "Reference": _cap(reference_graphs, max_trees),
        candidate_label: _cap(candidate_graphs, max_trees),
    }
    if any(not graphs for graphs in sources.values()):
        raise ValueError("Reference and generation must both contain at least one tree")
    colors = {"Reference": REFERENCE_COLOR, candidate_label: candidate_color}

    scalar_rows = {
        label: [paper_tree_scalar_stats(graph) for graph in graphs]
        for label, graphs in sources.items()
    }
    metric_panels: list[tuple[str, str, dict[str, Any], tuple[float, float], float]] = []
    for key, title, xlabel, bandwidth_scale in (
        ("axial_extent", "Axial extent", "Axial extent", 1.05),
        ("lateral_span", "Lateral span", "Lateral span", 1.05),
        ("max_path_length", "Maximum path length", "Maximum path length", 1.05),
        ("total_cable_length", "Total cable length", "Total cable length", 1.05),
    ):
        samples = {
            label: scalar_density_samples([row[key] for row in rows])
            for label, rows in scalar_rows.items()
        }
        domain = density_domain(list(samples.values()), lower=0.0)
        bandwidth = density_bandwidth(samples["Reference"], domain, scale_factor=bandwidth_scale)
        metric_panels.append((title, xlabel, samples, domain, bandwidth))

    for distribution_name, title, xlabel, bounds, bandwidth_scale in (
        (
            CRITICAL_BRANCH_CABLE_LENGTH,
            "Branch cable length",
            "Branch cable length",
            (0.0, None),
            1.15,
        ),
        (
            CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
            "Bifurcation angle",
            "Bifurcation angle (degrees)",
            (0.0, 180.0),
            1.10,
        ),
    ):
        samples = {
            label: tree_balanced_distribution_samples(graphs, distribution_name)
            for label, graphs in sources.items()
        }
        domain = density_domain(list(samples.values()), lower=bounds[0], upper=bounds[1])
        bandwidth = density_bandwidth(samples["Reference"], domain, scale_factor=bandwidth_scale)
        if distribution_name == CRITICAL_BRANCH_CABLE_LENGTH:
            domain = (-2.5 * bandwidth, domain[1])
        metric_panels.append((title, xlabel, samples, domain, bandwidth))

    contraction = {
        label: _tree_balanced_value_samples(graphs, contraction_ratio_values)
        for label, graphs in sources.items()
    }
    contraction_data_domain = density_domain(list(contraction.values()), lower=0.0, upper=1.0)
    contraction_bandwidth = density_bandwidth(
        contraction["Reference"], contraction_data_domain, scale_factor=1.05
    )
    metric_panels.append(
        (
            "Contraction ratio",
            "Contraction ratio",
            contraction,
            (contraction_data_domain[0], contraction_data_domain[1] + 2.5 * contraction_bandwidth),
            contraction_bandwidth,
        )
    )

    asymmetry = {
        label: scalar_density_samples([row["partition_asymmetry"] for row in rows])
        for label, rows in scalar_rows.items()
    }
    asymmetry_domain = density_domain(list(asymmetry.values()), lower=0.0, upper=1.0)
    asymmetry_bandwidth = density_bandwidth(
        asymmetry["Reference"], asymmetry_domain, scale_factor=1.05
    )
    metric_panels.append(
        ("Partition asymmetry", "Partition asymmetry", asymmetry, asymmetry_domain, asymmetry_bandwidth)
    )

    figure = plt.figure(figsize=(7.2, 2.25))
    grid = figure.add_gridspec(
        2, 4, left=0.06, right=0.99, top=0.96, bottom=0.12, hspace=0.52, wspace=0.30
    )
    axes = []
    for panel, axis in zip(metric_panels, grid.subplots().flat):
        title, xlabel, samples, domain, bandwidth = panel
        plot_density_comparison(
            axis,
            samples,
            colors=colors,
            domain=domain,
            bandwidth=bandwidth,
            xlabel=xlabel,
        )
        if title == "Branch cable length":
            axis.set_xticks([tick for tick in axis.get_xticks() if tick >= 0.0])
        elif title == "Contraction ratio":
            axis.set_xticks([tick for tick in axis.get_xticks() if tick <= 1.0])
        axes.append(axis)
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.0,
        fontsize=7.0,
    )
    paths = save_png_pdf(figure, output_stem)
    plt.close(figure)
    return paths


def _reference_context(
    reference_graphs: list[nx.Graph],
    *,
    sample_size: int,
    seed: int,
    dc_k: int,
    tmd_pca_ncomp: int,
) -> ReferenceContext:
    valid = [graph for graph in reference_graphs if table1._is_metric_ready(graph)]
    if 2 * sample_size > len(valid):
        raise ValueError(
            f"Need {2 * sample_size} reference trees for a disjoint floor, found {len(valid)}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(valid))
    reference = [valid[int(index)] for index in order[:sample_size]]
    peer = [valid[int(index)] for index in order[sample_size : 2 * sample_size]]
    embed_fn = lambda graph: table1.compute_tmd_embedding(  # noqa: E731
        graph,
        filtration="radial_root",
        n_bins=16,
        uhat=table1.SO2_AXIS,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gt_cache = table1.build_gt_cache(
            reference,
            uhat=table1.SO2_AXIS,
            embed_fn=embed_fn,
            tmd_pca_ncomp=tmd_pca_ncomp,
        )
        floor_joint = table1._compute_joint_metrics(peer, gt_cache, dc_k=dc_k)
    floor_w1 = table1._mean_tree_balanced_marginal_w1(peer, reference)
    return ReferenceContext(
        sample_size=sample_size,
        reference_graphs=reference,
        peer_graphs=peer,
        gt_cache=gt_cache,
        floor_joint=floor_joint,
        floor_w1=floor_w1,
    )


def _metric_row(
    loaded: LoadedGeneration,
    context: ReferenceContext,
    *,
    seed: int,
    dc_k: int,
) -> tuple[dict[str, float | int | None], dict[str, Any]]:
    prepared = table1._prepare_method(loaded.load_metadata.get("format", "Generation"), loaded.graphs, sampling_ms=None)
    metric_graphs = prepared.valid_graphs
    row: dict[str, float | int | None] = {
        "available_count": len(loaded.graphs),
        "evaluated_count": 0,
        "validity_pct": prepared.validity_pct,
        "connectivity_pct": prepared.connectivity_pct,
        "delta_mmd_morpho": None,
        "delta_mmd_tmd": None,
        "coverage_morpho": None,
        "density_morpho": None,
        "marginal_w1": None,
    }
    minimum = max(8, dc_k + 1)
    if len(metric_graphs) < minimum:
        return row, {
            "status": "insufficient_raw_valid_trees",
            "raw_valid_count": len(metric_graphs),
            "minimum_for_distribution_metrics": minimum,
        }
    if len(metric_graphs) < context.sample_size:
        raise ValueError(
            f"Reference context uses N={context.sample_size}, but only {len(metric_graphs)} raw-valid generations exist"
        )

    rng = np.random.default_rng(seed)
    candidate = table1._sample_graphs(metric_graphs, context.sample_size, rng)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        joint = table1._compute_joint_metrics(candidate, context.gt_cache, dc_k=dc_k)
    marginal_w1 = table1._mean_tree_balanced_marginal_w1(candidate, context.reference_graphs)
    row.update(
        {
            "evaluated_count": context.sample_size,
            "delta_mmd_morpho": float(joint["mmd_morpho"] - context.floor_joint["mmd_morpho"]),
            "delta_mmd_tmd": float(joint["mmd_tmd"] - context.floor_joint["mmd_tmd"]),
            "coverage_morpho": float(joint["coverage_morpho"]),
            "density_morpho": float(joint["density_morpho"]),
            "marginal_w1": float(marginal_w1),
        }
    )
    return row, {"status": "complete", "raw_valid_count": len(metric_graphs), "raw_joint_metrics": joint}


def _real_row(context: ReferenceContext) -> dict[str, float | int | None]:
    return {
        "available_count": 2 * context.sample_size,
        "evaluated_count": context.sample_size,
        "validity_pct": 100.0,
        "connectivity_pct": 100.0,
        "delta_mmd_morpho": 0.0,
        "delta_mmd_tmd": 0.0,
        "coverage_morpho": float(context.floor_joint["coverage_morpho"]),
        "density_morpho": float(context.floor_joint["density_morpho"]),
        "marginal_w1": float(context.floor_w1),
    }


def _tex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def _format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return r"\textemdash"
    numeric = float(value)
    if not np.isfinite(numeric):
        return r"\textemdash"
    return f"{numeric:.{digits}f}"


def render_table(
    spec: GenerationSpec,
    real: dict[str, float | int | None],
    candidate: dict[str, float | int | None],
    *,
    reference_dir: Path,
) -> str:
    def row(label: str, values: dict[str, float | int | None]) -> str:
        cells = [
            _tex_escape(label),
            str(values["available_count"]),
            str(values["evaluated_count"]),
            _format_metric(values["validity_pct"], 1),
            _format_metric(values["connectivity_pct"], 1),
            _format_metric(values["delta_mmd_morpho"], 4),
            _format_metric(values["delta_mmd_tmd"], 4),
            _format_metric(values["coverage_morpho"]),
            _format_metric(values["density_morpho"]),
            _format_metric(values["marginal_w1"]),
        ]
        return " & ".join(cells) + r" \\"

    return "\n".join(
        [
            "% AUTO-GENERATED by dendrite_gen.visualization.paper.run_tree_generation_tables",
            f"% generation: {spec.path}",
            f"% reference: {reference_dir}",
            r"\begin{table*}[t]",
            r"    \centering",
            (
                r"    \caption{Population-level results for "
                + _tex_escape(spec.label)
                + " against the "
                + _tex_escape(spec.reference_description)
                + " reference set.}"
            ),
            rf"    \label{{tab:{spec.name.replace('_', '-')}}}",
            r"    \resizebox{\textwidth}{!}{%",
            r"    \begin{tabular}{lrrrrrrrrr}",
            r"        \toprule",
            (
                r"        Method & Available & Eval. $N$ & Valid (\%) & Connected (\%) "
                r"& Morph. $\Delta\mathrm{MMD}^2$ & TMD $\Delta\mathrm{MMD}^2$ "
                r"& Coverage & Density & Mean norm. $W_1$ \\"
            ),
            r"        \midrule",
            "        " + row("Real--real", real),
            "        " + row(spec.label, candidate),
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table*}",
            "",
        ]
    )


def _alignment_note(spec: GenerationSpec, reference_count: int, generation_count: int) -> str:
    if spec.kind == "smol":
        return "unpaired population comparison; SMOL samples contain no reference IDs"
    if generation_count != reference_count:
        return (
            "unpaired population comparison; prediction/reference counts differ and the original "
            "evaluation manifest is unavailable"
        )
    if "_tmd" in spec.name and spec.name.startswith("parity_"):
        return (
            "population comparison; sorted-index alignment is strongly supported for optional "
            "target-specific follow-up, but is not used here"
        )
    return "population comparison; index alignment is not used"


def run_one(
    spec: GenerationSpec,
    *,
    data_root: Path,
    output_root: Path,
    sample_size: int | None,
    seed: int,
    dc_k: int,
    tmd_pca_ncomp: int,
    max_figure_trees: int | None,
    skip_figures: bool,
    reference_graph_cache: dict[str, list[nx.Graph]],
    reference_metric_cache: dict[tuple[str, int], ReferenceContext],
) -> Path:
    output_dir = output_root / spec.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = spec.reference_dir(data_root)
    if not reference_dir.is_dir():
        raise NotADirectoryError(f"Reference directory does not exist: {reference_dir}")

    if spec.reference_key not in reference_graph_cache:
        print(f"  Loading {spec.reference_key} reference trees...", flush=True)
        _, reference_graphs = load_gt_file_graphs(reference_dir)
        if spec.domain == "trees":
            _map_tree_z_to_metric_y(reference_graphs)
        reference_graph_cache[spec.reference_key] = reference_graphs
    reference_graphs = reference_graph_cache[spec.reference_key]

    loaded = load_generation(spec)
    if spec.domain == "trees":
        _map_tree_z_to_metric_y(loaded.graphs)
    prepared = table1._prepare_method(spec.label, loaded.graphs, sampling_ms=None)
    desired_sample_size = sample_size or len(reference_graphs) // 2
    desired_sample_size = min(desired_sample_size, len(reference_graphs) // 2)

    cache_key = (spec.reference_key, desired_sample_size)
    if cache_key not in reference_metric_cache:
        print(
            f"  Building {spec.reference_key} real--real metric floor "
            f"(N={desired_sample_size})...",
            flush=True,
        )
        reference_metric_cache[cache_key] = _reference_context(
            reference_graphs,
            sample_size=desired_sample_size,
            seed=seed,
            dc_k=dc_k,
            tmd_pca_ncomp=tmd_pca_ncomp,
        )
    context = reference_metric_cache[cache_key]

    print("  Computing table metrics...", flush=True)
    candidate_row, metric_details = _metric_row(loaded, context, seed=seed, dc_k=dc_k)
    real_row = _real_row(context)
    tex_path = output_dir / "table_tree_generation.tex"
    json_path = output_dir / "table_tree_generation_metrics.json"
    tex_path.write_text(
        render_table(spec, real_row, candidate_row, reference_dir=reference_dir), encoding="utf-8"
    )

    figure_paths: list[str] = []
    if not skip_figures:
        print("  Rendering population figure...", flush=True)
        figure_label = spec.label
        if spec.kind == "smol":
            figure_label += " (largest component)"
        png_path, pdf_path = make_population_figure(
            reference_graphs,
            loaded.graphs,
            candidate_label=figure_label,
            candidate_color=SEMLAFLOW_COLOR if spec.kind == "smol" else OURS_COLOR,
            output_stem=output_dir / "fig2_population_distributions",
            max_trees=max_figure_trees,
        )
        figure_paths = [str(png_path), str(pdf_path)]

    payload = {
        "status": "complete" if metric_details["status"] == "complete" else "partial",
        "generation": {
            "name": spec.name,
            "label": spec.label,
            "path": str(spec.path),
            "kind": spec.kind,
            "domain": spec.domain,
            "depth": spec.depth,
            "count": len(loaded.graphs),
            "load_metadata": loaded.load_metadata,
        },
        "reference": {
            "path": str(reference_dir),
            "count": len(reference_graphs),
            "domain": spec.domain,
            "depth": spec.depth,
        },
        "protocol": {
            "comparison": "population-level; no index pairing",
            "alignment_note": _alignment_note(spec, len(reference_graphs), len(loaded.graphs)),
            "axis": (
                "botanical z-up, mapped to the legacy metric implementation's y-up convention"
                if spec.domain == "trees"
                else "neuronal y-up"
            ),
            "reference_floor": "deterministic disjoint half split",
            "seed": seed,
            "requested_sample_size": sample_size,
            "reference_sample_size": context.sample_size,
            "candidate_distribution_policy": "raw-valid trees only",
            "figure_policy": (
                "all converted largest-component trees" if spec.kind == "smol" else "all generated SWC trees"
            ),
            "mmd": "unbiased RBF MMD^2; additive excess over real--real",
            "coverage_density_k": dc_k,
            "tmd": f"radial_root persistence image 16x16; PCA {tmd_pca_ncomp}",
            "marginal_w1": "mean normalized tree-balanced W1 over the six Figure 2 marginals",
        },
        "rows": {"Real--real": real_row, spec.label: candidate_row},
        "metric_details": metric_details,
        "raw_structure": {
            "validity_pct": prepared.validity_pct,
            "connectivity_pct": prepared.connectivity_pct,
            "raw_valid_count": len(prepared.valid_graphs),
        },
        "outputs": {"tex": str(tex_path), "json": str(json_path), "figures": figure_paths},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--only",
        nargs="*",
        default=(),
        metavar="PATTERN",
        help="Run only generation names matching one or more shell-style patterns.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dc-k", type=int, default=5)
    parser.add_argument("--tmd-pca-ncomp", type=int, default=32)
    parser.add_argument("--max-figure-trees", type=int, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--include-smol",
        action="store_true",
        help=(
            "Also process raw SemlaFlow .smol files using largest-component repair. "
            "They are excluded by default because their preprocessing policy needs "
            "to be chosen explicitly."
        ),
    )
    parser.add_argument(
        "--include-neurons",
        action="store_true",
        help=(
            "Also process parity_neurons_* SWC archives against "
            "data/neurons_conditional/test."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    specs = _selected_specs(
        discover_generation_specs(args.data_root, include_smol=args.include_smol),
        args.only,
    )
    if args.include_neurons:
        specs = _selected_specs(
            discover_generation_specs(
                args.data_root,
                include_smol=args.include_smol,
                include_neurons=True,
            ),
            args.only,
        )
    if not specs:
        raise ValueError("No tree generation artifacts were discovered")

    print(f"Discovered {len(specs)} tree generation run(s):", flush=True)
    for spec in specs:
        reference = spec.reference_dir(args.data_root)
        output = args.output_root / spec.output_name
        print(f"  {spec.name}: {spec.path} -> {reference} -> {output}", flush=True)
    if args.dry_run:
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    reference_graph_cache: dict[str, list[nx.Graph]] = {}
    reference_metric_cache: dict[tuple[str, int], ReferenceContext] = {}
    failures: list[str] = []
    for position, spec in enumerate(specs, start=1):
        print(f"[{position}/{len(specs)}] {spec.name}", flush=True)
        output_dir = args.output_root / spec.output_name
        try:
            written = run_one(
                spec,
                data_root=args.data_root,
                output_root=args.output_root,
                sample_size=args.sample_size,
                seed=args.seed,
                dc_k=args.dc_k,
                tmd_pca_ncomp=args.tmd_pca_ncomp,
                max_figure_trees=args.max_figure_trees,
                skip_figures=args.skip_figures,
                reference_graph_cache=reference_graph_cache,
                reference_metric_cache=reference_metric_cache,
            )
            print(f"  Wrote {written}", flush=True)
        except Exception as error:  # keep independent runs independent
            failures.append(spec.name)
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "error.json"
            failure_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "generation": spec.name,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  FAILED: {error} (details: {failure_path})", flush=True)
    if failures:
        print(f"Failed runs: {', '.join(failures)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
