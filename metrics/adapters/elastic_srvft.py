"""Adapter for the optional Elastic SRVFT research implementation.

The external checkout is intentionally not vendored.  This module loads its
``python_distance`` package from an exact local checkout, converts a rooted
NetworkX tree to the upstream contiguous SWC representation, and forms this
project's SO(2)-only quotient around the preferred z axis.

The upstream scalar is named ``E`` and is an alignment *energy*.  It has not
been established here as a mathematical metric, so the adapter preserves that
terminology and does not silently take a square root.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import math
from numbers import Integral
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, Callable, Hashable, Literal
import warnings

import networkx as nx
import numpy as np

from ..so2 import minimize_over_so2, rotation_matrix_about_axis


DepthPolicy = Literal["raise", "warn", "allow"]
Symmetrization = Literal["none", "mean"]

DEFAULT_ELASTIC_SRVFT_CHECKOUT = (
    Path(__file__).resolve().parents[1] / "external" / "elastic_srvft"
)
_MIN_EDGE_LENGTH = 1e-12
_NEGATIVE_ENERGY_TOLERANCE = 1e-10


class ElasticSRVFTError(RuntimeError):
    """Base class for adapter errors."""


class ElasticSRVFTNotConfigured(ElasticSRVFTError):
    """Raised when the expected external checkout cannot be found."""


class ElasticSRVFTDependencyError(ElasticSRVFTError):
    """Raised when the external Python implementation cannot be imported."""


class ElasticSRVFTUnsupportedTree(ElasticSRVFTError):
    """Raised when upstream conversion would silently omit tree structure."""


@dataclass(frozen=True)
class ElasticSRVFTResult:
    """One Elastic SRVFT energy evaluation and quotient diagnostics."""

    value: float
    energy: float
    forward_energy: float
    reverse_energy: float | None
    energy_at_zero_rotation: float
    forward_energy_at_zero_rotation: float
    reverse_energy_at_zero_rotation: float | None
    angle_rad: float
    grid_energy: float
    grid_angle_rad: float
    quotient_so2: bool
    grid_size: int
    refine: bool
    refinement_tolerance: float
    objective_evaluations: int
    upstream_energy_evaluations: int
    runtime_seconds: float
    lam_m: float
    lam_s: float
    lam_p: float
    symmetrization: Symmetrization
    depth_policy: DepthPolicy
    tree_a_nodes: int
    tree_b_nodes: int
    tree_a_terminal_leaves: int
    tree_b_terminal_leaves: int
    tree_a_represented_branches: int
    tree_b_represented_branches: int
    tree_a_omitted_frontier_branches: int
    tree_b_omitted_frontier_branches: int
    tree_a_canonical_order_ties: int
    tree_b_canonical_order_ties: int
    radius_used_in_energy: bool
    external_revision: str
    external_checkout: str


@dataclass(frozen=True)
class _ExternalAPI:
    compute_distance_energy: Callable[..., tuple[dict[str, Any], object, object]]
    comp_tree_from_swcdata_rad: Callable[..., dict[str, Any]]
    comp_tree_to_qcomp_tree_rad_4layers: Callable[..., dict[str, Any]]
    checkout: Path
    revision: str


@dataclass(frozen=True)
class _PreparedExternalTree:
    qtree: dict[str, Any]
    node_count: int
    terminal_leaf_count: int
    represented_branch_count: int
    omitted_frontier_branches: int
    canonical_order_ties: int


_API_CACHE: dict[Path, _ExternalAPI] = {}
_API_LOAD_LOCK = threading.RLock()


def _checkout_path(checkout: str | Path | None) -> Path:
    resolved = Path(checkout or DEFAULT_ELASTIC_SRVFT_CHECKOUT).expanduser().resolve()
    package_init = resolved / "python_distance" / "__init__.py"
    if not package_init.is_file():
        raise ElasticSRVFTNotConfigured(
            "Elastic SRVFT Python checkout not found. Expected "
            f"{package_init}. Clone the repository to "
            "metrics/external/elastic_srvft or pass checkout=...."
        )
    return resolved


def _module_is_from_checkout(module: ModuleType, checkout: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(module_file).resolve().relative_to(checkout / "python_distance")
    except ValueError:
        return False
    return True


def _git_revision(checkout: Path) -> str:
    try:
        top_level = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if Path(top_level.stdout.strip()).resolve() != checkout:
            return "not-a-git-checkout"
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = completed.stdout.strip() or "unknown"
    return f"{revision}-dirty" if dirty.stdout.strip() else revision


def _load_external_api(checkout: str | Path | None = None) -> _ExternalAPI:
    """Load ``python_distance`` from one exact checkout without changing sys.path."""
    resolved = _checkout_path(checkout)
    with _API_LOAD_LOCK:
        cached = _API_CACHE.get(resolved)
        if cached is not None:
            return cached

        package_name = "python_distance"
        conflicting = [
            (name, module)
            for name, module in sys.modules.items()
            if (name == package_name or name.startswith(f"{package_name}."))
            and module is not None
            and not _module_is_from_checkout(module, resolved)
        ]
        if conflicting:
            name, module = conflicting[0]
            raise ElasticSRVFTDependencyError(
                f"A conflicting {name!r} module is already loaded from "
                f"{getattr(module, '__file__', None)!r}; cannot safely mix it "
                f"with the Elastic SRVFT checkout at {resolved}."
            )

        existing = sys.modules.get(package_name)
        loaded_here = existing is None
        modules_before = set(sys.modules)
        try:
            if existing is None:
                package_dir = resolved / package_name
                spec = importlib.util.spec_from_file_location(
                    package_name,
                    package_dir / "__init__.py",
                    submodule_search_locations=[str(package_dir)],
                )
                if spec is None or spec.loader is None:
                    raise ElasticSRVFTDependencyError(
                        f"Could not construct an import spec for {package_dir}."
                    )
                module = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = module
                spec.loader.exec_module(module)

            reparam = importlib.import_module("python_distance.core.reparam")
            swc = importlib.import_module("python_distance.io.swc")
            convert = importlib.import_module("python_distance.io.convert")
        except ImportError as exc:
            if loaded_here:
                for name in set(sys.modules) - modules_before:
                    if name == package_name or name.startswith(f"{package_name}."):
                        sys.modules.pop(name, None)
            dependency = getattr(exc, "name", None) or "one of its dependencies"
            raise ElasticSRVFTDependencyError(
                f"Elastic SRVFT could not import {dependency!r}: {exc}. In the "
                "local trees environment install the checkout's "
                "python_distance/requirements.txt dependencies (at minimum "
                "numba>=0.58)."
            ) from exc
        except Exception:
            if loaded_here:
                for name in set(sys.modules) - modules_before:
                    if name == package_name or name.startswith(f"{package_name}."):
                        sys.modules.pop(name, None)
            raise

        api = _ExternalAPI(
            compute_distance_energy=reparam.compute_distance_energy,
            comp_tree_from_swcdata_rad=swc.comp_tree_from_swcdata_rad,
            comp_tree_to_qcomp_tree_rad_4layers=(
                convert.comp_tree_to_qcomp_tree_rad_4layers
            ),
            checkout=resolved,
            revision=_git_revision(resolved),
        )
        _API_CACHE[resolved] = api
        return api


def _validated_rooted_geometry(
    graph: nx.Graph,
    *,
    name: str,
) -> tuple[Hashable, dict[Hashable, np.ndarray], dict[Hashable, Hashable | None]]:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{name} must be a NetworkX graph.")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError(f"{name} must be a simple undirected tree.")
    if graph.number_of_nodes() < 2:
        raise ValueError(f"{name} must contain at least one edge.")
    if not nx.is_tree(graph):
        raise ValueError(f"{name} must be connected and acyclic.")
    root = graph.graph.get("root")
    if root not in graph:
        raise ValueError(f"{name}.graph['root'] must name an existing root node.")

    positions: dict[Hashable, np.ndarray] = {}
    for node in graph.nodes:
        if "pos" not in graph.nodes[node]:
            raise ValueError(f"{name} node {node!r} is missing 'pos'.")
        position = np.asarray(graph.nodes[node]["pos"], dtype=np.float64).reshape(-1)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name} node {node!r} must have a finite 3-D position.")
        positions[node] = position

    for node_a, node_b in graph.edges:
        edge_length = float(np.linalg.norm(positions[node_a] - positions[node_b]))
        if edge_length <= _MIN_EDGE_LENGTH:
            raise ElasticSRVFTUnsupportedTree(
                f"{name} edge {node_a!r}--{node_b!r} has length {edge_length:.3g}. "
                "The external longest-path decomposition can fail to terminate "
                "on zero-length edges; clean or contract duplicate-position "
                "nodes before using Elastic SRVFT."
            )

    parent: dict[Hashable, Hashable | None] = {root: None}
    queue = [root]
    for node in queue:
        for neighbor in graph.neighbors(node):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    return root, positions, parent


def _subtree_signatures(
    graph: nx.Graph,
    root: Hashable,
    positions: dict[Hashable, np.ndarray],
    parent: dict[Hashable, Hashable | None],
) -> tuple[dict[Hashable, list[Hashable]], dict[Hashable, bytes], int]:
    children = {node: [] for node in graph.nodes}
    for node, parent_node in parent.items():
        if parent_node is not None:
            children[parent_node].append(node)

    signatures: dict[Hashable, bytes] = {}
    canonical_order_ties = 0
    # ``parent`` was populated breadth-first, so reversing it is a valid
    # iterative postorder even for very deep neuronal chains.
    for node in reversed(parent):
        child_signatures = sorted(signatures[child] for child in children[node])
        parent_node = parent[node]
        if parent_node is None:
            edge_signature = (0.0, 0.0, 0.0)
        else:
            delta = positions[node] - positions[parent_node]
            edge_signature = (
                round(float(np.linalg.norm(delta)), 12),
                round(float(delta[2]), 12),
                round(float(np.linalg.norm(delta[:2])), 12),
            )
        payload = (
            f"{edge_signature[0]:.12g}|{edge_signature[1]:.12g}|"
            f"{edge_signature[2]:.12g}|{len(children[node])}|"
        ).encode("ascii") + b"".join(child_signatures)
        signatures[node] = hashlib.sha256(payload).digest()
        canonical_order_ties += len(child_signatures) - len(set(child_signatures))

    return children, signatures, canonical_order_ties


def _canonical_preorder(
    root: Hashable,
    children: dict[Hashable, list[Hashable]],
    signatures: dict[Hashable, bytes],
) -> list[Hashable]:
    """Return parent-first order without Python recursion.

    Equal signatures remain in the graph's stable adjacency order. Such ties
    are reported by the public result because the upstream implementation uses
    the first equal-length terminal path as its trunk.
    """
    order: list[Hashable] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        ordered_children = sorted(children[node], key=signatures.__getitem__)
        stack.extend(reversed(ordered_children))
    return order


def _graph_to_external_swc(
    graph: nx.Graph,
    *,
    name: str,
    default_radius: float,
) -> tuple[np.ndarray, int, int]:
    if not math.isfinite(default_radius) or default_radius <= 0.0:
        raise ValueError("default_radius must be finite and positive.")
    root, positions, parent = _validated_rooted_geometry(graph, name=name)
    children, signatures, canonical_order_ties = _subtree_signatures(
        graph, root, positions, parent
    )
    order = _canonical_preorder(root, children, signatures)
    external_id = {node: index + 1 for index, node in enumerate(order)}
    origin = positions[root]
    raw = np.empty((len(order), 7), dtype=np.float64)
    for row, node in enumerate(order):
        attributes = graph.nodes[node]
        radius = float(attributes.get("radius", default_radius))
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError(f"{name} node {node!r} has an invalid radius.")
        swc_type = int(attributes.get("swc_type", 3))
        parent_node = parent[node]
        parent_id = -1 if parent_node is None else external_id[parent_node]
        raw[row] = (
            external_id[node],
            swc_type,
            *(positions[node] - origin),
            radius,
            parent_id,
        )

    terminal_leaf_count = sum(not children[node] for node in order)
    return raw, terminal_leaf_count, canonical_order_ties


def _four_layer_diagnostics(comp_tree: dict[str, Any]) -> tuple[int, int]:
    first_layer = list(comp_tree.get("beta_children", []))
    second_layer = [
        node for parent in first_layer for node in parent.get("beta_children", [])
    ]
    third_layer = [
        node for parent in second_layer for node in parent.get("beta_children", [])
    ]
    omitted_frontier = sum(int(node.get("K_sideNum", 0)) for node in third_layer)
    represented = 1 + len(first_layer) + len(second_layer) + len(third_layer)
    return represented, omitted_frontier


def _prepare_external_tree(
    graph: nx.Graph,
    api: _ExternalAPI,
    *,
    name: str,
    default_radius: float,
    depth_policy: DepthPolicy,
) -> _PreparedExternalTree:
    raw, terminal_leaves, canonical_order_ties = _graph_to_external_swc(
        graph,
        name=name,
        default_radius=default_radius,
    )
    comp_tree = api.comp_tree_from_swcdata_rad(raw, 4)
    represented, omitted_frontier = _four_layer_diagnostics(comp_tree)
    if omitted_frontier:
        message = (
            f"{name} exceeds the external implementation's fixed four branch "
            f"layers; at least {omitted_frontier} layer-5 frontier branches "
            "would be silently omitted."
        )
        if depth_policy == "raise":
            raise ElasticSRVFTUnsupportedTree(
                f"{message} Pass depth_policy='warn' or 'allow' only for an "
                "explicit truncated-representation study."
            )
        if depth_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    qtree = api.comp_tree_to_qcomp_tree_rad_4layers(comp_tree)
    return _PreparedExternalTree(
        qtree=qtree,
        node_count=graph.number_of_nodes(),
        terminal_leaf_count=terminal_leaves,
        represented_branch_count=represented,
        omitted_frontier_branches=omitted_frontier,
        canonical_order_ties=canonical_order_ties,
    )


def _rotate_qtree_copy(qtree: dict[str, Any], angle_rad: float) -> dict[str, Any]:
    """Rotate every geometric q field once, breaking harmless array aliases."""
    rotation = rotation_matrix_about_axis(angle_rad, (0.0, 0.0, 1.0))
    rotated = copy.deepcopy(qtree)

    def visit(node: dict[str, Any]) -> None:
        node["q0"] = rotation @ np.asarray(node["q0"], dtype=np.float64)
        node["q"] = [
            rotation @ np.asarray(branch, dtype=np.float64)
            for branch in node.get("q", [])
        ]
        if "b00_startP" in node:
            start = np.asarray(node["b00_startP"], dtype=np.float64).reshape(3)
            node["b00_startP"] = rotation @ start
        for child in node.get("q_children", []):
            visit(child)

    visit(rotated)
    return rotated


def _validate_options(
    *,
    lam_m: float,
    lam_s: float,
    lam_p: float,
    depth_policy: str,
    symmetrization: str,
    grid_size: int,
    refinement_tolerance: float,
) -> None:
    for name, value in (("lam_m", lam_m), ("lam_s", lam_s), ("lam_p", lam_p)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if depth_policy not in {"raise", "warn", "allow"}:
        raise ValueError("depth_policy must be 'raise', 'warn', or 'allow'.")
    if symmetrization not in {"none", "mean"}:
        raise ValueError("symmetrization must be 'none' or 'mean'.")
    if isinstance(grid_size, bool) or not isinstance(grid_size, Integral):
        raise ValueError("grid_size must be an integer.")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if not math.isfinite(refinement_tolerance) or refinement_tolerance <= 0.0:
        raise ValueError("refinement_tolerance must be finite and positive.")


def elastic_srvft_distance(
    tree_a: nx.Graph,
    tree_b: nx.Graph,
    *,
    checkout: str | Path | None = None,
    lam_m: float = 0.2,
    lam_s: float = 1.0,
    lam_p: float = 0.2,
    quotient_so2: bool = True,
    grid_size: int = 8,
    refine: bool = False,
    refinement_tolerance: float = 1e-3,
    symmetrization: Symmetrization = "none",
    depth_policy: DepthPolicy = "raise",
    default_radius: float = 1.0,
) -> ElasticSRVFTResult:
    """Compute the upstream Elastic SRVFT energy for one rooted tree pair.

    The upstream implementation performs branch assignment and elastic curve
    reparameterization but no rotational alignment.  With ``quotient_so2=True``
    this adapter minimizes the energy over a relative rotation of ``tree_b``
    about z.  The default eight-angle grid is intentionally coarse because one
    upstream evaluation can take tens of seconds on full neurons; refinement is
    therefore opt-in and every approximation setting is returned in the result.

    The external representation is limited to four branch layers.  The default
    ``depth_policy='raise'`` prevents its original silent truncation behavior.
    """
    _validate_options(
        lam_m=lam_m,
        lam_s=lam_s,
        lam_p=lam_p,
        depth_policy=depth_policy,
        symmetrization=symmetrization,
        grid_size=grid_size,
        refinement_tolerance=refinement_tolerance,
    )
    started = time.perf_counter()
    api = _load_external_api(checkout)
    prepared_a = _prepare_external_tree(
        tree_a,
        api,
        name="tree_a",
        default_radius=default_radius,
        depth_policy=depth_policy,
    )
    prepared_b = _prepare_external_tree(
        tree_b,
        api,
        name="tree_b",
        default_radius=default_radius,
        depth_policy=depth_policy,
    )

    cache: dict[float, tuple[float, float, float | None]] = {}
    upstream_energy_evaluations = 0

    def checked_energy(alignment: dict[str, Any], *, direction: str) -> float:
        value = float(alignment["E"])
        if not math.isfinite(value):
            raise RuntimeError(
                f"Elastic SRVFT returned a non-finite {direction} energy."
            )
        if value < -_NEGATIVE_ENERGY_TOLERANCE:
            raise RuntimeError(
                f"Elastic SRVFT returned a negative {direction} energy ({value})."
            )
        return max(value, 0.0)

    def components(angle: float) -> tuple[float, float, float | None]:
        nonlocal upstream_energy_evaluations
        canonical = float(angle) % (2.0 * math.pi)
        key = round(canonical, 14)
        cached = cache.get(key)
        if cached is not None:
            return cached

        rotated_b = _rotate_qtree_copy(prepared_b.qtree, canonical)
        forward_alignment, _, _ = api.compute_distance_energy(
            copy.deepcopy(prepared_a.qtree),
            copy.deepcopy(rotated_b),
            lam_m=lam_m,
            lam_s=lam_s,
            lam_p=lam_p,
        )
        upstream_energy_evaluations += 1
        forward = checked_energy(forward_alignment, direction="forward")

        reverse: float | None = None
        value = forward
        if symmetrization == "mean":
            reverse_alignment, _, _ = api.compute_distance_energy(
                copy.deepcopy(rotated_b),
                copy.deepcopy(prepared_a.qtree),
                lam_m=lam_m,
                lam_s=lam_s,
                lam_p=lam_p,
            )
            upstream_energy_evaluations += 1
            reverse = checked_energy(reverse_alignment, direction="reverse")
            value = 0.5 * (forward + reverse)

        result = (value, forward, reverse)
        cache[key] = result
        return result

    zero_value, zero_forward, zero_reverse = components(0.0)
    if quotient_so2:
        minimum = minimize_over_so2(
            lambda angle: components(angle)[0],
            grid_size=grid_size,
            refine=refine,
            refinement_tolerance=refinement_tolerance,
        )
        angle_rad = float(minimum.angle_rad)
        grid_energy = float(minimum.grid_value)
        grid_angle_rad = float(minimum.grid_angle_rad)
        objective_evaluations = int(minimum.evaluations)
    else:
        angle_rad = 0.0
        grid_energy = float(zero_value)
        grid_angle_rad = 0.0
        objective_evaluations = 1

    energy, forward_energy, reverse_energy = components(angle_rad)
    return ElasticSRVFTResult(
        value=float(energy),
        energy=float(energy),
        forward_energy=float(forward_energy),
        reverse_energy=None if reverse_energy is None else float(reverse_energy),
        energy_at_zero_rotation=float(zero_value),
        forward_energy_at_zero_rotation=float(zero_forward),
        reverse_energy_at_zero_rotation=(
            None if zero_reverse is None else float(zero_reverse)
        ),
        angle_rad=angle_rad,
        grid_energy=grid_energy,
        grid_angle_rad=grid_angle_rad,
        quotient_so2=bool(quotient_so2),
        grid_size=int(grid_size),
        refine=bool(refine),
        refinement_tolerance=float(refinement_tolerance),
        objective_evaluations=objective_evaluations,
        upstream_energy_evaluations=upstream_energy_evaluations,
        runtime_seconds=float(time.perf_counter() - started),
        lam_m=float(lam_m),
        lam_s=float(lam_s),
        lam_p=float(lam_p),
        symmetrization=symmetrization,
        depth_policy=depth_policy,
        tree_a_nodes=prepared_a.node_count,
        tree_b_nodes=prepared_b.node_count,
        tree_a_terminal_leaves=prepared_a.terminal_leaf_count,
        tree_b_terminal_leaves=prepared_b.terminal_leaf_count,
        tree_a_represented_branches=prepared_a.represented_branch_count,
        tree_b_represented_branches=prepared_b.represented_branch_count,
        tree_a_omitted_frontier_branches=(
            prepared_a.omitted_frontier_branches
        ),
        tree_b_omitted_frontier_branches=(
            prepared_b.omitted_frontier_branches
        ),
        tree_a_canonical_order_ties=prepared_a.canonical_order_ties,
        tree_b_canonical_order_ties=prepared_b.canonical_order_ties,
        radius_used_in_energy=False,
        external_revision=api.revision,
        external_checkout=str(api.checkout),
    )


__all__ = [
    "DEFAULT_ELASTIC_SRVFT_CHECKOUT",
    "DepthPolicy",
    "ElasticSRVFTDependencyError",
    "ElasticSRVFTError",
    "ElasticSRVFTNotConfigured",
    "ElasticSRVFTResult",
    "ElasticSRVFTUnsupportedTree",
    "Symmetrization",
    "elastic_srvft_distance",
]
