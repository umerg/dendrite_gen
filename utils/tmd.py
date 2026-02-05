"""
tmd.py

Mixed-method TMD embedding utilities.

This module provides a single top-level API that can compute persistence-image
embeddings per-filtration using either:
  - 0D graph persistence (from tmd_conditioning_utils.py), or
  - paper-style TMD barcode (from tmd_paper_embedding_utils.py).

Density-profile embeddings are intentionally omitted.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import networkx as nx

try:
    # When imported as part of the utils package (repo root on sys.path)
    from utils.tmd_conditioning_utils import (  # type: ignore
        FiltrationName,
        PersistenceDiagram0D,
        assert_rooted_tree_graph,
        compute_0d_persistence_diagram,
        filtration_height_z,
        filtration_path_length_from_root,
        filtration_radial_rho,
        normalize_filtration_values,
        persistence_image,
    )
    from utils.tmd_paper_embedding_utils import (  # type: ignore
        barcode_to_diagram,
        build_critical_tree,
        compute_tmd_barcode,
        root_undirected_tree,
    )
except ModuleNotFoundError:
    # Fallback for direct module execution/import when utils is already on sys.path
    from tmd_conditioning_utils import (  # type: ignore
        FiltrationName,
        PersistenceDiagram0D,
        assert_rooted_tree_graph,
        compute_0d_persistence_diagram,
        filtration_height_z,
        filtration_path_length_from_root,
        filtration_radial_rho,
        normalize_filtration_values,
        persistence_image,
    )
    from tmd_paper_embedding_utils import (  # type: ignore
        barcode_to_diagram,
        build_critical_tree,
        compute_tmd_barcode,
        root_undirected_tree,
    )

MethodName = Literal["0d", "tmd"]


def compute_tmd_barcode_diagram(
    G: nx.Graph,
    *,
    filtration: FiltrationName = "path",
    normalize_mode: Literal["minmax", "max", "none"] = "minmax",
    weight_edges_by_euclidean: bool = True,
    simplify_to_critical_tree: bool = True,
) -> tuple[np.ndarray, PersistenceDiagram0D]:
    """
    Compute a paper-style TMD barcode and its canonicalized persistence diagram.

    Returns:
        barcode: (M,2) float array of (birth, death) pairs (raw TMD intervals)
        diagram: PersistenceDiagram0D with birth <= death for each interval
    """
    if G.number_of_nodes() == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return np.zeros((0, 2), dtype=np.float64), PersistenceDiagram0D(births=empty, deaths=empty)

    assert_rooted_tree_graph(G)

    if filtration == "path":
        f_full = filtration_path_length_from_root(
            G, weight_edges_by_euclidean=weight_edges_by_euclidean
        )
    elif filtration == "height":
        f_full = filtration_height_z(G)
    elif filtration == "rho":
        f_full = filtration_radial_rho(G)
    else:
        raise ValueError(f"Unknown filtration name: {filtration!r}")

    if normalize_mode != "none":
        f_full = normalize_filtration_values(f_full, mode=normalize_mode)

    root = G.graph["root"]
    rooted = root_undirected_tree(G, root)
    rooted_use = build_critical_tree(rooted) if simplify_to_critical_tree else rooted
    f = {nid: float(f_full[nid]) for nid in rooted_use.children.keys()}

    barcode = compute_tmd_barcode(rooted_use, f)
    diagram = barcode_to_diagram(barcode)
    return barcode, diagram


def _resolve_method_map(
    filtrations: Sequence[FiltrationName],
    method_by_filtration: Optional[
        Union[Dict[FiltrationName, MethodName], Sequence[MethodName]]
    ],
) -> Dict[FiltrationName, MethodName]:
    default_map: Dict[FiltrationName, MethodName] = {
        "path": "tmd",
        "height": "0d",
        "rho": "0d",
    }

    if method_by_filtration is None:
        method_map = dict(default_map)
    elif isinstance(method_by_filtration, (list, tuple)):
        if len(method_by_filtration) != len(filtrations):
            raise ValueError(
                "method_by_filtration sequence must match length of filtrations."
            )
        method_map = dict(default_map)
        for name, method in zip(filtrations, method_by_filtration):
            method_map[name] = method
    elif isinstance(method_by_filtration, dict):
        method_map = dict(default_map)
        method_map.update(method_by_filtration)
    else:
        raise TypeError(
            "method_by_filtration must be a dict or sequence of method names."
        )

    for name in filtrations:
        method = method_map.get(name)
        if method not in ("0d", "tmd"):
            raise ValueError(
                f"Invalid method for filtration {name!r}: {method!r}. "
                "Expected '0d' or 'tmd'."
            )

    return method_map


def compute_tmd_mixed(
    G: nx.Graph,
    *,
    filtrations: Sequence[FiltrationName] = ("path", "height", "rho"),
    method_by_filtration: Optional[
        Union[Dict[FiltrationName, MethodName], Sequence[MethodName]]
    ] = None,
    n_bins: int = 16,
    sigma: float = 0.05,
    normalize_mode: Literal["minmax", "max"] = "minmax",
    weighting: Literal["none", "persistence"] = "persistence",
    weight_edges_by_euclidean: bool = True,
    simplify_to_critical_tree: bool = True,
) -> np.ndarray:
    """
    Compute a concatenated global embedding using mixed methods per filtration.

    By default:
        - "path" uses paper-style TMD barcode ("tmd")
        - "height" and "rho" use 0D graph persistence ("0d")

    method_by_filtration can be:
        - dict mapping filtration -> "0d" or "tmd"
        - sequence of method names aligned with `filtrations`

    Returns:
        e: np.ndarray shape (len(filtrations) * n_bins * n_bins,), dtype float32
    """
    assert_rooted_tree_graph(G)

    method_map = _resolve_method_map(filtrations, method_by_filtration)
    needs_tmd = any(method_map[name] == "tmd" for name in filtrations)

    if needs_tmd:
        root = G.graph["root"]
        rooted = root_undirected_tree(G, root)
        rooted_use = build_critical_tree(rooted) if simplify_to_critical_tree else rooted
    else:
        rooted_use = None

    emb_list: List[np.ndarray] = []

    for name in filtrations:
        if name == "path":
            f_full = filtration_path_length_from_root(
                G, weight_edges_by_euclidean=weight_edges_by_euclidean
            )
        elif name == "height":
            f_full = filtration_height_z(G)
        elif name == "rho":
            f_full = filtration_radial_rho(G)
        else:
            raise ValueError(f"Unknown filtration name: {name!r}")

        f_full = normalize_filtration_values(f_full, mode=normalize_mode)

        if method_map[name] == "0d":
            diag = compute_0d_persistence_diagram(
                G, f_full, include_infinite_bar=False
            )
        elif method_map[name] == "tmd":
            if rooted_use is None:
                raise RuntimeError("Internal error: rooted tree not initialized.")
            f = {nid: float(f_full[nid]) for nid in rooted_use.children.keys()}
            barcode = compute_tmd_barcode(rooted_use, f)
            diag = barcode_to_diagram(barcode)
        else:
            raise ValueError(f"Unknown method: {method_map[name]!r}")

        pi = persistence_image(
            diag,
            n_bins=n_bins,
            sigma=sigma,
            birth_range=(0.0, 1.0),
            pers_range=(0.0, 1.0),
            weighting=weighting,
        )
        emb_list.append(pi.astype(np.float32))

    return np.concatenate(emb_list, axis=0).astype(np.float32)
