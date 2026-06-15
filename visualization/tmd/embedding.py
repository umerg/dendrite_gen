"""Persistence-image embedding helpers for TMD visualizations."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from dendrite_gen.utils.tmd_conditioning_utils import persistence_image


ReducerName = Literal["auto", "umap", "pca"]
WeightingName = Literal["none", "persistence"]


@dataclass(frozen=True)
class TmdEmbeddingRecord:
    """One tree represented by a concatenated persistence-image vector."""

    source: str
    pair_index: int
    tree_name: str
    vector: np.ndarray
    attributes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TmdDiagramRecord:
    """One tree represented by persistence diagrams and tree attributes."""

    source: str
    pair_index: int
    tree_name: str
    diagrams: Mapping[str, object]
    attributes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TmdEmbeddingResult:
    """Reduced TMD embedding coordinates with plotting metadata."""

    records: list[TmdEmbeddingRecord]
    coords: np.ndarray
    reducer: str


@dataclass(frozen=True)
class TmdPairDistanceRecord:
    """One GT/pred pair summarized by diagram distance and a GT attribute."""

    pair_index: int
    tree_name: str
    embedding_name: str
    distance_metric: str
    distance: float
    attribute_name: str
    attribute_value: float


def _diagram_pairs(diagram: object) -> np.ndarray:
    if diagram is None:
        return np.zeros((0, 2), dtype=np.float64)
    pairs = diagram.as_pairs()
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pairs, dtype=np.float64)


def persistence_image_ranges(
    diagram_sets: Sequence[Mapping[str, object]],
    *,
    filtrations: Sequence[str],
    normalize_mode: str,
) -> dict[str, tuple[float, float]]:
    """Return shared persistence-image ranges for a set of diagrams."""
    if normalize_mode == "minmax":
        return {"birth": (0.0, 1.0), "persistence": (0.0, 1.0)}

    births = []
    persistences = []
    for diagrams in diagram_sets:
        for filtration in filtrations:
            pairs = _diagram_pairs(diagrams.get(filtration))
            if pairs.size == 0:
                continue
            persistence = pairs[:, 1] - pairs[:, 0]
            keep = np.isfinite(pairs[:, 0]) & np.isfinite(persistence) & (persistence > 1e-12)
            if not np.any(keep):
                continue
            births.append(pairs[keep, 0])
            persistences.append(persistence[keep])

    if births:
        birth_vals = np.concatenate(births, axis=0)
        birth_lo = float(np.min(birth_vals))
        birth_hi = float(np.max(birth_vals))
    else:
        birth_lo, birth_hi = 0.0, 1.0

    if persistences:
        persistence_vals = np.concatenate(persistences, axis=0)
        persistence_lo = 0.0
        persistence_hi = float(np.max(persistence_vals))
    else:
        persistence_lo, persistence_hi = 0.0, 1.0

    if birth_hi <= birth_lo:
        birth_hi = birth_lo + 1e-6
    if persistence_hi <= persistence_lo:
        persistence_hi = persistence_lo + 1e-6

    return {
        "birth": (birth_lo, birth_hi),
        "persistence": (persistence_lo, persistence_hi),
    }


def diagrams_to_persistence_image_vector(
    diagrams: Mapping[str, object],
    *,
    filtrations: Sequence[str],
    n_bins: int = 16,
    sigma: float = 0.05,
    weighting: WeightingName = "persistence",
    birth_range: tuple[float, float] = (0.0, 1.0),
    persistence_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Vectorize one tree's diagrams as concatenated persistence images."""
    vectors = []
    for filtration in filtrations:
        diagram = diagrams.get(filtration)
        if diagram is None:
            vectors.append(np.zeros((n_bins * n_bins,), dtype=np.float32))
            continue
        vectors.append(
            persistence_image(
                diagram,
                n_bins=n_bins,
                sigma=sigma,
                birth_range=birth_range,
                pers_range=persistence_range,
                weighting=weighting,
            )
        )
    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vectors, axis=0).astype(np.float32)


def _standardize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    mean = vectors.mean(axis=0, keepdims=True)
    std = vectors.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (vectors - mean) / std


def _reduce_with_pca(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if vectors.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.zeros((vectors.shape[0], 2), dtype=np.float64)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, vt.shape[0])
    coords = centered @ vt[:n_components].T
    if n_components == 1:
        coords = np.column_stack([coords[:, 0], np.zeros((coords.shape[0],), dtype=np.float64)])
    return coords.astype(np.float64)


def reduce_tmd_embedding_records(
    records: Sequence[TmdEmbeddingRecord],
    *,
    reducer: ReducerName = "auto",
    random_state: int = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> TmdEmbeddingResult:
    """Reduce TMD embedding vectors to two dimensions."""
    if not records:
        return TmdEmbeddingResult(records=[], coords=np.zeros((0, 2), dtype=np.float64), reducer="none")

    vectors = np.stack([record.vector for record in records], axis=0)
    vectors = _standardize_vectors(vectors)

    if reducer == "umap" and vectors.shape[0] < 3:
        raise RuntimeError("UMAP reducer requires at least three embedded trees.")

    use_umap = reducer in {"auto", "umap"} and vectors.shape[0] >= 3
    if use_umap:
        try:
            import umap  # type: ignore

            n_neighbors = min(max(2, umap_n_neighbors), vectors.shape[0] - 1)
            coords = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=umap_min_dist,
                metric="euclidean",
                random_state=random_state,
            ).fit_transform(vectors)
            return TmdEmbeddingResult(records=list(records), coords=np.asarray(coords), reducer="umap")
        except Exception as exc:
            if reducer == "umap":
                raise RuntimeError("UMAP was requested, but the UMAP reducer could not run.") from exc

    coords = _reduce_with_pca(vectors)
    return TmdEmbeddingResult(records=list(records), coords=coords, reducer="pca")


def persistence_diagram_wasserstein_distance(
    diagram_a: object,
    diagram_b: object,
    *,
    order: int = 1,
) -> float:
    """Compute a persistence-diagram Wasserstein distance with diagonal matching."""
    if order < 1:
        raise ValueError("Wasserstein order must be >= 1.")

    pairs_a = _canonical_persistent_pairs(_diagram_pairs(diagram_a))
    pairs_b = _canonical_persistent_pairs(_diagram_pairs(diagram_b))
    n_a = pairs_a.shape[0]
    n_b = pairs_b.shape[0]

    if n_a == 0 and n_b == 0:
        return 0.0
    if n_a == 0:
        return float(np.sum(_diagonal_distances(pairs_b) ** order) ** (1.0 / order))
    if n_b == 0:
        return float(np.sum(_diagonal_distances(pairs_a) ** order) ** (1.0 / order))

    from scipy.optimize import linear_sum_assignment

    cost = np.zeros((n_a + n_b, n_b + n_a), dtype=np.float64)
    cost[:n_a, :n_b] = _pairwise_l2_distances(pairs_a, pairs_b) ** order
    cost[:n_a, n_b:] = np.repeat(
        (_diagonal_distances(pairs_a) ** order)[:, None],
        n_a,
        axis=1,
    )
    cost[n_a:, :n_b] = np.repeat(
        (_diagonal_distances(pairs_b) ** order)[None, :],
        n_b,
        axis=0,
    )

    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    return float(total ** (1.0 / order))


def pair_persistence_diagram_distances(
    records: Sequence[TmdDiagramRecord],
    *,
    attribute: str,
    embedding_name: str,
    filtrations: Sequence[str],
    distance_metric: Literal["wasserstein"] = "wasserstein",
    wasserstein_order: int = 1,
) -> list[TmdPairDistanceRecord]:
    """Return one GT-vs-pred persistence-diagram distance per paired tree."""
    if distance_metric != "wasserstein":
        raise ValueError(f"Unsupported distance metric: {distance_metric!r}")

    by_pair: dict[int, dict[str, TmdDiagramRecord]] = {}
    for record in records:
        by_pair.setdefault(int(record.pair_index), {})[str(record.source)] = record

    pair_records: list[TmdPairDistanceRecord] = []
    for pair_index in sorted(by_pair):
        pair = by_pair[pair_index]
        if "gt" not in pair or "pred" not in pair:
            continue

        gt = pair["gt"]
        pred = pair["pred"]

        distance = 0.0
        for filtration in filtrations:
            distance += persistence_diagram_wasserstein_distance(
                gt.diagrams.get(filtration),
                pred.diagrams.get(filtration),
                order=wasserstein_order,
            )

        try:
            attribute_value = float(gt.attributes.get(attribute, np.nan))
        except (TypeError, ValueError):
            attribute_value = float("nan")

        pair_records.append(
            TmdPairDistanceRecord(
                pair_index=pair_index,
                tree_name=gt.tree_name,
                embedding_name=embedding_name,
                distance_metric=distance_metric,
                distance=float(distance),
                attribute_name=attribute,
                attribute_value=attribute_value,
            )
        )
    return pair_records


def _canonical_persistent_pairs(pairs: np.ndarray) -> np.ndarray:
    """Return finite (birth, death) pairs with positive persistence."""
    pairs = np.asarray(pairs, dtype=np.float64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pairs = pairs.reshape(-1, 2)
    finite = np.isfinite(pairs).all(axis=1)
    pairs = pairs[finite]
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    pairs = np.stack([lo, hi], axis=1)
    persistent = (pairs[:, 1] - pairs[:, 0]) > 1e-12
    return pairs[persistent]


def _diagonal_distances(pairs: np.ndarray) -> np.ndarray:
    """Euclidean distance from each birth/death point to the diagonal."""
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.abs(pairs[:, 1] - pairs[:, 0]) / np.sqrt(2.0)


def _pairwise_l2_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def write_tmd_embedding_points_csv(result: TmdEmbeddingResult, out_path: Path) -> Path:
    """Write reduced TMD embedding coordinates and metadata to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attribute_names = sorted(
        {
            attribute
            for record in result.records
            for attribute in getattr(record, "attributes", {}).keys()
        }
    )
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "pair_index", "tree_name", "x", "y", "reducer", *attribute_names])
        for record, coord in zip(result.records, result.coords):
            writer.writerow(
                [
                    record.source,
                    record.pair_index,
                    record.tree_name,
                    float(coord[0]),
                    float(coord[1]),
                    result.reducer,
                    *[
                        float(record.attributes[attribute])
                        if attribute in record.attributes
                        else ""
                        for attribute in attribute_names
                    ],
                ]
            )
    return out_path
