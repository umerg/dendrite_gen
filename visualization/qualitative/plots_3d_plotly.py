"""Plotly-based interactive 3D cylinder rendering helpers for tree graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots_3d import BRANCH_COLOR, compute_node_radii
from .plotly_defaults import (
    DEFAULT_PLOTLY_LEAF_COUNT,
    DEFAULT_PLOTLY_LEAF_OPACITY,
    DEFAULT_PLOTLY_LEAF_SCALE,
    DEFAULT_PLOTLY_LEAF_SEED,
    DEFAULT_PLOTLY_LEAVES,
    DEFAULT_PLOTLY_TEXTURE,
    DEFAULT_PLOTLY_TEXTURE_MAX_AXIAL_SEGMENTS,
    DEFAULT_PLOTLY_TEXTURE_STRENGTH,
    DEFAULT_PLOTLY_TEXTURE_TARGET_LENGTH_SCALE,
)

if TYPE_CHECKING:
    import networkx as nx


def _pos_to_xyz(pos: object) -> np.ndarray:
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _graph_positions(graph: "nx.Graph") -> dict[int, np.ndarray]:
    return {
        node: _pos_to_xyz(graph.nodes[node].get("pos", np.zeros(3)))
        for node in graph.nodes
    }


def _perpendicular_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(direction, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    axis = axis / axis_norm
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(axis, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return u, v


BARK_COLORSCALE = [
    [0.0, "#2d1c0f"],
    [0.25, "#5a371b"],
    [0.5, "#8a5a2b"],
    [0.75, "#b77738"],
    [1.0, "#e09946"],
]

LEAF_COLORSCALE = [
    [0.0, "#405f2d"],
    [0.35, "#5f7f3f"],
    [0.7, "#7f9b55"],
    [1.0, "#a6b66f"],
]
LEAF_TRACE_NAME = "Low-poly leaves"
LEAF_OPACITY_SLIDER_VALUES = tuple(round(value, 2) for value in np.linspace(0.0, 0.85, 18))

LEAF_POLYHEDRON_VERTICES = np.array(
    [
        [-1.0, 1.61803398875, 0.0],
        [1.0, 1.61803398875, 0.0],
        [-1.0, -1.61803398875, 0.0],
        [1.0, -1.61803398875, 0.0],
        [0.0, -1.0, 1.61803398875],
        [0.0, 1.0, 1.61803398875],
        [0.0, -1.0, -1.61803398875],
        [0.0, 1.0, -1.61803398875],
        [1.61803398875, 0.0, -1.0],
        [1.61803398875, 0.0, 1.0],
        [-1.61803398875, 0.0, -1.0],
        [-1.61803398875, 0.0, 1.0],
    ],
    dtype=float,
)
LEAF_POLYHEDRON_VERTICES /= np.linalg.norm(
    LEAF_POLYHEDRON_VERTICES,
    axis=1,
    keepdims=True,
)
LEAF_POLYHEDRON_FACES = [
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
]


def _hex_to_rgb01(color: str) -> np.ndarray:
    value = color.strip()
    if value.startswith("#") and len(value) == 7:
        try:
            return np.array(
                [
                    int(value[1:3], 16),
                    int(value[3:5], 16),
                    int(value[5:7], 16),
                ],
                dtype=float,
            ) / 255.0
        except ValueError:
            pass
    return np.array([0.54, 0.35, 0.17], dtype=float)


def _rgb01_to_hex(rgb: np.ndarray) -> str:
    rgb255 = np.clip(np.rint(np.asarray(rgb, dtype=float) * 255.0), 0, 255).astype(int)
    return f"#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}"


def _bark_colorscale(texture_strength: float) -> list[list[float | str]]:
    strength = float(np.clip(texture_strength, 0.0, 1.0))
    base_rgb = _hex_to_rgb01("#8a5a2b")
    colorscale: list[list[float | str]] = []
    for stop, color in BARK_COLORSCALE:
        rgb = _hex_to_rgb01(str(color))
        colorscale.append([float(stop), _rgb01_to_hex(base_rgb + strength * (rgb - base_rgb))])
    return colorscale


def _hash01(*values: float) -> float:
    x = 0.0
    for idx, value in enumerate(values):
        x += float(value) * (12.9898 + 37.719 * idx)
    hashed = np.sin(x) * 43758.5453123
    return float(hashed - np.floor(hashed))


def _bark_face_value(
    *,
    edge_index: int,
    face_index: int,
    theta: float,
    axial: float,
    texture_strength: float,
) -> float:
    if texture_strength <= 0.0:
        return 0.5

    phase = 0.71 * (edge_index + 1)
    axial_wave = np.sin(2.0 * np.pi * (1.35 * axial + 0.09 * edge_index))
    broad_streak = np.sin(4.0 * theta + phase + 0.45 * axial_wave)
    fine_streak = np.sin(10.0 * theta + 0.37 * phase + 0.7 * axial)
    hairline = np.sin(18.0 * theta + 1.91 * phase + 0.9 * axial_wave)
    longitudinal_grain = np.sin(2.0 * np.pi * (3.2 * axial + 0.13 * edge_index))
    noise = _hash01(edge_index + 1, face_index + 1) - 0.5
    warm_noise = _hash01(edge_index + 11, face_index + 29) - 0.5

    crack = max(0.0, hairline - 0.58) / 0.42
    value = 0.52 + (
        0.30 * broad_streak
        + 0.18 * fine_streak
        + 0.22 * longitudinal_grain
        + 0.24 * noise
        + 0.10 * warm_noise
        - 0.48 * crack
    )
    return float(np.clip(value, 0.0, 1.0))


def _texture_axial_segment_count(
    *,
    length: float,
    texture: str,
    texture_strength: float,
    texture_target_length: float,
    texture_max_axial_segments: int,
) -> int:
    if texture != "bark" or texture_strength <= 0.0:
        return 1
    target_length = max(float(texture_target_length), 1e-9)
    adaptive_count = int(np.ceil(max(float(length), 0.0) / target_length))
    return max(1, min(max(int(texture_max_axial_segments), 1), adaptive_count))


def _unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        fallback_arr = np.asarray(fallback, dtype=float)
        fallback_norm = float(np.linalg.norm(fallback_arr))
        if fallback_norm <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return fallback_arr / fallback_norm
    return arr / norm


def _resolve_leaf_root(graph: "nx.Graph", pos: dict[int, np.ndarray]) -> int | None:
    graph_root = graph.graph.get("root")
    if graph_root in graph and graph_root in pos:
        return graph_root
    if 0 in graph and 0 in pos:
        return 0
    if 1 in graph and 1 in pos:
        return 1
    if pos:
        return min(pos, key=lambda node: (float(pos[node][2]), str(node)))
    return None


def _graph_path_distances(graph: "nx.Graph", pos: dict[int, np.ndarray]) -> dict[int, float]:
    distances: dict[int, float] = {}
    root = _resolve_leaf_root(graph, pos)
    starts = []
    if root is not None:
        starts.append(root)
    starts.extend(sorted((node for node in graph.nodes if node in pos), key=str))

    for start in starts:
        if start in distances or start not in pos:
            continue
        distances[start] = 0.0
        queue = [start]
        for node in queue:
            for neighbor in sorted(graph.neighbors(node), key=str):
                if neighbor in distances or neighbor not in pos:
                    continue
                edge_length = float(np.linalg.norm(pos[node] - pos[neighbor]))
                distances[neighbor] = distances[node] + max(edge_length, 1e-12)
                queue.append(neighbor)
    return distances


def _leaf_tip_nodes(graph: "nx.Graph", pos: dict[int, np.ndarray]) -> list[int]:
    root = _resolve_leaf_root(graph, pos)
    return sorted(
        [
            node
            for node in graph.nodes
            if node in pos and node != root and graph.degree[node] <= 1
        ],
        key=str,
    )


def _leaf_anchor_nodes(
    graph: "nx.Graph",
    pos: dict[int, np.ndarray],
    *,
    leaf_count: int,
    seed: int,
) -> list[int]:
    requested_count = max(int(leaf_count), 0)
    if requested_count <= 0:
        return []

    nodes = [node for node in graph.nodes if node in pos]
    if not nodes:
        return []

    tips = _leaf_tip_nodes(graph, pos)
    if tips:
        return tips

    all_points = np.stack([pos[node] for node in nodes], axis=0)
    z_min = float(np.min(all_points[:, 2]))
    z_range = max(float(np.ptp(all_points[:, 2])), 1e-12)
    distances = _graph_path_distances(graph, pos)
    max_distance = max([float(value) for value in distances.values()] or [1.0])
    max_distance = max(max_distance, 1e-12)
    root = _resolve_leaf_root(graph, pos)

    scored: list[tuple[float, int]] = []
    for node_index, node in enumerate(nodes):
        if node == root and len(nodes) > 1:
            continue
        dist_norm = float(distances.get(node, 0.0)) / max_distance
        height_norm = (float(pos[node][2]) - z_min) / z_range
        tip_bonus = 0.20 if graph.degree[node] <= 1 else 0.0
        branch_bonus = 0.08 if graph.degree[node] >= 3 else 0.0
        jitter = 0.10 * _hash01(seed, node_index + 1, dist_norm, height_norm)
        score = 0.52 * dist_norm + 0.34 * height_norm + tip_bonus + branch_bonus + jitter
        scored.append((float(score), node))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    candidate_count = min(len(scored), max(requested_count * 5, requested_count, 12))
    candidates = [node for _, node in scored[:candidate_count]]

    anchors: list[int] = []
    for extra_index in range(requested_count):
        draw = _hash01(seed, extra_index + 1, 91.0)
        candidate_index = min(int((draw**1.7) * len(candidates)), len(candidates) - 1)
        anchors.append(candidates[candidate_index])
    return anchors


def _leaf_anchor_groups(
    anchors: list[int],
    pos: dict[int, np.ndarray],
    *,
    blob_count: int,
    seed: int,
) -> list[list[int]]:
    if not anchors:
        return []

    requested_count = max(int(blob_count), 1)
    if requested_count >= len(anchors):
        groups = [[anchor] for anchor in anchors]
        for extra_index in range(requested_count - len(anchors)):
            anchor_index = min(
                int(_hash01(seed, extra_index + 1, 307.0) * len(anchors)),
                len(anchors) - 1,
            )
            groups.append([anchors[anchor_index]])
        return groups

    anchor_points = np.stack([pos[anchor] for anchor in anchors], axis=0)
    selected_indices = [
        int(
            np.argmax(
                anchor_points[:, 2]
                + np.asarray(
                    [
                        0.01 * _hash01(seed, idx + 1, 311.0)
                        for idx in range(len(anchors))
                    ],
                    dtype=float,
                )
            )
        )
    ]
    while len(selected_indices) < requested_count:
        selected_points = anchor_points[selected_indices]
        distances = np.linalg.norm(
            anchor_points[:, None, :] - selected_points[None, :, :],
            axis=2,
        )
        min_distances = np.min(distances, axis=1)
        min_distances[selected_indices] = -1.0
        jitter = np.asarray(
            [
                0.08 * _hash01(seed, len(selected_indices) + 1, idx + 1, 313.0)
                for idx in range(len(anchors))
            ],
            dtype=float,
        )
        selected_indices.append(int(np.argmax(min_distances * (1.0 + jitter))))

    groups: list[list[int]] = [[] for _ in selected_indices]
    coverage = np.zeros(len(anchors), dtype=bool)
    selected_points = anchor_points[selected_indices]
    for anchor_index, anchor in enumerate(anchors):
        distances = np.linalg.norm(selected_points - anchor_points[anchor_index], axis=1)
        group_index = int(np.argmin(distances))
        groups[group_index].append(anchor)
        coverage[anchor_index] = True

    if not np.all(coverage):
        for anchor_index, covered in enumerate(coverage):
            if covered:
                continue
            distances = np.linalg.norm(selected_points - anchor_points[anchor_index], axis=1)
            groups[int(np.argmin(distances))].append(anchors[anchor_index])

    return [group for group in groups if group]


def _leaf_cluster_points(
    graph: "nx.Graph",
    pos: dict[int, np.ndarray],
    *,
    anchors: list[int],
    cluster_index: int,
    seed: int,
) -> np.ndarray:
    valid_anchors = [anchor for anchor in anchors if anchor in pos]
    if not valid_anchors:
        return np.zeros((0, 3), dtype=float)

    hop_limit = 5 + int(7 * _hash01(seed, cluster_index + 1, 17.0))
    seen = set(valid_anchors)
    queue = [(anchor, 0) for anchor in valid_anchors]
    cluster_nodes: list[int] = []
    for node, depth in queue:
        if node in pos:
            cluster_nodes.append(node)
        if depth >= hop_limit:
            continue
        neighbors = [neighbor for neighbor in sorted(graph.neighbors(node), key=str) if neighbor in pos]
        for neighbor_index, neighbor in enumerate(neighbors):
            if neighbor in seen:
                continue
            grow_probability = 0.88 if depth == 0 else 0.78 - 0.045 * depth
            if graph.degree[neighbor] != 2:
                grow_probability += 0.10
            grow_probability = float(np.clip(grow_probability, 0.28, 0.94))
            if depth == 0 or _hash01(seed, cluster_index + 1, depth + 5, neighbor_index + 3) <= grow_probability:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))

    if len(cluster_nodes) < 2:
        for anchor in valid_anchors:
            for neighbor in sorted(graph.neighbors(anchor), key=str):
                if neighbor in pos and neighbor not in cluster_nodes:
                    cluster_nodes.append(neighbor)
                if len(cluster_nodes) >= 2:
                    break
            if len(cluster_nodes) >= 2:
                break

    return np.stack([pos[node] for node in cluster_nodes], axis=0)


def _principal_axis(points: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return _unit_vector(fallback, np.array([0.0, 0.0, 1.0], dtype=float))
    centered = points - np.mean(points, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return _unit_vector(fallback, np.array([0.0, 0.0, 1.0], dtype=float))
    return _unit_vector(vh[0], fallback)


def _leaf_cluster_frame(
    cluster_points: np.ndarray,
    outward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_axis = _unit_vector(outward, np.array([0.0, 0.0, 1.0], dtype=float))
    branch_axis = _principal_axis(cluster_points, np.array([1.0, 0.0, 0.0], dtype=float))
    branch_axis = branch_axis - float(np.dot(branch_axis, normal_axis)) * normal_axis
    if float(np.linalg.norm(branch_axis)) <= 1e-12:
        branch_axis, side_axis = _perpendicular_frame(normal_axis)
    else:
        branch_axis = _unit_vector(branch_axis, np.array([1.0, 0.0, 0.0], dtype=float))
        side_axis = _unit_vector(
            np.cross(normal_axis, branch_axis),
            np.array([0.0, 1.0, 0.0], dtype=float),
        )
    normal_axis = _unit_vector(np.cross(branch_axis, side_axis), normal_axis)
    if float(np.dot(normal_axis, outward)) < 0.0:
        normal_axis = -normal_axis
    return branch_axis, side_axis, normal_axis


def _append_leaf_polyhedron_mesh(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    facevalues: list[float],
    *,
    center: np.ndarray,
    basis: np.ndarray,
    axis_lengths: np.ndarray,
    polyhedron_index: int,
    seed: int,
) -> None:
    axis_lengths = np.maximum(np.asarray(axis_lengths, dtype=float), 1e-12)
    offset = len(vertices)
    for vertex_index, unit_vertex in enumerate(LEAF_POLYHEDRON_VERTICES):
        radial_jitter = 0.78 + 0.36 * _hash01(seed, polyhedron_index + 1, vertex_index + 31)
        local = np.asarray(unit_vertex, dtype=float) * axis_lengths * radial_jitter
        vertices.append(center + basis @ local)

    for face_index, face in enumerate(LEAF_POLYHEDRON_FACES):
        triangles.append(
            (
                offset + int(face[0]),
                offset + int(face[1]),
                offset + int(face[2]),
            )
        )
        facevalues.append(
            float(0.18 + 0.74 * _hash01(seed, polyhedron_index + 1, face_index + 101))
        )


def _append_leaf_cluster_mesh(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    facevalues: list[float],
    *,
    cluster_points: np.ndarray,
    crown_center: np.ndarray,
    bbox_diag: float,
    cluster_index: int,
    leaf_scale: float,
    seed: int,
) -> None:
    if cluster_points.size == 0:
        return

    cluster_center = np.mean(cluster_points, axis=0)
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    spread = float(np.percentile(distances, 75)) if len(distances) else 0.0
    scale = max(float(leaf_scale), 0.0)
    min_radius = bbox_diag * 0.034 * scale
    max_radius = bbox_diag * 0.155 * scale
    base_radius = float(np.clip(spread * 2.25 + min_radius, min_radius, max_radius))
    if base_radius <= 0.0:
        return

    outward = _unit_vector(
        cluster_center - crown_center + np.array([0.0, 0.0, 0.18 * bbox_diag], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    branch_axis, side_axis, normal_axis = _leaf_cluster_frame(cluster_points, outward)
    basis = np.stack([branch_axis, side_axis, normal_axis], axis=1)
    local_points = (cluster_points - cluster_center) @ basis
    local_extent = (
        0.5 * np.ptp(local_points, axis=0)
        if len(local_points) > 1
        else np.zeros(3, dtype=float)
    )
    frame_u, frame_v = _perpendicular_frame(outward)

    axis_floor = np.array([1.42, 1.12, 0.92], dtype=float) * base_radius
    axis_lengths = np.maximum(
        axis_floor,
        np.array([0.88, 0.76, 0.62], dtype=float) * local_extent
        + np.array([0.64, 0.54, 0.44], dtype=float) * base_radius,
    )
    axis_lengths = np.minimum(
        axis_lengths,
        np.array([0.32, 0.27, 0.22], dtype=float) * bbox_diag * max(float(leaf_scale), 0.0),
    )

    point_index = int(
        min(
            _hash01(seed, cluster_index + 1, 223.0) * len(cluster_points),
            len(cluster_points) - 1,
        )
    )
    anchor_point = cluster_points[point_index]
    radial_angle = 2.0 * np.pi * _hash01(seed, cluster_index + 1, 227.0)
    radial_offset = (
        (0.10 + 0.26 * _hash01(seed, cluster_index + 1, 229.0))
        * base_radius
        * (np.cos(radial_angle) * frame_u + np.sin(radial_angle) * frame_v)
    )
    lift = (0.18 + 0.26 * _hash01(seed, cluster_index + 1, 233.0)) * base_radius
    polyhedron_center = (
        0.66 * cluster_center
        + 0.34 * anchor_point
        + radial_offset
        + lift * outward
    )

    rotation = 2.0 * np.pi * _hash01(seed, cluster_index + 1, 239.0)
    rotated_basis = np.stack(
        [
            np.cos(rotation) * branch_axis + np.sin(rotation) * side_axis,
            -np.sin(rotation) * branch_axis + np.cos(rotation) * side_axis,
            normal_axis,
        ],
        axis=1,
    )
    scale_jitter = np.array(
        [
            0.86 + 0.30 * _hash01(seed, cluster_index + 1, 251.0),
            0.82 + 0.34 * _hash01(seed, cluster_index + 1, 257.0),
            0.78 + 0.30 * _hash01(seed, cluster_index + 1, 263.0),
        ],
        dtype=float,
    )
    _append_leaf_polyhedron_mesh(
        vertices,
        triangles,
        facevalues,
        center=polyhedron_center,
        basis=rotated_basis,
        axis_lengths=axis_lengths * scale_jitter,
        polyhedron_index=cluster_index,
        seed=seed,
    )


def _leaf_mesh_trace(
    graph: "nx.Graph",
    pos: dict[int, np.ndarray],
    *,
    leaf_count: int,
    leaf_opacity: float,
    leaf_scale: float,
    leaf_seed: int,
) -> go.Mesh3d | None:
    anchors = _leaf_anchor_nodes(
        graph,
        pos,
        leaf_count=leaf_count,
        seed=leaf_seed,
    )
    if not anchors:
        return None

    all_points = np.stack(list(pos.values()), axis=0)
    bbox_diag = float(np.linalg.norm(np.ptp(all_points, axis=0)))
    if bbox_diag <= 1e-12:
        bbox_diag = 1.0
    anchor_points = np.stack([pos[node] for node in anchors], axis=0)
    crown_center = np.mean(anchor_points, axis=0)
    anchor_groups = _leaf_anchor_groups(
        anchors,
        pos,
        blob_count=leaf_count,
        seed=leaf_seed,
    )

    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    facevalues: list[float] = []
    for cluster_index, anchor_group in enumerate(anchor_groups):
        cluster_points = _leaf_cluster_points(
            graph,
            pos,
            anchors=anchor_group,
            cluster_index=cluster_index,
            seed=leaf_seed,
        )
        _append_leaf_cluster_mesh(
            vertices,
            triangles,
            facevalues,
            cluster_points=cluster_points,
            crown_center=crown_center,
            bbox_diag=bbox_diag,
            cluster_index=cluster_index,
            leaf_scale=leaf_scale,
            seed=leaf_seed,
        )

    if not vertices or not triangles:
        return None

    pts = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=int)
    return go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=facevalues,
        intensitymode="cell",
        colorscale=LEAF_COLORSCALE,
        cmin=0.0,
        cmax=1.0,
        opacity=float(np.clip(leaf_opacity, 0.0, 1.0)),
        name=LEAF_TRACE_NAME,
        flatshading=True,
        lighting=dict(ambient=0.72, diffuse=0.54, specular=0.05, roughness=0.95),
        lightposition=dict(x=100, y=200, z=300),
        hoverinfo="skip",
        showscale=False,
    )


def _append_frustum_mesh(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    *,
    segments: int,
    cap_ends: bool,
    facevalues: list[float] | None = None,
    texture: str = "none",
    texture_strength: float = 0.0,
    texture_target_length: float = 1.0,
    texture_max_axial_segments: int = 24,
    edge_index: int = 0,
) -> None:
    direction = p1 - p0
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 1e-12:
        return

    u, v = _perpendicular_frame(direction)
    radial_segments = max(int(segments), 3)
    theta = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)
    axial_segments = _texture_axial_segment_count(
        length=direction_length,
        texture=texture,
        texture_strength=texture_strength,
        texture_target_length=texture_target_length,
        texture_max_axial_segments=texture_max_axial_segments,
    )
    rings: list[list[np.ndarray]] = []
    for axial_idx in range(axial_segments + 1):
        axial = axial_idx / axial_segments
        center = (1.0 - axial) * p0 + axial * p1
        radius = (1.0 - axial) * float(r0) + axial * float(r1)
        rings.append([center + radius * (np.cos(t) * u + np.sin(t) * v) for t in theta])

    offset = len(vertices)
    for ring in rings:
        vertices.extend(ring)
    n = radial_segments
    for axial_idx in range(axial_segments):
        ring0_offset = offset + axial_idx * n
        ring1_offset = offset + (axial_idx + 1) * n
        axial_mid = (axial_idx + 0.5) / axial_segments
        for idx in range(n):
            nxt = (idx + 1) % n
            a = ring0_offset + idx
            b = ring0_offset + nxt
            c = ring1_offset + nxt
            d = ring1_offset + idx
            triangles.append((a, b, c))
            triangles.append((a, c, d))
            if facevalues is not None:
                theta_mid = float(theta[idx] + np.pi / n)
                value = (
                    _bark_face_value(
                        edge_index=edge_index,
                        face_index=axial_idx * n + idx,
                        theta=theta_mid,
                        axial=axial_mid,
                        texture_strength=texture_strength,
                    )
                    if texture == "bark"
                    else 0.5
                )
                facevalues.extend([value, value])
            if cap_ends and axial_idx == 0:
                triangles.append((offset, b, a))
                if facevalues is not None:
                    facevalues.append(0.38)
            if cap_ends and axial_idx == axial_segments - 1:
                end_offset = offset + axial_segments * n
                triangles.append((end_offset, d, c))
                if facevalues is not None:
                    facevalues.append(0.38)


def _append_sphere_mesh(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    center: np.ndarray,
    radius: float,
    *,
    segments: int,
) -> None:
    if radius <= 0.0:
        return

    lon_count = max(int(segments), 6)
    lat_count = max(lon_count // 2, 4)
    offset = len(vertices)
    vertices.append(center + np.array([0.0, 0.0, radius], dtype=float))

    for lat_idx in range(1, lat_count):
        phi = np.pi * lat_idx / lat_count
        z = radius * np.cos(phi)
        ring_radius = radius * np.sin(phi)
        for lon_idx in range(lon_count):
            theta = 2.0 * np.pi * lon_idx / lon_count
            vertices.append(
                center
                + np.array(
                    [
                        ring_radius * np.cos(theta),
                        ring_radius * np.sin(theta),
                        z,
                    ],
                    dtype=float,
                )
            )

    bottom_idx = len(vertices)
    vertices.append(center + np.array([0.0, 0.0, -radius], dtype=float))

    first_ring = offset + 1
    for lon_idx in range(lon_count):
        nxt = (lon_idx + 1) % lon_count
        triangles.append((offset, first_ring + lon_idx, first_ring + nxt))

    for lat_idx in range(lat_count - 2):
        ring = first_ring + lat_idx * lon_count
        next_ring = ring + lon_count
        for lon_idx in range(lon_count):
            nxt = (lon_idx + 1) % lon_count
            a = ring + lon_idx
            b = ring + nxt
            c = next_ring + nxt
            d = next_ring + lon_idx
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    last_ring = first_ring + (lat_count - 2) * lon_count
    for lon_idx in range(lon_count):
        nxt = (lon_idx + 1) % lon_count
        triangles.append((bottom_idx, last_ring + nxt, last_ring + lon_idx))


def _mesh_trace(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    *,
    name: str,
    color: str,
    opacity: float = 1.0,
    lighting: dict[str, float] | None = None,
    facevalues: list[float] | None = None,
    colorscale: list[list[float | str]] | None = None,
) -> go.Mesh3d | None:
    if not vertices or not triangles:
        return None
    pts = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=int)
    trace_kwargs = dict(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
        name=name,
        flatshading=False,
        lighting=lighting
        or dict(ambient=0.42, diffuse=0.82, specular=0.18, roughness=0.72),
        lightposition=dict(x=100, y=200, z=300),
        hoverinfo="skip",
        showscale=False,
    )
    if facevalues is None:
        trace_kwargs["color"] = color
    else:
        trace_kwargs.update(
            intensity=facevalues,
            intensitymode="cell",
            colorscale=colorscale or BARK_COLORSCALE,
            cmin=0.0,
            cmax=1.0,
        )
    return go.Mesh3d(**trace_kwargs)


def _joint_nodes(graph: "nx.Graph") -> list[int]:
    root = graph.graph.get("root")
    return [
        node
        for node in graph.nodes
        if graph.degree[node] != 2 or node == root
    ]


def _joint_mesh_trace(
    graph: "nx.Graph",
    pos: dict[int, np.ndarray],
    radii: dict[int, float],
    *,
    name: str,
    branch_color: str,
    joint_scale: float,
    joint_segments: int,
) -> go.Mesh3d | None:
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    for node in _joint_nodes(graph):
        if node not in pos or node not in radii:
            continue
        _append_sphere_mesh(
            vertices,
            triangles,
            pos[node],
            max(float(radii[node]) * float(joint_scale), 1e-9),
            segments=joint_segments,
        )
    joint_lighting = dict(
        ambient=0.76,
        diffuse=0.38,
        specular=0.02,
        roughness=0.95,
        fresnel=0.02,
    )
    return _mesh_trace(
        vertices,
        triangles,
        name=name,
        color=branch_color,
        lighting=joint_lighting,
    )


def _tree_mesh_traces(
    graph: "nx.Graph",
    *,
    name: str,
    segments: int,
    radius_attr: str,
    radius_scale: float,
    default_radius: float,
    branch_color: str,
    cap_ends: bool,
    show_joints: bool,
    joint_scale: float,
    joint_segments: int,
    plotly_texture: str,
    plotly_texture_strength: float,
    plotly_texture_target_length_scale: float,
    plotly_texture_max_axial_segments: int,
    plotly_leaves: bool,
    plotly_leaf_count: int,
    plotly_leaf_opacity: float,
    plotly_leaf_scale: float,
    plotly_leaf_seed: int,
) -> tuple[list[go.Mesh3d | go.Scatter3d], dict[int, np.ndarray]]:
    pos = _graph_positions(graph)
    radii = compute_node_radii(
        graph,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
    )
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    texture_enabled = plotly_texture == "bark" and plotly_texture_strength > 0.0
    facevalues: list[float] | None = [] if texture_enabled else None
    edge_items: list[tuple[int, int, int, float]] = []
    for edge_index, (u, v) in enumerate(graph.edges()):
        if u not in pos or v not in pos:
            continue
        length = float(np.linalg.norm(pos[v] - pos[u]))
        if length <= 1e-12:
            continue
        edge_items.append((edge_index, u, v, length))

    edge_lengths = [length for _, _, _, length in edge_items]
    if edge_lengths:
        median_edge_length = float(np.median(edge_lengths))
    else:
        median_edge_length = 1.0
    texture_target_length = median_edge_length * max(
        float(plotly_texture_target_length_scale),
        1e-6,
    )

    for edge_index, u, v, _length in edge_items:
        _append_frustum_mesh(
            vertices,
            triangles,
            pos[u],
            pos[v],
            radii[u],
            radii[v],
            segments=segments,
            cap_ends=cap_ends,
            facevalues=facevalues,
            texture=plotly_texture if texture_enabled else "none",
            texture_strength=plotly_texture_strength if texture_enabled else 0.0,
            texture_target_length=texture_target_length,
            texture_max_axial_segments=plotly_texture_max_axial_segments,
            edge_index=edge_index,
        )

    if not vertices or not triangles:
        if not pos:
            coords = np.zeros((0, 3), dtype=float)
        else:
            coords = np.stack(list(pos.values()), axis=0)
        return (
            [
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(size=3, color=branch_color),
                    name=name,
                    hoverinfo="skip",
                )
            ],
            pos,
        )

    traces: list[go.Mesh3d | go.Scatter3d] = []
    branch_lighting = (
        dict(ambient=0.68, diffuse=0.52, specular=0.04, roughness=0.94, fresnel=0.02)
        if texture_enabled
        else None
    )
    branch_trace = _mesh_trace(
        vertices,
        triangles,
        name=name,
        color=branch_color,
        lighting=branch_lighting,
        facevalues=facevalues,
        colorscale=(
            _bark_colorscale(plotly_texture_strength)
            if texture_enabled
            else None
        ),
    )
    if show_joints:
        joint_trace = _joint_mesh_trace(
            graph,
            pos,
            radii,
            name=f"{name} joints",
            branch_color=branch_color,
            joint_scale=joint_scale,
            joint_segments=joint_segments,
        )
        if joint_trace is not None:
            traces.append(joint_trace)
    if branch_trace is not None:
        traces.append(branch_trace)
    if plotly_leaves:
        leaf_trace = _leaf_mesh_trace(
            graph,
            pos,
            leaf_count=plotly_leaf_count,
            leaf_opacity=plotly_leaf_opacity,
            leaf_scale=plotly_leaf_scale,
            leaf_seed=plotly_leaf_seed,
        )
        if leaf_trace is not None:
            traces.append(leaf_trace)

    if not traces:
        coords = np.stack(list(pos.values()), axis=0)
        traces.append(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(size=3, color=branch_color),
                name=name,
                hoverinfo="skip",
            )
        )
    return traces, pos


def _camera_eye(elev: float, azim: float) -> dict[str, float]:
    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    distance = 1.8
    return dict(
        x=float(distance * np.cos(elev_rad) * np.cos(azim_rad)),
        y=float(distance * np.cos(elev_rad) * np.sin(azim_rad)),
        z=float(distance * np.sin(elev_rad)),
    )


def _scene_layout(*, elev: float, azim: float, show_axes: bool) -> dict:
    axis = dict(
        visible=show_axes,
        showgrid=False,
        zeroline=False,
        showbackground=False,
    )
    return dict(
        aspectmode="data",
        xaxis=axis,
        yaxis=axis,
        zaxis=axis,
        camera=dict(eye=_camera_eye(elev, azim), up=dict(x=0, y=0, z=1)),
    )


def _write_html(fig: go.Figure, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )
    return out_path


def _add_leaf_opacity_slider(fig: go.Figure, *, initial_opacity: float) -> None:
    leaf_trace_indices = [
        idx
        for idx, trace in enumerate(fig.data)
        if getattr(trace, "name", None) == LEAF_TRACE_NAME
    ]
    if not leaf_trace_indices:
        return

    initial = float(np.clip(initial_opacity, 0.0, 1.0))
    opacity_values = sorted(set(LEAF_OPACITY_SLIDER_VALUES + (round(initial, 2),)))
    active_idx = int(np.argmin([abs(value - initial) for value in opacity_values]))
    steps = [
        dict(
            method="restyle",
            label=f"{value:.2f}",
            args=[
                {"opacity": [float(value)] * len(leaf_trace_indices)},
                leaf_trace_indices,
            ],
        )
        for value in opacity_values
    ]
    fig.update_layout(
        sliders=[
            dict(
                active=active_idx,
                currentvalue=dict(prefix="Leaf opacity: ", font=dict(size=12)),
                len=0.62,
                pad=dict(t=6, b=4),
                x=0.19,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                steps=steps,
            )
        ],
        margin=dict(b=56),
    )


def plot_tree_cylinder_single_plotly(
    graph: "nx.Graph",
    *,
    out_path: Path,
    title: str | None = None,
    elev: float = 20.0,
    azim: float = 30.0,
    segments: int = 16,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    branch_color: str = BRANCH_COLOR,
    cap_ends: bool = False,
    show_axes: bool = False,
    show_joints: bool = True,
    joint_scale: float = 1.05,
    joint_segments: int = 10,
    plotly_texture: str = DEFAULT_PLOTLY_TEXTURE,
    plotly_texture_strength: float = DEFAULT_PLOTLY_TEXTURE_STRENGTH,
    plotly_texture_target_length_scale: float = DEFAULT_PLOTLY_TEXTURE_TARGET_LENGTH_SCALE,
    plotly_texture_max_axial_segments: int = DEFAULT_PLOTLY_TEXTURE_MAX_AXIAL_SEGMENTS,
    plotly_leaves: bool = DEFAULT_PLOTLY_LEAVES,
    plotly_leaf_count: int = DEFAULT_PLOTLY_LEAF_COUNT,
    plotly_leaf_opacity: float = DEFAULT_PLOTLY_LEAF_OPACITY,
    plotly_leaf_scale: float = DEFAULT_PLOTLY_LEAF_SCALE,
    plotly_leaf_seed: int = DEFAULT_PLOTLY_LEAF_SEED,
) -> Path:
    """Render one tree as an interactive Plotly HTML cylinder model."""
    traces, _ = _tree_mesh_traces(
        graph,
        name=title or "Tree",
        segments=segments,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
        branch_color=branch_color,
        cap_ends=cap_ends,
        show_joints=show_joints,
        joint_scale=joint_scale,
        joint_segments=joint_segments,
        plotly_texture=plotly_texture,
        plotly_texture_strength=plotly_texture_strength,
        plotly_texture_target_length_scale=plotly_texture_target_length_scale,
        plotly_texture_max_axial_segments=plotly_texture_max_axial_segments,
        plotly_leaves=plotly_leaves,
        plotly_leaf_count=plotly_leaf_count,
        plotly_leaf_opacity=plotly_leaf_opacity,
        plotly_leaf_scale=plotly_leaf_scale,
        plotly_leaf_seed=plotly_leaf_seed,
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=_scene_layout(elev=elev, azim=azim, show_axes=show_axes),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=42 if title else 0, b=0),
        showlegend=False,
    )
    _add_leaf_opacity_slider(fig, initial_opacity=plotly_leaf_opacity)
    return _write_html(fig, Path(out_path))


def plot_tree_cylinder_pair_plotly(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    out_path: Path,
    title_gt: str = "Ground Truth",
    title_pred: str = "Prediction",
    elev: float = 20.0,
    azim: float = 30.0,
    segments: int = 16,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    branch_color: str = BRANCH_COLOR,
    cap_ends: bool = False,
    show_axes: bool = False,
    show_joints: bool = True,
    joint_scale: float = 1.05,
    joint_segments: int = 10,
    plotly_texture: str = DEFAULT_PLOTLY_TEXTURE,
    plotly_texture_strength: float = DEFAULT_PLOTLY_TEXTURE_STRENGTH,
    plotly_texture_target_length_scale: float = DEFAULT_PLOTLY_TEXTURE_TARGET_LENGTH_SCALE,
    plotly_texture_max_axial_segments: int = DEFAULT_PLOTLY_TEXTURE_MAX_AXIAL_SEGMENTS,
    plotly_leaves: bool = DEFAULT_PLOTLY_LEAVES,
    plotly_leaf_count: int = DEFAULT_PLOTLY_LEAF_COUNT,
    plotly_leaf_opacity: float = DEFAULT_PLOTLY_LEAF_OPACITY,
    plotly_leaf_scale: float = DEFAULT_PLOTLY_LEAF_SCALE,
    plotly_leaf_seed: int = DEFAULT_PLOTLY_LEAF_SEED,
) -> Path:
    """Render a side-by-side interactive Plotly GT/pred comparison."""
    traces_gt, _ = _tree_mesh_traces(
        gt,
        name=title_gt,
        segments=segments,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
        branch_color=branch_color,
        cap_ends=cap_ends,
        show_joints=show_joints,
        joint_scale=joint_scale,
        joint_segments=joint_segments,
        plotly_texture=plotly_texture,
        plotly_texture_strength=plotly_texture_strength,
        plotly_texture_target_length_scale=plotly_texture_target_length_scale,
        plotly_texture_max_axial_segments=plotly_texture_max_axial_segments,
        plotly_leaves=plotly_leaves,
        plotly_leaf_count=plotly_leaf_count,
        plotly_leaf_opacity=plotly_leaf_opacity,
        plotly_leaf_scale=plotly_leaf_scale,
        plotly_leaf_seed=plotly_leaf_seed,
    )
    traces_pred, _ = _tree_mesh_traces(
        pred,
        name=title_pred,
        segments=segments,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
        branch_color=branch_color,
        cap_ends=cap_ends,
        show_joints=show_joints,
        joint_scale=joint_scale,
        joint_segments=joint_segments,
        plotly_texture=plotly_texture,
        plotly_texture_strength=plotly_texture_strength,
        plotly_texture_target_length_scale=plotly_texture_target_length_scale,
        plotly_texture_max_axial_segments=plotly_texture_max_axial_segments,
        plotly_leaves=plotly_leaves,
        plotly_leaf_count=plotly_leaf_count,
        plotly_leaf_opacity=plotly_leaf_opacity,
        plotly_leaf_scale=plotly_leaf_scale,
        plotly_leaf_seed=plotly_leaf_seed,
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(title_gt, title_pred),
        horizontal_spacing=0.02,
    )
    for trace in traces_gt:
        fig.add_trace(trace, row=1, col=1)
    for trace in traces_pred:
        fig.add_trace(trace, row=1, col=2)
    scene = _scene_layout(elev=elev, azim=azim, show_axes=show_axes)
    fig.update_layout(
        scene=scene,
        scene2=scene,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=42, b=0),
        showlegend=False,
    )
    _add_leaf_opacity_slider(fig, initial_opacity=plotly_leaf_opacity)
    return _write_html(fig, Path(out_path))
