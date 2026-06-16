"""Plotly-based interactive 3D cylinder rendering helpers for tree graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots_3d import BRANCH_COLOR, compute_node_radii

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
) -> None:
    direction = p1 - p0
    if float(np.linalg.norm(direction)) <= 1e-12:
        return

    u, v = _perpendicular_frame(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, max(int(segments), 3), endpoint=False)
    ring0 = [p0 + r0 * (np.cos(t) * u + np.sin(t) * v) for t in theta]
    ring1 = [p1 + r1 * (np.cos(t) * u + np.sin(t) * v) for t in theta]

    offset = len(vertices)
    vertices.extend(ring0)
    vertices.extend(ring1)
    n = len(ring0)
    for idx in range(n):
        nxt = (idx + 1) % n
        a = offset + idx
        b = offset + nxt
        c = offset + n + nxt
        d = offset + n + idx
        triangles.append((a, b, c))
        triangles.append((a, c, d))
        if cap_ends:
            triangles.append((offset, b, a))
            triangles.append((offset + n, d, c))


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
) -> go.Mesh3d | None:
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
        color=color,
        opacity=opacity,
        name=name,
        flatshading=False,
        lighting=lighting
        or dict(ambient=0.42, diffuse=0.82, specular=0.18, roughness=0.72),
        lightposition=dict(x=100, y=200, z=300),
        hoverinfo="skip",
        showscale=False,
    )


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
    for u, v in graph.edges():
        if u not in pos or v not in pos:
            continue
        _append_frustum_mesh(
            vertices,
            triangles,
            pos[u],
            pos[v],
            radii[u],
            radii[v],
            segments=segments,
            cap_ends=cap_ends,
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
    branch_trace = _mesh_trace(
        vertices,
        triangles,
        name=name,
        color=branch_color,
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
    return _write_html(fig, Path(out_path))
