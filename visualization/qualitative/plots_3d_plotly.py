"""Plotly-based interactive 3D cylinder rendering helpers for tree graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots_3d import BRANCH_COLOR, compute_node_radii
from .plotly_defaults import (
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
    plotly_texture: str = DEFAULT_PLOTLY_TEXTURE,
    plotly_texture_strength: float = DEFAULT_PLOTLY_TEXTURE_STRENGTH,
    plotly_texture_target_length_scale: float = DEFAULT_PLOTLY_TEXTURE_TARGET_LENGTH_SCALE,
    plotly_texture_max_axial_segments: int = DEFAULT_PLOTLY_TEXTURE_MAX_AXIAL_SEGMENTS,
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
    plotly_texture: str = DEFAULT_PLOTLY_TEXTURE,
    plotly_texture_strength: float = DEFAULT_PLOTLY_TEXTURE_STRENGTH,
    plotly_texture_target_length_scale: float = DEFAULT_PLOTLY_TEXTURE_TARGET_LENGTH_SCALE,
    plotly_texture_max_axial_segments: int = DEFAULT_PLOTLY_TEXTURE_MAX_AXIAL_SEGMENTS,
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
