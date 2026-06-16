"""PyVista-based 3D cylinder rendering helpers for tree graphs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils.styles import DEFAULT_DPI
from .plots_3d import BRANCH_COLOR, compute_node_radii

if TYPE_CHECKING:
    import networkx as nx


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "The PyVista backend requires `pyvista` and its VTK dependency. "
            "Install it in the active environment with `python -m pip install pyvista`."
        ) from exc
    return pv


def _ensure_plotting_supported(pv) -> None:
    if pv.system_supports_plotting():
        return
    if os.environ.get("DENDRITE_GEN_ALLOW_UNSUPPORTED_PYVISTA") == "1":
        return
    raise RuntimeError(
        "PyVista/VTK cannot create a plotting context in this environment. "
        "This usually means the process is running headless without an OpenGL "
        "or OSMesa render context. Run the command from a GUI-capable terminal, "
        "install an OSMesa-enabled VTK build, or set "
        "DENDRITE_GEN_ALLOW_UNSUPPORTED_PYVISTA=1 to try anyway."
    )


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


def _frustum_mesh_arrays(
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    *,
    segments: int,
    cap_ends: bool,
    point_offset: int,
) -> tuple[list[np.ndarray], list[int]]:
    direction = p1 - p0
    if float(np.linalg.norm(direction)) <= 1e-12:
        return [], []

    u, v = _perpendicular_frame(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, max(int(segments), 3), endpoint=False)
    ring0 = [p0 + r0 * (np.cos(t) * u + np.sin(t) * v) for t in theta]
    ring1 = [p1 + r1 * (np.cos(t) * u + np.sin(t) * v) for t in theta]
    points = ring0 + ring1

    faces: list[int] = []
    n = len(ring0)
    for idx in range(n):
        nxt = (idx + 1) % n
        faces.extend(
            [
                4,
                point_offset + idx,
                point_offset + nxt,
                point_offset + n + nxt,
                point_offset + n + idx,
            ]
        )
        if cap_ends:
            faces.extend([3, point_offset + idx, point_offset + nxt, point_offset])
            faces.extend(
                [3, point_offset + n + idx, point_offset + n, point_offset + n + nxt]
            )

    return points, faces


def _tree_polydata(
    graph: "nx.Graph",
    *,
    segments: int,
    radius_attr: str,
    radius_scale: float,
    default_radius: float,
    cap_ends: bool,
):
    pv = _require_pyvista()
    pos = _graph_positions(graph)
    radii = compute_node_radii(
        graph,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
    )

    points: list[np.ndarray] = []
    faces: list[int] = []
    for u, v in graph.edges():
        if u not in pos or v not in pos:
            continue
        edge_points, edge_faces = _frustum_mesh_arrays(
            pos[u],
            pos[v],
            radii[u],
            radii[v],
            segments=segments,
            cap_ends=cap_ends,
            point_offset=len(points),
        )
        points.extend(edge_points)
        faces.extend(edge_faces)

    if not points:
        return pv.PolyData(), pos

    mesh = pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))
    if mesh.n_cells:
        mesh = mesh.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
        )
    return mesh, pos


def _camera_position(
    pos: dict[int, np.ndarray],
    *,
    elev: float,
    azim: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if not pos:
        return (3.0, 3.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)

    pts = np.stack(list(pos.values()), axis=0)
    center = (pts.min(axis=0) + pts.max(axis=0)) * 0.5
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    distance = max(span * 2.6, 1.0)

    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    offset = distance * np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ],
        dtype=float,
    )
    camera = center + offset
    return tuple(camera), tuple(center), (0.0, 0.0, 1.0)


def _render_graph_on_plotter(
    plotter,
    graph: "nx.Graph",
    *,
    title: str | None,
    elev: float,
    azim: float,
    segments: int,
    radius_attr: str,
    radius_scale: float,
    default_radius: float,
    branch_color: str,
    cap_ends: bool,
    show_axes: bool,
) -> None:
    mesh, pos = _tree_polydata(
        graph,
        segments=segments,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
        cap_ends=cap_ends,
    )
    if mesh.n_cells:
        plotter.add_mesh(
            mesh,
            color=branch_color,
            smooth_shading=True,
            specular=0.15,
            diffuse=0.85,
            ambient=0.25,
        )
    elif pos:
        plotter.add_points(
            np.stack(list(pos.values()), axis=0),
            color=branch_color,
            point_size=8,
        )

    if title:
        plotter.add_text(title, position="upper_edge", font_size=11, color="black")
    if show_axes:
        plotter.show_axes()

    plotter.set_background("white")
    plotter.camera_position = _camera_position(pos, elev=elev, azim=azim)
    plotter.camera.zoom(1.15)
    plotter.enable_lightkit()


def plot_tree_cylinder_single_pyvista(
    graph: "nx.Graph",
    *,
    out_path: Path,
    title: str | None = None,
    figsize: tuple[float, float] = (5.2, 6.0),
    elev: float = 20.0,
    azim: float = 30.0,
    segments: int = 16,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    branch_color: str = BRANCH_COLOR,
    cap_ends: bool = False,
    show_axes: bool = False,
) -> Path:
    """Render one tree as a PyVista cylinder model."""
    pv = _require_pyvista()
    _ensure_plotting_supported(pv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window_size = (int(figsize[0] * DEFAULT_DPI), int(figsize[1] * DEFAULT_DPI))
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    _render_graph_on_plotter(
        plotter,
        graph,
        title=title,
        elev=elev,
        azim=azim,
        segments=segments,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
        branch_color=branch_color,
        cap_ends=cap_ends,
        show_axes=show_axes,
    )
    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path


def plot_tree_cylinder_pair_pyvista(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    out_path: Path,
    title_gt: str = "Ground Truth",
    title_pred: str = "Prediction",
    figsize: tuple[float, float] = (10.0, 5.6),
    elev: float = 20.0,
    azim: float = 30.0,
    segments: int = 16,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    branch_color: str = BRANCH_COLOR,
    cap_ends: bool = False,
    show_axes: bool = False,
) -> Path:
    """Render a side-by-side GT/pred PyVista cylinder comparison."""
    pv = _require_pyvista()
    _ensure_plotting_supported(pv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window_size = (int(figsize[0] * DEFAULT_DPI), int(figsize[1] * DEFAULT_DPI))
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=window_size)

    plotter.subplot(0, 0)
    _render_graph_on_plotter(
        plotter,
        gt,
        title=title_gt,
        elev=elev,
        azim=azim,
        segments=segments,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
        branch_color=branch_color,
        cap_ends=cap_ends,
        show_axes=show_axes,
    )

    plotter.subplot(0, 1)
    _render_graph_on_plotter(
        plotter,
        pred,
        title=title_pred,
        elev=elev,
        azim=azim,
        segments=segments,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
        branch_color=branch_color,
        cap_ends=cap_ends,
        show_axes=show_axes,
    )

    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path
