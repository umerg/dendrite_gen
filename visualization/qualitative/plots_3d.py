"""3D cylinder plotting helpers for tree graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from ..utils.styles import DEFAULT_DPI

if TYPE_CHECKING:
    import networkx as nx


BRANCH_COLOR = "#8a5a2b"


def _pos_to_xyz(pos: np.ndarray | list | tuple) -> np.ndarray:
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _graph_positions(graph: "nx.Graph") -> dict[int, np.ndarray]:
    return {node: _pos_to_xyz(graph.nodes[node].get("pos", np.zeros(3))) for node in graph.nodes}


def _valid_radius(value: object) -> float | None:
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(radius) or radius <= 0.0:
        return None
    return radius


def compute_node_radii(
    graph: "nx.Graph",
    *,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    min_radius: float | None = None,
    max_radius: float | None = None,
) -> dict[int, float]:
    """Return a render radius for every node.

    Existing positive ``radius_attr`` values are used first. Missing values
    default to ``default_radius``.
    """
    radii: dict[int, float] = {}
    for node in graph.nodes:
        radius = _valid_radius(graph.nodes[node].get(radius_attr)) if radius_attr else None
        if radius is None:
            radius = float(default_radius)
        radius *= float(radius_scale)
        if min_radius is not None:
            radius = max(radius, float(min_radius))
        if max_radius is not None:
            radius = min(radius, float(max_radius))
        radii[node] = max(float(radius), 1e-9)
    return radii


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


def _cylinder_faces(
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    *,
    segments: int,
    cap_ends: bool,
) -> list[list[np.ndarray]]:
    direction = p1 - p0
    if float(np.linalg.norm(direction)) <= 1e-12:
        return []
    u, v = _perpendicular_frame(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, max(int(segments), 3), endpoint=False)
    ring0 = np.array([p0 + r0 * (np.cos(t) * u + np.sin(t) * v) for t in theta])
    ring1 = np.array([p1 + r1 * (np.cos(t) * u + np.sin(t) * v) for t in theta])

    faces: list[list[np.ndarray]] = []
    n = ring0.shape[0]
    for idx in range(n):
        nxt = (idx + 1) % n
        faces.append([ring0[idx], ring0[nxt], ring1[nxt], ring1[idx]])
        if cap_ends:
            faces.append([p0, ring0[idx], ring0[nxt]])
            faces.append([p1, ring1[nxt], ring1[idx]])
    return faces


def _set_axes_tight(ax: plt.Axes, pos: dict[int, np.ndarray], radii: dict[int, float]) -> None:
    if not pos:
        return
    pts = np.stack(list(pos.values()), axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    radius_pad = max(max(radii.values(), default=0.0), 1e-3)
    pad = np.maximum(ranges * 0.05, radius_pad)
    mins = mins - pad
    maxs = maxs + pad
    spans = np.maximum(maxs - mins, 1e-6)
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_box_aspect(spans)


def _style_3d_axis(ax: plt.Axes, *, show_axes: bool) -> None:
    if show_axes:
        ax.grid(False)
        return
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_tree_cylinder_3d(
    ax: plt.Axes,
    graph: "nx.Graph",
    *,
    title: str | None = None,
    elev: float = 20.0,
    azim: float = 30.0,
    segments: int = 12,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    branch_color: str = BRANCH_COLOR,
    alpha: float = 1.0,
    cap_ends: bool = False,
    show_axes: bool = False,
) -> None:
    """Draw a tree as a set of tapered cylinders on a 3D axis."""
    pos = _graph_positions(graph)
    if not pos:
        if title:
            ax.set_title(title)
        _style_3d_axis(ax, show_axes=show_axes)
        return

    radii = compute_node_radii(
        graph,
        radius_attr=radius_attr,
        radius_scale=radius_scale,
        default_radius=default_radius,
    )
    edges = [(u, v) for u, v in graph.edges() if u in pos and v in pos]

    faces: list[list[np.ndarray]] = []
    for u, v in edges:
        edge_faces = _cylinder_faces(
            pos[u],
            pos[v],
            radii[u],
            radii[v],
            segments=segments,
            cap_ends=cap_ends,
        )
        if not edge_faces:
            continue
        faces.extend(edge_faces)

    if faces:
        collection = Poly3DCollection(
            faces,
            facecolors=branch_color,
            edgecolors="none",
            linewidths=0.0,
            alpha=alpha,
        )
        ax.add_collection3d(collection)
    elif len(pos) == 1:
        pt = next(iter(pos.values()))
        ax.scatter([pt[0]], [pt[1]], [pt[2]], s=20.0, color=branch_color, depthshade=True)

    if title:
        ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_axes_tight(ax, pos, radii)
    _style_3d_axis(ax, show_axes=show_axes)


def plot_tree_cylinder_single_3d(
    graph: "nx.Graph",
    *,
    out_path: Path,
    title: str | None = None,
    figsize: tuple[float, float] = (5.2, 6.0),
    **kwargs,
) -> Path:
    """Render one tree as a 3D cylinder model."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_tree_cylinder_3d(ax, graph, title=title, **kwargs)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_path


def plot_tree_cylinder_pair_3d(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    out_path: Path,
    title_gt: str = "Ground Truth",
    title_pred: str = "Prediction",
    figsize: tuple[float, float] = (10.0, 5.6),
    **kwargs,
) -> Path:
    """Render a side-by-side GT/pred 3D cylinder comparison."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
    plot_tree_cylinder_3d(
        ax_gt,
        gt,
        title=title_gt,
        **kwargs,
    )
    plot_tree_cylinder_3d(
        ax_pred,
        pred,
        title=title_pred,
        **kwargs,
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_path
