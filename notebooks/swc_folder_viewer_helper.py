from pathlib import Path
import sys
from math import floor

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from ipywidgets import Button, Checkbox, HBox, HTML, IntSlider, Layout, Output, VBox
from IPython.display import clear_output, display

# This helper lives under /notebooks; parent is repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def list_swc_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    files: list[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix != ".swc":
            continue
        files.append(p)
    return files


def load_swc_folder(folder: Path) -> tuple[list[Path], list[nx.Graph]]:
    folder = Path(folder)
    swc_files = list_swc_files(folder)
    if not swc_files:
        raise FileNotFoundError(f"No .swc files found in {folder}")

    swc_graphs = [load_swc_graph_relaxed(path) for path in swc_files]
    if len(swc_graphs) != len(swc_files):
        raise RuntimeError("Mismatch between discovered SWC files and loaded graphs.")

    return swc_files, swc_graphs


def load_swc_graph_relaxed(path: Path) -> nx.Graph:
    """Load SWC as an undirected graph without enforcing tree-ness.

    This is intentionally tolerant for sampled/generated SWC files that may contain
    disconnected components or cycles.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SWC file not found: {path}")

    graph = nx.Graph()
    parent_links: list[tuple[int, int]] = []
    root_id: int | None = None

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed SWC line (expected >=7 cols) in {path}: '{line}'")

            nid = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            parent = int(parts[6])

            if parent <= 0 and root_id is None:
                root_id = nid

            graph.add_node(nid, pos=np.array([x, y, z], dtype=np.float64))
            if parent > 0:
                parent_links.append((parent, nid))

    for parent, child in parent_links:
        if parent not in graph:
            # Keep visualization robust for malformed references.
            graph.add_node(parent, pos=np.zeros(3, dtype=np.float64))
        graph.add_edge(parent, child)

    if root_id is None and graph.number_of_nodes() > 0:
        root_id = sorted(graph.nodes())[0]

    if root_id is not None:
        root_pos = graph.nodes[root_id]["pos"].copy()
        for nid in graph.nodes:
            graph.nodes[nid]["pos"] = graph.nodes[nid]["pos"] - root_pos
    graph.graph["root"] = root_id
    graph.graph["is_tree"] = nx.is_tree(graph) if graph.number_of_nodes() > 0 else True
    return graph


def compute_depths(graph: nx.Graph) -> tuple[dict[int, int], int, int | None]:
    nodes = list(graph.nodes())
    if not nodes:
        return {}, 0, None

    root = graph.graph.get("root", None)
    if root not in graph:
        root = 0 if 0 in graph else nodes[0]

    lengths = nx.single_source_shortest_path_length(graph, root)
    missing_depth = max(lengths.values()) + 1 if lengths else 0
    depths = {n: lengths.get(n, missing_depth) for n in nodes}
    max_depth = max(depths.values(), default=0)
    return depths, max_depth, root


def build_swc_figure(
    graph: nx.Graph,
    title: str,
    depth_limit: int | None = None,
    show_node_labels: bool = False,
    camera_eye: dict | None = None,
) -> tuple[go.Figure, int, int | None]:
    depths, max_depth, root = compute_depths(graph)

    if depth_limit is None:
        visible_nodes = list(graph.nodes())
    else:
        visible_nodes = [n for n, d in depths.items() if d <= depth_limit]

    subgraph = graph.subgraph(visible_nodes).copy()
    if subgraph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no nodes at selected depth)")
        return fig, max_depth, root

    positions: dict[int, np.ndarray] = {}
    for node in subgraph.nodes():
        pos = np.asarray(subgraph.nodes[node].get("pos", np.zeros(3)), dtype=float).flatten()
        if pos.size < 3:
            pos = np.pad(pos, (0, 3 - pos.size), constant_values=0.0)
        positions[node] = pos[:3]

    coord_matrix = np.stack(list(positions.values()), axis=0)
    coord_min = coord_matrix.min(axis=0)
    coord_max = coord_matrix.max(axis=0)
    coord_center = 0.5 * (coord_min + coord_max)

    # Recenter for display so asymmetric trees are framed around the viewport center.
    for node in positions:
        positions[node] = positions[node] - coord_center

    half_spans = 0.5 * (coord_max - coord_min)
    radius = max(float(np.max(half_spans)), 1e-3) * 1.08

    edge_x, edge_y, edge_z = [], [], []
    for u, v in subgraph.edges():
        x0, y0, z0 = positions[u]
        x1, y1, z1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    node_x, node_y, node_z, labels = [], [], [], []
    for node, pos in positions.items():
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_z.append(pos[2])
        labels.append(str(node))

    if camera_eye is None:
        camera_eye = {"x": 1.8, "y": 1.8, "z": 1.8}

    fig = go.Figure()
    if edge_x:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color="lightgray", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text" if show_node_labels else "markers",
            text=labels if show_node_labels else None,
            textposition="top center",
            marker=dict(size=5, color="royalblue", line=dict(color="black", width=0.4)),
            hovertemplate="node=%{text}<extra></extra>" if show_node_labels else None,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
            xaxis=dict(range=[-radius, radius]),
            yaxis=dict(range=[-radius, radius]),
            zaxis=dict(range=[-radius, radius]),
            camera=dict(eye=camera_eye),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    return fig, max_depth, root


def create_swc_folder_viewer(folder: Path) -> VBox:
    folder = Path(folder).resolve()
    swc_files, swc_graphs = load_swc_folder(folder)

    page_size = 4
    num_files = len(swc_files)
    num_pages = max(1, (num_files + page_size - 1) // page_size)
    current_page = {"value": 0}

    prev_button = Button(description="Previous page", button_style="")
    next_button = Button(description="Next page", button_style="")
    pager_html = HTML()

    depth_slider = IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Depth",
        continuous_update=False,
        layout=dict(width="70%"),
    )
    labels_checkbox = Checkbox(value=False, description="Show node labels")
    status_html = HTML()
    # Keep each panel compact so four figures fit without forcing a huge row width.
    panel_outputs = [
        Output(
            layout=Layout(
                width="24.5%",
                min_width="240px",
                max_width="360px",
                overflow="hidden",
            )
        )
        for _ in range(page_size)
    ]

    def page_bounds(page_idx: int) -> tuple[int, int]:
        start = page_idx * page_size
        end = min(num_files, start + page_size)
        return start, end

    def update_pager_controls() -> None:
        page_idx = current_page["value"]
        prev_button.disabled = page_idx <= 0
        next_button.disabled = page_idx >= num_pages - 1

        start, end = page_bounds(page_idx)
        if num_files == 0:
            file_span = "0-0"
        else:
            file_span = f"{start + 1}-{end}"
        pager_html.value = (
            f"Page {page_idx + 1}/{num_pages} | "
            f"Files {file_span} of {num_files}"
        )

    def _panel_figure_size() -> tuple[int, int]:
        # Derive a practical square-ish canvas for 4-up plotting in notebook width.
        default_notebook_width = 1400
        side = floor(default_notebook_width / page_size) - 35
        side = int(np.clip(side, 260, 340))
        return side, side

    def render_page(reset_depth: bool = False) -> None:
        page_idx = current_page["value"]
        start, end = page_bounds(page_idx)
        page_indices = list(range(start, end))

        page_max_depth = 0
        page_details: list[str] = []
        graph_meta: dict[int, tuple[dict[int, int], int, int | None]] = {}

        for idx in page_indices:
            graph = swc_graphs[idx]
            depths, max_depth, root = compute_depths(graph)
            graph_meta[idx] = (depths, max_depth, root)
            page_max_depth = max(page_max_depth, max_depth)

        if reset_depth or depth_slider.max != page_max_depth:
            depth_slider.max = page_max_depth
            depth_slider.value = page_max_depth

        depth_limit = depth_slider.value

        for slot in range(page_size):
            with panel_outputs[slot]:
                clear_output(wait=True)

                idx = start + slot
                if idx >= num_files:
                    print("No file on this slot.")
                    continue

                graph = swc_graphs[idx]
                name = swc_files[idx].name
                depths, max_depth, root = graph_meta[idx]

                fig, _, _ = build_swc_figure(
                    graph,
                    title=f"{name} | n={graph.number_of_nodes()}",
                    depth_limit=depth_limit,
                    show_node_labels=labels_checkbox.value,
                )
                fig_w, fig_h = _panel_figure_size()
                fig.update_layout(
                    width=fig_w,
                    height=fig_h,
                    autosize=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                display(fig)

                visible_n = sum(1 for n in graph.nodes() if depths.get(n, max_depth + 1) <= depth_limit)
                hidden_n = graph.number_of_nodes() - visible_n
                root_txt = f"root={root}" if root is not None else "root=unknown"
                tree_txt = "tree" if graph.graph.get("is_tree", True) else "not-tree"
                page_details.append(
                    f"{idx + 1}:{name} ({tree_txt}, {root_txt}, visible={visible_n}, hidden={hidden_n})"
                )

        details_txt = " | ".join(page_details) if page_details else "No files to display."
        status_html.value = (
            f"Folder: {folder}<br>"
            f"Depth={depth_limit} | {details_txt}"
        )

        update_pager_controls()

    def on_depth_change(_: dict) -> None:
        render_page(reset_depth=False)

    def on_label_toggle(_: dict) -> None:
        render_page(reset_depth=False)

    def on_prev_click(_: Button) -> None:
        if current_page["value"] <= 0:
            return
        current_page["value"] -= 1
        render_page(reset_depth=True)

    def on_next_click(_: Button) -> None:
        if current_page["value"] >= num_pages - 1:
            return
        current_page["value"] += 1
        render_page(reset_depth=True)

    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)
    depth_slider.observe(on_depth_change, names="value")
    labels_checkbox.observe(on_label_toggle, names="value")

    pager_row = HBox([prev_button, next_button, pager_html], layout=Layout(width="100%"))
    plot_row = HBox(
        panel_outputs,
        layout=Layout(
            width="100%",
            justify_content="space-between",
            align_items="flex-start",
            overflow="hidden",
        ),
    )

    render_page(reset_depth=True)
    ui = VBox([pager_row, depth_slider, labels_checkbox, status_html, plot_row])
    return ui
