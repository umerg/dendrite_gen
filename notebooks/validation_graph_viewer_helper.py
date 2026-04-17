from pathlib import Path
import pickle
import sys

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from ipywidgets import Button, Dropdown, IntSlider, HTML, HBox, Layout, Output, VBox
from IPython.display import clear_output, display

# This helper lives under /notebooks; parent is repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.data_loading import load_swc_graphs_from_dir


def step_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.startswith("step_"):
        suffix = stem.split("step_", 1)[1]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, stem)


def discover_validation_pickles(output_root: Path) -> list[Path]:
    output_root = Path(output_root)
    if not output_root.exists() or not output_root.is_dir():
        raise NotADirectoryError(f"Validation output folder not found: {output_root}")

    pickles = sorted(output_root.glob("*.pkl"), key=step_sort_key)
    if not pickles:
        raise FileNotFoundError(f"No validation pickles found under {output_root}")
    return pickles


def load_validation_pickle(path: Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")
    return payload


def load_ground_truth_graphs(gt_root: Path) -> tuple[list[nx.Graph], str | None]:
    try:
        graphs = load_swc_graphs_from_dir(gt_root)
        return graphs, None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def pretty_label(path: Path, output_root: Path) -> str:
    path = Path(path)
    output_root = Path(output_root)
    try:
        return str(path.relative_to(output_root))
    except ValueError:
        return str(path)


def compute_node_depths(graph: nx.Graph) -> tuple[dict[int, int], int, int | None]:
    nodes = list(graph.nodes())
    if not nodes:
        return {}, 0, None

    root = graph.graph.get("root")
    if root not in graph:
        root = 0 if 0 in graph else nodes[0]

    lengths = nx.single_source_shortest_path_length(graph, root)
    missing_depth = max(lengths.values()) + 1 if lengths else 0
    depths = {n: lengths.get(n, missing_depth) for n in nodes}
    max_depth = max(depths.values(), default=0)
    return depths, max_depth, root


def build_plotly_figure(
    graph: nx.Graph,
    title: str,
    node_color: str,
    edge_color: str,
    camera_eye: dict,
) -> go.Figure:
    positions: dict[int, np.ndarray] = {}
    for node in graph.nodes():
        pos = np.asarray(graph.nodes[node].get("pos", np.zeros(3)), dtype=float).flatten()
        if pos.size < 3:
            pos = np.pad(pos, (0, 3 - pos.size), constant_values=0.0)
        positions[node] = pos[:3]

    coord_matrix = np.stack(list(positions.values()), axis=0)
    coord_min = coord_matrix.min(axis=0)
    coord_max = coord_matrix.max(axis=0)
    coord_center = 0.5 * (coord_min + coord_max)

    # Recenter plotted coordinates so asymmetric trees don't sit in a corner.
    for node in positions:
        positions[node] = positions[node] - coord_center

    half_spans = 0.5 * (coord_max - coord_min)
    radius = max(float(np.max(half_spans)), 1e-3) * 1.08

    node_x, node_y, node_z, node_labels = [], [], [], []
    for node, pos in positions.items():
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_z.append(pos[2])
        node_labels.append(str(node))

    edge_x, edge_y, edge_z = [], [], []
    for u, v in graph.edges():
        x0, y0, z0 = positions[u]
        x1, y1, z1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    fig = go.Figure()
    if edge_x:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=edge_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(size=6, color=node_color, line=dict(color="black", width=0.5)),
            text=node_labels,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
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
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def create_validation_viewer(
    output_root: Path,
    gt_root: Path,
    camera_eye: dict | None = None,
) -> VBox:
    output_root = Path(output_root).resolve()
    gt_root = Path(gt_root).resolve()
    shared_camera_eye = camera_eye or {"x": 1.8, "y": 1.8, "z": 1.8}

    available_pickles = discover_validation_pickles(output_root)
    gt_graphs, gt_load_error = load_ground_truth_graphs(gt_root)

    with available_pickles[-1].open("rb") as f:
        latest_payload = pickle.load(f)
    pred_counts = {
        beta_key: len(beta_payload.get("pred_graphs", []))
        for beta_key, beta_payload in latest_payload.items()
        if isinstance(beta_payload, dict)
    }

    pickle_options = [(pretty_label(p, output_root), str(p)) for p in available_pickles]
    file_dropdown = Dropdown(
        options=pickle_options,
        description="Result",
        value=pickle_options[-1][1],
        layout=dict(width="70%"),
    )
    beta_dropdown = Dropdown(options=[], description="EMA beta", layout=dict(width="40%"))
    graph_slider = IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Graph idx",
        continuous_update=False,
        layout=dict(width="60%"),
    )
    prev_graph_button = Button(description="Previous", button_style="")
    next_graph_button = Button(description="Next", button_style="")
    depth_slider = IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Depth",
        continuous_update=False,
        layout=dict(width="60%"),
        disabled=True,
    )
    status_html = HTML(value="Select a pickle to begin.")
    gt_plot_output = Output(layout=Layout(width="49%", overflow="hidden"))
    pred_plot_output = Output(layout=Layout(width="49%", overflow="hidden"))

    validation_payload = load_validation_pickle(file_dropdown.value)

    state = {
        "current_pred_depths": {},
        "current_pred_max_depth": 0,
        "current_pred_root": None,
        "current_gt_depths": {},
        "current_gt_max_depth": 0,
        "current_gt_root": None,
        "suppress_graph_event": False,
        "suppress_depth_event": False,
    }

    def set_graph_slider(value: int) -> None:
        state["suppress_graph_event"] = True
        graph_slider.value = value
        state["suppress_graph_event"] = False

    def update_graph_nav_buttons() -> None:
        prev_graph_button.disabled = graph_slider.disabled or graph_slider.value <= graph_slider.min
        next_graph_button.disabled = graph_slider.disabled or graph_slider.value >= graph_slider.max

    def set_depth_slider(value: int) -> None:
        state["suppress_depth_event"] = True
        depth_slider.value = value
        state["suppress_depth_event"] = False

    def clear_depth_state() -> None:
        state["current_pred_depths"] = {}
        state["current_pred_max_depth"] = 0
        state["current_pred_root"] = None
        state["current_gt_depths"] = {}
        state["current_gt_max_depth"] = 0
        state["current_gt_root"] = None

        state["suppress_depth_event"] = True
        depth_slider.disabled = True
        depth_slider.max = 0
        depth_slider.value = 0
        depth_slider.description = "Depth"
        state["suppress_depth_event"] = False

    def clear_plot_outputs(msg_gt: str = "", msg_pred: str = "") -> None:
        with gt_plot_output:
            clear_output(wait=True)
            if msg_gt:
                print(msg_gt)
        with pred_plot_output:
            clear_output(wait=True)
            if msg_pred:
                print(msg_pred)

    def update_plot(recompute_depth: bool) -> None:
        beta_key = beta_dropdown.value
        pred_graphs = validation_payload.get(beta_key, {}).get("pred_graphs", []) if beta_key else []
        if not pred_graphs:
            return

        idx = int(np.clip(graph_slider.value, 0, len(pred_graphs) - 1))
        pred_graph = pred_graphs[idx]
        gt_graph = gt_graphs[idx] if (not gt_load_error and idx < len(gt_graphs)) else None

        if recompute_depth or not state["current_pred_depths"]:
            pred_depths, pred_max_depth, pred_root = compute_node_depths(pred_graph)
            state["current_pred_depths"] = pred_depths
            state["current_pred_max_depth"] = pred_max_depth
            state["current_pred_root"] = pred_root

            if gt_graph is not None:
                gt_depths, gt_max_depth, gt_root = compute_node_depths(gt_graph)
                state["current_gt_depths"] = gt_depths
                state["current_gt_max_depth"] = gt_max_depth
                state["current_gt_root"] = gt_root
            else:
                state["current_gt_depths"] = {}
                state["current_gt_max_depth"] = 0
                state["current_gt_root"] = None

            depth_slider.min = 0
            depth_slider.max = max(state["current_pred_max_depth"], state["current_gt_max_depth"])
            depth_slider.step = 1
            depth_slider.disabled = False
            if pred_root is not None:
                depth_slider.description = f"Depth (pred root={pred_root})"
            else:
                depth_slider.description = "Depth"
            set_depth_slider(depth_slider.max)

        depth_limit = depth_slider.value if not depth_slider.disabled else max(
            state["current_pred_max_depth"], state["current_gt_max_depth"]
        )

        pred_visible_nodes = [n for n, d in state["current_pred_depths"].items() if d <= depth_limit]
        pred_subgraph = pred_graph.subgraph(pred_visible_nodes).copy()
        pred_hidden_nodes = pred_graph.number_of_nodes() - pred_subgraph.number_of_nodes()

        with pred_plot_output:
            clear_output(wait=True)
            if pred_subgraph.number_of_nodes() == 0:
                print("Prediction: no nodes visible at this depth. Increase the slider.")
            else:
                pred_title = (
                    f"{beta_key} | Pred graph {idx} | nodes={pred_subgraph.number_of_nodes()} "
                    f"(hidden {pred_hidden_nodes})"
                )
                pred_fig = build_plotly_figure(
                    pred_subgraph,
                    pred_title,
                    node_color="royalblue",
                    edge_color="lightgray",
                    camera_eye=shared_camera_eye,
                )
                display(pred_fig)

        gt_hidden_nodes = None
        if gt_graph is not None:
            gt_visible_nodes = [n for n, d in state["current_gt_depths"].items() if d <= depth_limit]
            gt_subgraph = gt_graph.subgraph(gt_visible_nodes).copy()
            gt_hidden_nodes = gt_graph.number_of_nodes() - gt_subgraph.number_of_nodes()

            with gt_plot_output:
                clear_output(wait=True)
                if gt_subgraph.number_of_nodes() == 0:
                    print("GT: no nodes visible at this depth. Increase the slider.")
                else:
                    gt_title = (
                        f"GT graph {idx} | nodes={gt_subgraph.number_of_nodes()} "
                        f"(hidden {gt_hidden_nodes})"
                    )
                    gt_fig = build_plotly_figure(
                        gt_subgraph,
                        gt_title,
                        node_color="forestgreen",
                        edge_color="darkseagreen",
                        camera_eye=shared_camera_eye,
                    )
                    display(gt_fig)
        else:
            with gt_plot_output:
                clear_output(wait=True)
                if gt_load_error:
                    print(f"GT unavailable: {gt_load_error}")
                else:
                    print(
                        f"No GT graph for index {idx}. "
                        f"Pred graphs: {len(pred_graphs)} | GT graphs: {len(gt_graphs)}."
                    )

        if gt_hidden_nodes is None:
            gt_extra = "GT unavailable for current index."
        elif gt_hidden_nodes:
            gt_extra = f"GT hidden beyond depth {depth_limit}: {gt_hidden_nodes}"
        else:
            gt_extra = "GT all nodes visible."

        if pred_hidden_nodes:
            pred_extra = f"Pred hidden beyond depth {depth_limit}: {pred_hidden_nodes}"
        else:
            pred_extra = "Pred all nodes visible."

        status_html.value = (
            f"{len(pred_graphs)} generated graph(s) available for {beta_key}. "
            f"Pairing: same index with GT from deterministic dataset order. "
            f"{gt_extra} {pred_extra}"
        )

    def refresh_graph_slider() -> None:
        beta_key = beta_dropdown.value
        graphs = validation_payload.get(beta_key, {}).get("pred_graphs", []) if beta_key else []
        has_graphs = len(graphs) > 0

        graph_slider.disabled = not has_graphs
        graph_slider.max = max(0, len(graphs) - 1)
        if graph_slider.value > graph_slider.max:
            set_graph_slider(graph_slider.max)

        if has_graphs:
            if gt_load_error:
                gt_msg = f"GT unavailable ({gt_load_error}). Showing predictions only."
            else:
                gt_msg = f"GT loaded: {len(gt_graphs)} graph(s) from {gt_root}."
            status_html.value = f"{len(graphs)} generated graph(s) available for {beta_key}. {gt_msg}"
            set_graph_slider(graph_slider.max)
            update_plot(recompute_depth=True)
        else:
            clear_depth_state()
            status_html.value = f"No `pred_graphs` stored for {beta_key}."
            clear_plot_outputs(
                msg_gt="No GT graph to display.",
                msg_pred="No graphs to display for this beta selection.",
            )

        update_graph_nav_buttons()

    def refresh_beta_dropdown() -> None:
        clear_depth_state()
        betas = sorted(validation_payload.keys())
        if not betas:
            beta_dropdown.options = []
            beta_dropdown.value = None
            graph_slider.disabled = True
            status_html.value = "Selected pickle contains no EMA entries."
            clear_plot_outputs(
                msg_gt="No GT data to display.",
                msg_pred="No data to display for this pickle.",
            )
            return

        beta_dropdown.options = betas
        if beta_dropdown.value not in betas:
            beta_dropdown.value = betas[0]
        refresh_graph_slider()

    def handle_file_change(change: dict) -> None:
        nonlocal validation_payload
        new_path = change.get("new")
        if not new_path:
            return
        validation_payload = load_validation_pickle(new_path)
        refresh_beta_dropdown()

    def handle_beta_change(change: dict) -> None:
        if change.get("new") is None:
            return
        refresh_graph_slider()

    def handle_graph_change(_: dict) -> None:
        if state["suppress_graph_event"]:
            return
        update_graph_nav_buttons()
        update_plot(recompute_depth=True)

    def handle_depth_change(_: dict) -> None:
        if state["suppress_depth_event"]:
            return
        update_plot(recompute_depth=False)

    def handle_prev_graph_click(_: Button) -> None:
        if graph_slider.disabled:
            return
        if graph_slider.value <= graph_slider.min:
            return
        graph_slider.value = graph_slider.value - 1

    def handle_next_graph_click(_: Button) -> None:
        if graph_slider.disabled:
            return
        if graph_slider.value >= graph_slider.max:
            return
        graph_slider.value = graph_slider.value + 1

    file_dropdown.observe(handle_file_change, names="value")
    beta_dropdown.observe(handle_beta_change, names="value")
    graph_slider.observe(handle_graph_change, names="value")
    depth_slider.observe(handle_depth_change, names="value")
    prev_graph_button.on_click(handle_prev_graph_click)
    next_graph_button.on_click(handle_next_graph_click)

    refresh_beta_dropdown()

    if gt_load_error:
        print(f"Loaded {len(available_pickles)} validation pickle(s); GT load failed: {gt_load_error}")
    else:
        mismatch = {k: v for k, v in pred_counts.items() if v != len(gt_graphs)}
        if mismatch:
            print(f"Loaded {len(available_pickles)} validation pickle(s); GT count={len(gt_graphs)}; mismatches={mismatch}")
        else:
            print(f"Loaded {len(available_pickles)} validation pickle(s); GT count={len(gt_graphs)}; counts match latest pickle.")

    graph_nav_row = HBox([graph_slider, prev_graph_button, next_graph_button], layout=Layout(width="100%"))

    plot_row = HBox(
        [gt_plot_output, pred_plot_output],
        layout=Layout(width="100%", justify_content="space-between", align_items="flex-start", overflow="hidden"),
    )
    ui = VBox([file_dropdown, beta_dropdown, graph_nav_row, depth_slider, status_html, plot_row])
    return ui
