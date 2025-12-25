import os
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
from torch_sparse import SparseTensor
from typing import Sequence
from matplotlib.lines import Line2D

def adj_pos_to_nx(adj: SparseTensor, pos: th.Tensor):
	"""Convert adjacency (SparseTensor) and positions (Tensor[N,3]) to a NetworkX graph.

	Nodes have attribute 'pos' = numpy array length 3.
	Edges are undirected; duplicates removed.
	"""
	if adj is None:
		raise ValueError("adj cannot be None")
	if pos is None:
		raise ValueError("pos cannot be None")
	if pos.dim() != 2 or pos.size(1) < 3:
		raise ValueError("pos must be [N,3] or wider")
	row, col, _ = adj.coo()
	G = nx.Graph()
	for i in range(pos.size(0)):
		G.add_node(i, pos=pos[i, :3].detach().cpu().numpy())
	for r, c in zip(row.tolist(), col.tolist()):
		if r <= c:  # add each undirected edge once
			G.add_edge(r, c)
	return G

def plot_nx_3d(G: nx.Graph, out_file: Path, elev: int = 20, azim: int = 45, node_colors: list[str] | None = None):
	"""Plot a 3D scatter with edges using stored 'pos' node attributes.

	node_colors: optional list of hex/HTML colors length == num nodes.
	Saves figure to out_file (parent directory created if needed).
	"""
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111, projection='3d')

	xs, ys, zs = [], [], []
	for n, data in G.nodes(data=True):
		x,y,z = data['pos']
		xs.append(x); ys.append(y); zs.append(z)
	if node_colors is None:
		node_colors = ['steelblue'] * len(xs)
	ax.scatter(xs, ys, zs, s=25, c=node_colors, depthshade=True)

	for u, v in G.edges():
		x = [G.nodes[u]['pos'][0], G.nodes[v]['pos'][0]]
		y = [G.nodes[u]['pos'][1], G.nodes[v]['pos'][1]]
		z = [G.nodes[u]['pos'][2], G.nodes[v]['pos'][2]]
		ax.plot(x, y, z, color='gray', linewidth=1)

	ax.view_init(elev=elev, azim=azim)
	ax.set_axis_off()
	fig.tight_layout()
	fig.savefig(out_file)
	plt.close(fig)

def _make_leaf_colors(num_nodes: int, leaf_local_idx: list[int] | None, leaf_expansion: list[int] | None) -> list[str]:
	"""Return per-node color list.

	Non-leaf: blue, leaf terminal (1): red, leaf expand (2): green.
	"""
	base = ['#1f77b4'] * num_nodes  # blue
	if leaf_local_idx is None:
		return base
	exp_map = {}
	if leaf_expansion is not None and len(leaf_expansion) == len(leaf_local_idx):
		exp_map = {i: e for i, e in zip(leaf_local_idx, leaf_expansion)}
	for i in leaf_local_idx:
		label = exp_map.get(i, 1)  # default terminal
		if label == 2:
			base[i] = '#2ca02c'  # green
		else:
			base[i] = '#d62728'  # red
	return base


def _make_enhanced_node_colors(num_nodes: int, leaf_local_idx: list[int] | None, 
                              leaf_expansion: list[int] | None, root_local_idx: list[int] | None) -> list[str]:
	"""Return enhanced per-node color list with distinct colors for roots, leaves, and internal nodes.
	
	Root nodes: gold/orange, Internal nodes: steel blue, 
	Leaf terminal (1): red, Leaf expand (2): green.
	"""
	base = ['#4682b4'] * num_nodes  # steel blue for internal nodes
	
	# Color root nodes first (highest priority)
	if root_local_idx is not None:
		for i in root_local_idx:
			if 0 <= i < num_nodes:
				base[i] = '#ffa500'  # orange for roots
	
	# Color leaf nodes (overrides internal, but not root)
	if leaf_local_idx is not None:
		exp_map = {}
		if leaf_expansion is not None and len(leaf_expansion) == len(leaf_local_idx):
			exp_map = {i: e for i, e in zip(leaf_local_idx, leaf_expansion)}
		
		for i in leaf_local_idx:
			if 0 <= i < num_nodes and i not in (root_local_idx or []):  # don't override root color
				label = exp_map.get(i, 1)  # default terminal
				if label == 2:
					base[i] = '#32cd32'  # lime green for expanding leaves
				else:
					base[i] = '#dc143c'  # crimson for terminal leaves
	
	return base

def plot_gt_and_masked(
	adj: SparseTensor,
	pos_gt: th.Tensor,
	pos_masked: th.Tensor,
	out_dir: Path,
	prefix: str,
	step: int,
	batch_id: int,
	leaf_local_idx: list[int] | None = None,
	leaf_expansion: list[int] | None = None,
):
	"""Helper to plot ground-truth vs masked input graphs in 3D with leaf coloring.

	Creates two files:
	  {prefix}_gt_step{step}_b{batch_id}.png
	  {prefix}_masked_step{step}_b{batch_id}.png
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	G_gt = adj_pos_to_nx(adj, pos_gt)
	G_masked = adj_pos_to_nx(adj, pos_masked)
	gt_file = out_dir / f"{prefix}_gt_step{step}_b{batch_id}.png"
	masked_file = out_dir / f"{prefix}_masked_step{step}_b{batch_id}.png"
	colors_gt = _make_leaf_colors(G_gt.number_of_nodes(), leaf_local_idx, leaf_expansion)
	colors_masked = colors_gt  # same coloring for masked variant
	plot_nx_3d(G_gt, gt_file, node_colors=colors_gt)
	plot_nx_3d(G_masked, masked_file, node_colors=colors_masked)
	return gt_file, masked_file


def plot_gt_and_masked_enhanced(
	adj: SparseTensor,
	pos_gt: th.Tensor,
	pos_masked: th.Tensor,
	out_dir: Path,
	prefix: str,
	step: int,
	batch_id: int,
	leaf_local_idx: list[int] | None = None,
	leaf_expansion: list[int] | None = None,
	root_local_idx: list[int] | None = None,
	node_labels: list[str] | None = None,
):
	"""Enhanced helper to plot ground-truth vs masked input graphs in 3D with root/leaf coloring.
	
	Creates two files with enhanced node coloring:
	  {prefix}_gt_step{step}_b{batch_id}.png
	  {prefix}_masked_step{step}_b{batch_id}.png
	  
	Color scheme:
	- Root nodes: Orange
	- Internal nodes: Steel blue
	- Terminal leaves: Crimson red
	- Expanding leaves: Lime green
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	G_gt = adj_pos_to_nx(adj, pos_gt)
	G_masked = adj_pos_to_nx(adj, pos_masked)
	
	gt_file = out_dir / f"{prefix}_gt_step{step}_b{batch_id}.png"
	masked_file = out_dir / f"{prefix}_masked_step{step}_b{batch_id}.png"
	
	# Enhanced coloring with root identification
	colors_enhanced = _make_enhanced_node_colors(G_gt.number_of_nodes(), leaf_local_idx, leaf_expansion, root_local_idx)
	
	# Create plots with larger node sizes and enhanced styling
	plot_nx_3d_enhanced(G_gt, gt_file, node_colors=colors_enhanced, title=f"GT Graph {batch_id} (Step {step})", node_labels=node_labels)
	plot_nx_3d_enhanced(G_masked, masked_file, node_colors=colors_enhanced, title=f"Masked Graph {batch_id} (Step {step})", node_labels=node_labels)
	
	return gt_file, masked_file


def plot_nx_3d_enhanced(G: nx.Graph, out_file: Path, elev: int = 20, azim: int = 45, 
                       node_colors: list[str] | None = None, title: str = "", node_labels: list[str] | None = None):
	"""Enhanced 3D plot with better styling and node size differentiation.
	
	node_colors: optional list of hex/HTML colors length == num nodes.
	Saves figure to out_file (parent directory created if needed).
	"""
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')

	xs, ys, zs = [], [], []
	node_sizes = []
	
	for n, data in G.nodes(data=True):
		x, y, z = data['pos']
		xs.append(x); ys.append(y); zs.append(z)
		
		# Vary node size based on color (root nodes larger)
		if node_colors and n < len(node_colors):
			if node_colors[n] == '#ffa500':  # orange = root
				node_sizes.append(100)
			elif node_colors[n] in ['#dc143c', '#32cd32']:  # red/green = leaves
				node_sizes.append(60)
			else:  # internal nodes
				node_sizes.append(40)
		else:
			node_sizes.append(50)
	
	if node_colors is None:
		node_colors = ['steelblue'] * len(xs)
	
	# Plot nodes with varied sizes
	ax.scatter(xs, ys, zs, s=node_sizes, c=node_colors, depthshade=True, 
	          alpha=0.8, edgecolors='black', linewidth=0.5)

	# Plot edges with improved styling
	for u, v in G.edges():
		x = [G.nodes[u]['pos'][0], G.nodes[v]['pos'][0]]
		y = [G.nodes[u]['pos'][1], G.nodes[v]['pos'][1]]
		z = [G.nodes[u]['pos'][2], G.nodes[v]['pos'][2]]
		ax.plot(x, y, z, color='#666666', linewidth=1.5, alpha=0.7)

	# Add node labels for small graphs
	if len(xs) <= 10:
		for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
			label = node_labels[i] if node_labels and i < len(node_labels) else f'{i}'
			ax.text(x, y, z, f'  {label}', fontsize=8, color='black', weight='bold')

	ax.view_init(elev=elev, azim=azim)
	ax.set_axis_off()
	
	if title:
		fig.suptitle(title, fontsize=12, weight='bold')
	
	fig.tight_layout()
	fig.savefig(out_file, dpi=150, bbox_inches='tight')
	plt.close(fig)


def log_root_children_debug(
	out_dir: Path,
	*,
	step: int,
	batch_index: int,
	graph_index: int,
	node_ids: Sequence[int],
	parent_local: th.Tensor,
	pos_gt: th.Tensor,
	pos_masked: th.Tensor,
	geo_lr_mask: th.Tensor,
	sibling_order: th.Tensor | None = None,
	leaf_mask: th.Tensor,
	leaf_train_mask: th.Tensor,
	new_leaf_mask: th.Tensor | None = None,
	leaf_expansion_state: th.Tensor | None = None,
	adj: SparseTensor | None = None,
	graph_size: int | None = None,
) -> Path:
	"""Persist a textual + visual snapshot for a root+children graph."""
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	text_file = out_dir / f"root_children_step{step}_b{batch_index}_g{graph_index}.txt"

	root_local_idx = (parent_local < 0).nonzero(as_tuple=False).flatten().tolist()
	child_local_idx = [
		i
		for i, p in enumerate(parent_local.tolist())
		if p >= 0 and root_local_idx and p == root_local_idx[0]
	]

	with text_file.open("w") as fp:
		fp.write(f"Graph {graph_index} (debug batch {batch_index}, step {step})\n")
		fp.write(f"Global -> local node ids: {list(zip(node_ids, range(len(node_ids))))}\n")
		fp.write(f"Root local indices: {root_local_idx}\n")
		fp.write(f"Child local indices: {child_local_idx}\n\n")
		if graph_size is not None:
			fp.write(f"Graph size: {graph_size}\n\n")
		fp.write("Per-node summary:\n")
		node_labels: list[str] = []
		for local_idx, global_idx in enumerate(node_ids):
			pos_abs = pos_gt[local_idx].detach().cpu().numpy().tolist()
			pos_mask = pos_masked[local_idx].detach().cpu().numpy().tolist()
			if parent_local[local_idx].item() < 0:
				label = f"{local_idx}:root"
			else:
				is_left = bool(geo_lr_mask[local_idx].item())
				label = f"{local_idx}:{'L' if is_left else 'R'}"
			expansion_label = None
			if leaf_expansion_state is not None and leaf_expansion_state.numel() == len(node_ids):
				exp_val = int(leaf_expansion_state[local_idx].item())
				if exp_val >= 0:
					expansion_label = exp_val
					label += f"[{exp_val}]"
			node_labels.append(label)
			fp.write(
				f"  node(local={local_idx}, global={global_idx}): "
				f"parent_local={int(parent_local[local_idx].item())}, "
				f"geo_left={bool(geo_lr_mask[local_idx].item())}, "
				f"is_leaf={bool(leaf_mask[local_idx].item())}, "
				f"is_train_leaf={bool(leaf_train_mask[local_idx].item())}, "
			)
			if new_leaf_mask is not None and new_leaf_mask.numel() == leaf_mask.numel():
				fp.write(f"is_new_leaf={bool(new_leaf_mask[local_idx].item())}, ")
			if sibling_order is not None and sibling_order.numel() == len(node_ids):
				fp.write(f"sibling_order={int(sibling_order[local_idx].item())}, ")
			if expansion_label is not None:
				fp.write(f"leaf_expansion_state={expansion_label}, ")
			fp.write(f"pos_gt={pos_abs}, pos_masked={pos_mask}\n")

	if adj is not None:
		root_idx_list = root_local_idx if root_local_idx else None
	plot_gt_and_masked_enhanced(
			adj=adj,
			pos_gt=pos_gt,
			pos_masked=pos_masked,
			out_dir=out_dir,
			prefix=f"root_children_step{step}_g{graph_index}",
			step=step,
			batch_id=batch_index,
			leaf_local_idx=[i for i, flag in enumerate(leaf_mask.tolist()) if flag],
			leaf_expansion=None,
			root_local_idx=root_idx_list,
			node_labels=node_labels,
		)

	return text_file


def plot_geometry_debug(
	pos: th.Tensor,
	node_id: int,
	parent_id: int | None,
	neighbor_in: Sequence[int],
	neighbor_out: Sequence[int],
	uhat: th.Tensor,
	e1_node: th.Tensor,
	e2_node: th.Tensor,
	e1_parent: th.Tensor | None,
	e2_parent: th.Tensor | None,
	edge_vecs_in: Sequence[th.Tensor],
	edge_vecs_out: Sequence[th.Tensor],
	edge_decomp_in: Sequence[tuple[th.Tensor, th.Tensor]],  # (r_par, r_perp)
	edge_decomp_out: Sequence[tuple[th.Tensor, th.Tensor]],
	angles_in: Sequence[tuple[float, float]],               # (cos, sin) relative to node frame
	angles_out: Sequence[tuple[float, float]],
	out_dir: Path,
	prefix: str = "geom",
	step: int | None = None,
	show_scale: bool = True,
	show_parent_frame: bool = True,
) -> Path:
	"""Create a detailed 3D debug plot for per-node SO(2) frame geometry.

	Visual elements:
	  - Central node (lime), parent (orange), incoming neighbors (green), outgoing neighbors (blue).
	  - Node frame (e1: red arrow, e2: purple arrow, uhat: black arrow).
	  - Parent frame (lighter/dashed arrows) if available.
	  - Edge vectors drawn from node position to neighbor positions.
	  - Par/Perp decomposition of each edge: r_par dashed gray, r_perp solid colored.
	  - Text panel listing angles (deg) for incoming/outgoing edges.
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	tag = f"{prefix}_node{node_id}" + (f"_step{step}" if step is not None else "")
	out_file = out_dir / f"{tag}.png"

	node_pos = pos[node_id]
	parent_pos = pos[parent_id] if (parent_id is not None and parent_id >= 0) else None

	# scaling for quivers
	if len(neighbor_in) + len(neighbor_out) > 0:
		dists = []
		for nid in list(neighbor_in) + list(neighbor_out):
			dists.append((pos[nid] - node_pos).norm().item())
		scale = max(1e-6, sum(dists) / max(1, len(dists))) * 0.5
	else:
		scale = 0.5

	fig = plt.figure(figsize=(11, 6))
	ax = fig.add_subplot(121, projection='3d')

	# Plot neighbors
	legend_handles = []

	def _scatter(ids, color, label, size=50):
		if not ids:
			return None
		pts = pos[th.tensor(ids, dtype=th.long)]
		ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=size, c=color, depthshade=True, label=label, edgecolors='black')
		return Line2D([0],[0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8, markeredgecolor='black')

	# central node
	# central node & parent & neighbors (collect legend handles)
	ax.scatter([node_pos[0]], [node_pos[1]], [node_pos[2]], s=110, c='#32cd32', depthshade=True, label='node', edgecolors='black')
	legend_handles.append(Line2D([0],[0], marker='o', color='w', label='node', markerfacecolor='#32cd32', markersize=9, markeredgecolor='black'))
	if parent_pos is not None:
		ax.scatter([parent_pos[0]], [parent_pos[1]], [parent_pos[2]], s=95, c='#ffa500', depthshade=True, label='parent', edgecolors='black')
		legend_handles.append(Line2D([0],[0], marker='o', color='w', label='parent', markerfacecolor='#ffa500', markersize=9, markeredgecolor='black'))
	h_in = _scatter(list(neighbor_in), '#2ca02c', 'incoming', size=65)
	if h_in is not None:
		legend_handles.append(h_in)
	h_out = _scatter(list(neighbor_out), '#1f77b4', 'outgoing', size=65)
	if h_out is not None:
		legend_handles.append(h_out)

	# Draw edge vectors & decompositions
	# store one-time legend handles for edge vector styles
	edge_handle_in = None
	edge_handle_out = None
	par_handle = None
	perp_handle_in = None
	perp_handle_out = None

	def _draw_edge(vec: th.Tensor, r_par: th.Tensor, r_perp: th.Tensor, base: th.Tensor, color: str, kind: str):
		end = base + vec
		ax.plot([base[0], end[0]], [base[1], end[1]], [base[2], end[2]], color=color, linewidth=2)
		par_end = base + r_par
		ax.plot([base[0], par_end[0]], [base[1], par_end[1]], [base[2], par_end[2]], color='#666666', linestyle='dashed', linewidth=1)
		perp_end = base + r_perp
		ax.plot([base[0], perp_end[0]], [base[1], perp_end[1]], [base[2], perp_end[2]], color=color, linestyle='dotted', linewidth=1)
		# create legend handles once
		nonlocal edge_handle_in, edge_handle_out, par_handle, perp_handle_in, perp_handle_out
		if kind == 'in' and edge_handle_in is None:
			edge_handle_in = Line2D([0],[0], color='#2ca02c', lw=2, label='edge (incoming)')
			perp_handle_in = Line2D([0],[0], color='#2ca02c', lw=1, linestyle='dotted', label='perp (incoming)')
		elif kind == 'out' and edge_handle_out is None:
			edge_handle_out = Line2D([0],[0], color='#1f77b4', lw=2, label='edge (outgoing)')
			perp_handle_out = Line2D([0],[0], color='#1f77b4', lw=1, linestyle='dotted', label='perp (outgoing)')
		if par_handle is None:
			par_handle = Line2D([0],[0], color='#666666', lw=1, linestyle='dashed', label='parallel')

	# Iterate over edge vectors, their decompositions, and angles in sync
	# Incoming edges only: orient arrows from neighbor (source) to node (destination)
	for vec, (r_par, r_perp), (c, s) in zip(edge_vecs_in, edge_decomp_in, angles_in):
		# For incoming edges in PyG, vec = coors[dst] - coors[src] = node_pos - neighbor_pos.
		# Base should be neighbor_pos = node_pos - vec so arrow points into node.
		base = node_pos - vec
		_draw_edge(vec, r_par, r_perp, base, '#2ca02c', 'in')
	# Suppress outgoing edges to reduce clutter (user request)
	# (edge_vecs_out ignored intentionally)

	# add decomposition legend handles if any edges existed
	for h in [edge_handle_in, par_handle, perp_handle_in]:  # exclude outgoing handles entirely
		if h is not None:
			legend_handles.append(h)

	# Frames at node
	def _quiver(base: th.Tensor, direction: th.Tensor, color: str, label: str, lw: float = 3, alpha: float = 1.0):
		tip = base + direction * scale
		ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], color=color, linewidth=lw, alpha=alpha)
		ax.text(tip[0], tip[1], tip[2], label, fontsize=8, color=color)

	_quiver(node_pos, e1_node, 'red', 'e1')
	_quiver(node_pos, e2_node, 'purple', 'e2')
	_quiver(node_pos, uhat, 'black', 'u')
	legend_handles.extend([
		Line2D([0],[0], color='red', lw=3, label='e1 (node)'),
		Line2D([0],[0], color='purple', lw=3, label='e2 (node)'),
		Line2D([0],[0], color='black', lw=3, label='axis u')
	])
	if show_parent_frame and parent_pos is not None and e1_parent is not None and e2_parent is not None:
		_quiver(parent_pos, e1_parent, '#ff9999', 'pe1', lw=2, alpha=0.9)
		_quiver(parent_pos, e2_parent, '#c299ff', 'pe2', lw=2, alpha=0.9)
		legend_handles.extend([
			Line2D([0],[0], color='#ff9999', lw=2, label='e1 (parent)'),
			Line2D([0],[0], color='#c299ff', lw=2, label='e2 (parent)')
		])

	# optional scale reference arrow
	if show_scale:
		ref_dir = th.tensor([1.0, 0.0, 0.0], device=pos.device)
		_quiver(node_pos, ref_dir, '#888888', f'scale={scale:.2f}', lw=2, alpha=0.6)
		legend_handles.append(Line2D([0],[0], color='#888888', lw=2, label='scale ref'))

	ax.view_init(elev=25, azim=35)
	ax.set_title(f"Geometry Debug Node {node_id}")
	ax.set_axis_off()
	# build consolidated legend
	if legend_handles:
		# remove duplicate labels preserving order
		seen = set()
		uniq_handles = []
		for h in legend_handles:
			lbl = h.get_label()
			if lbl not in seen:
				seen.add(lbl)
				uniq_handles.append(h)
		ax.legend(handles=uniq_handles, loc='upper left', fontsize=8, frameon=True)

	# Text panel for angles
	ax2 = fig.add_subplot(122)
	ax2.axis('off')
	lines = []
	if angles_in:
		lines.append("Incoming edges (cos,sin,degrees):")
		for idx, (c, s) in enumerate(angles_in):
			deg = th.rad2deg(th.atan2(th.tensor(s), th.tensor(c))).item()
			lines.append(f"  in[{idx}]: cos={c:.3f} sin={s:.3f} deg={deg:.1f}")
	if angles_out:
		lines.append("Outgoing edges (cos,sin,degrees):")
		for idx, (c, s) in enumerate(angles_out):
			deg = th.rad2deg(th.atan2(th.tensor(s), th.tensor(c))).item()
			lines.append(f"  out[{idx}]: cos={c:.3f} sin={s:.3f} deg={deg:.1f}")
	if parent_pos is not None:
		dist = (node_pos - parent_pos).norm().item()
		lines.append(f"Parent distance: {dist:.3f}")
	ax2.text(0.0, 1.0, "\n".join(lines), va='top', ha='left', fontsize=9, family='monospace')

	fig.tight_layout()
	fig.savefig(out_file, dpi=160)
	plt.close(fig)
	return out_file
