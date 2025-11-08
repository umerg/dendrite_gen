import os
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
from torch_sparse import SparseTensor

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
