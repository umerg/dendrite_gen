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

def plot_nx_3d(G: nx.Graph, out_file: Path, elev: int = 20, azim: int = 45):
	"""Plot a 3D scatter with edges using stored 'pos' node attributes.

	Saves figure to out_file (parent directory created if needed).
	"""
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111, projection='3d')

	xs, ys, zs = [], [], []
	for n, data in G.nodes(data=True):
		x,y,z = data['pos']
		xs.append(x); ys.append(y); zs.append(z)
	ax.scatter(xs, ys, zs, s=25, c='steelblue', depthshade=True)

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

def plot_gt_and_masked(adj: SparseTensor, pos_gt: th.Tensor, pos_masked: th.Tensor, out_dir: Path, prefix: str, step: int, batch_id: int):
	"""Helper to plot ground-truth vs masked input graphs in 3D.

	Creates two files:
	  {prefix}_gt_step{step}_b{batch_id}.png
	  {prefix}_masked_step{step}_b{batch_id}.png
	"""
	out_dir.mkdir(parents=True, exist_ok=True)
	G_gt = adj_pos_to_nx(adj, pos_gt)
	G_masked = adj_pos_to_nx(adj, pos_masked)
	gt_file = out_dir / f"{prefix}_gt_step{step}_b{batch_id}.png"
	masked_file = out_dir / f"{prefix}_masked_step{step}_b{batch_id}.png"
	plot_nx_3d(G_gt, gt_file)
	plot_nx_3d(G_masked, masked_file)
	return gt_file, masked_file
