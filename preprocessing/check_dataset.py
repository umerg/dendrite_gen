#!/usr/bin/env python
"""Dataset verification script.

Checks a directory of cleaned SWC neuron tree files for structural constraints:

Conditions enforced (failures recorded per file):
  * Only allowed node types are: root, binary branching nodes (exactly 2 children), terminal leaves (0 children).
	Concretely: for every non-root node, number of children must be 0 or 2.
  * Root must be the first node in the adjacency ordering produced by `nx_graph_to_adj_pos` (insertion order).
  * Root position must be exactly (0, 0, 0) within a tolerance.
  * (Optional) Root must itself have exactly 2 children if `--enforce-root-binary` is passed.

Outputs a summary plus optional JSON with detailed failures.

Usage:
  python preprocessing/check_dataset.py /path/to/dir
  python preprocessing/check_dataset.py /path/to/dir --json report.json
  python preprocessing/check_dataset.py /path/to/dir --enforce-root-binary --tolerance 1e-6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import networkx as nx

# Allow running as a standalone script without needing PYTHONPATH or -m invocation.
# We insert the repository root (parent of this file's directory) into sys.path so that
# 'utils' and other top-level modules can be imported reliably.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from utils.data_loading import load_swc_graph, nx_graph_to_adj_pos


def iter_swc_files(dir_path: Path):
	"""Yield SWC file paths matching inclusion criteria (mirrors helper logic)."""
	for swc_file in sorted(dir_path.iterdir()):
		if not swc_file.is_file():
			continue
		name = swc_file.name
		if name.startswith("._"):
			continue
		if not name.endswith(".csv.swc"):
			continue
		yield swc_file


def children_counts(G: nx.Graph, root: int) -> Dict[int, int]:
	"""Return mapping node -> number of children relative to the chosen root.

	We do a BFS outward; for each visited node its children are its neighbors excluding its parent.
	Root has no parent so its children count is its degree.
	"""
	counts = {}
	parent = {root: None}
	queue = [root]
	for nid in queue:
		nbrs = list(G.neighbors(nid))
		p = parent[nid]
		if p is not None:
			# exclude parent from children
			child_list = [x for x in nbrs if x != p]
		else:
			child_list = nbrs
		counts[nid] = len(child_list)
		for c in child_list:
			parent[c] = nid
			queue.append(c)
	return counts


def verify_graph(G: nx.Graph, file_path: Path, tolerance: float, enforce_root_binary: bool) -> List[str]:
	"""Verify a single graph, returning a list of failure reason codes."""
	failures: List[str] = []

	# Obtain adjacency ordering; first node should be root candidate
	A, P, node_order = nx_graph_to_adj_pos(G)
	root_candidate = int(node_order[0])

	# Children counts relative to root
	ccounts = children_counts(G, root_candidate)

	# Root first? (we define root as node with (0,0,0) OR first insertion order). We only check ordering.
	# If some other node has (0,0,0) position but appears later, we still treat first as root per spec.
	# (Spec: "position 0 node is the root").

	# Ensure root position is (0,0,0)
	root_pos = P[0]  # matches node_order[0]
	if not (abs(root_pos[0]) <= tolerance and abs(root_pos[1]) <= tolerance and abs(root_pos[2]) <= tolerance):
		failures.append("root_pos_not_zero")

	# Determine allowed children counts for non-root nodes: 0 (leaf) or 2 (binary branching)
	for nid, child_count in ccounts.items():
		if nid == root_candidate:
			continue
		if child_count not in (0, 2):
			failures.append("nonroot_not_binary_or_leaf")
			break  # One failure sufficient for structure

	if enforce_root_binary:
		root_children = ccounts[root_candidate]
		if root_children != 2:
			failures.append("root_children_not_2")

	# Additional structural check: ensure no node has >2 children (strict binary)
	for nid, child_count in ccounts.items():
		if child_count > 2:
			failures.append("node_has_>2_children")
			break

	# Sanity: the tree should be a tree (single component & acyclic)
	if not nx.is_tree(G):  # should never happen given loader validation
		failures.append("not_tree")

	return failures


def main():
	parser = argparse.ArgumentParser(description="Verify cleaned SWC tree dataset structure.")
	parser.add_argument("directory", type=Path, help="Directory containing .csv.swc cleaned files")
	parser.add_argument("--json", type=Path, help="Optional path to write JSON report")
	parser.add_argument("--tolerance", type=float, default=0.0, help="Tolerance for root position being (0,0,0)")
	parser.add_argument(
		"--enforce-root-binary",
		action="store_true",
		help="Require root to have exactly two children (binary root).",
	)
	args = parser.parse_args()

	if not args.directory.exists() or not args.directory.is_dir():
		raise SystemExit(f"Directory not found or not a directory: {args.directory}")

	results = []
	total = 0
	failures_total = 0

	for swc_path in iter_swc_files(args.directory):
		total += 1
		try:
			G = load_swc_graph(swc_path)
			failures = verify_graph(
				G,
				swc_path,
				tolerance=args.tolerance,
				enforce_root_binary=args.enforce_root_binary,
			)
		except Exception as e:  # Capture loader exceptions as file-level failures
			failures = [f"exception:{type(e).__name__}"]
		if failures:
			failures_total += 1
		results.append({
			"file": swc_path.name,
			"failures": failures,
		})

	# Summary
	print("Dataset Verification Summary")
	print("Directory:", args.directory)
	print(f"Files checked: {total}")
	print(f"Files failing any condition: {failures_total}")
	if failures_total:
		# Aggregate failure types counts
		agg = {}
		for r in results:
			for f in r["failures"]:
				agg[f] = agg.get(f, 0) + 1
		print("Failure type counts:")
		for k, v in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0])):
			print(f"  {k}: {v}")
		# List failing file names for easy inspection
		failing_files = [r["file"] for r in results if r["failures"]]
		if failing_files:
			print("Failing files:")
			for fname in failing_files:
				print(f"  {fname}")
	else:
		print("All files passed the verification conditions.")

	# Optional JSON output
	if args.json:
		payload = {
			"directory": str(args.directory),
			"total_files": total,
			"failed_files": failures_total,
			"results": results,
			"tolerance": args.tolerance,
			"enforce_root_binary": args.enforce_root_binary,
		}
		with args.json.open("w") as f:
			json.dump(payload, f, indent=2)
		print(f"Wrote JSON report to {args.json}")


if __name__ == "__main__":
	main()

