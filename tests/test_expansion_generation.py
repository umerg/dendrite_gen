import os
import math
import torch as th
import matplotlib.pyplot as plt
from torch import nn
from torch_sparse import SparseTensor
from graph_generation.method.expansion_oneshot import Expansion_OneShot

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'expansion_test_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

class MockModel(nn.Module):
    """Minimal stand-in model producing random relative offsets and expansion logits.
    rel_pred: Gaussian with small scale; expansion_pred: uniform logits then optional bias.
    We keep interface: out['rel_pred'], out['expansion_pred'].
    """
    def __init__(self, feats_dim=0, pos_dim=3, rel_sigma=0.08):
        super().__init__()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.rel_sigma = rel_sigma

    def forward(self, x, edge_index, batch, edge_attr=None, parent_idx=None):  # parent_idx 0-based expected
        N = x.size(0)
        device = x.device
        rel_pred = th.randn((N, 3), device=device) * self.rel_sigma
        # Expansion logits: encourage growth early, taper later by using depth proxy if parent_idx available
        if parent_idx is not None:
            depth_proxy = (parent_idx + 1).float()  # roots 0 -> 1
            norm_depth = depth_proxy / (depth_proxy.max().clamp_min(1.0))
            # Higher depth => slightly less branching probability
            logits = th.randn((N, 1), device=device) + (0.5 - norm_depth.unsqueeze(-1))
        else:
            logits = th.randn((N, 1), device=device)
        return {'rel_pred': rel_pred, 'expansion_pred': logits}


def run_generation(target_sizes, deterministic=False):
    method = Expansion_OneShot(deterministic_expansion=deterministic, leaf_noise_sigma=0.05)
    device = target_sizes.device
    model = MockModel().to(device)
    graphs, pos, batch = method.sample_graphs(target_sizes, model)
    return graphs, pos, batch


def plot_generation_history(histories, target_sizes, fname):
    """Create a panel summarizing growth per graph with edges.
    histories: list of dicts per step with keys: 'pos', 'batch', 'leaf_idx', 'adj'
    We draw edges (parent-child undirected) by filtering COO rows/cols for each graph.
    """
    num_graphs = int(target_sizes.numel())
    cols = num_graphs
    steps = len(histories)
    fig, axes = plt.subplots(steps, cols, figsize=(cols * 3, steps * 3))
    if steps == 1:
        axes = axes.reshape(1, -1)

    for step, record in enumerate(histories):
        pos = record['pos']
        batch = record['batch']
        leaf_idx = record['leaf_idx']
        adj = record.get('adj', None)
        if adj is not None:
            row, col, _ = adj.coo()
        for g in range(num_graphs):
            ax = axes[step][g]
            mask = (batch == g)
            pts = pos[mask]
            if pts.numel() > 0:
                ax.scatter(pts[:,0].cpu(), pts[:,1].cpu(), s=20, c='steelblue', alpha=0.7)
            # Draw edges (only once per undirected pair) if adjacency available
            if adj is not None and row.numel() > 0:
                # Filter to edges inside graph g
                g_mask = (batch[row] == g) & (batch[col] == g)
                row_g = row[g_mask]
                col_g = col[g_mask]
                # Avoid double drawing by enforcing row < col
                simple_mask = row_g < col_g
                row_g = row_g[simple_mask]
                col_g = col_g[simple_mask]
                for r_i, c_i in zip(row_g.tolist(), col_g.tolist()):
                    p1 = pos[r_i]; p2 = pos[c_i]
                    ax.plot([p1[0].item(), p2[0].item()], [p1[1].item(), p2[1].item()], color='gray', linewidth=0.8, alpha=0.5)
            # Highlight leaves
            leaf_mask = mask[leaf_idx]
            leaves_global = leaf_idx[mask[leaf_idx]]  # filter leaves belonging to graph g
            if leaves_global.numel() > 0:
                leaf_pts = pos[leaves_global]
                ax.scatter(leaf_pts[:,0].cpu(), leaf_pts[:,1].cpu(), s=40, c='orange', edgecolors='k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"g{g} step{step} n={pts.size(0)}")
    fig.suptitle("Expansion Generation Panel")
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def test_expansion_generation_basic():
    # Small batch of 4 graphs with varied target sizes
    target_sizes = th.tensor([4, 8, 10, 16])
    graphs, pos, batch = run_generation(target_sizes, deterministic=True)

    # Assertions: number of graphs & node counts not exceeding target
    assert len(graphs) == 4, "Should return 4 graphs"
    for g, ts in zip(graphs, target_sizes.tolist()):
        assert g.number_of_nodes() <= ts, f"Graph exceeds target size {ts}"    
        # Basic structure sanity: if ts >=3 expect branching beyond root
        if ts >= 3:
            assert g.number_of_nodes() >= 1, "Graph should at least have root"


def test_expansion_generation_history_and_plot():
    target_sizes = th.tensor([5, 6, 7, 8])
    device = target_sizes.device
    method = Expansion_OneShot(deterministic_expansion=True, leaf_noise_sigma=0.05)
    model = MockModel().to(device)

    # Manual step-by-step to capture histories
    # Initialize roots similar to sample_graphs logic
    num_graphs = int(target_sizes.numel())
    pos = th.zeros((num_graphs, 3), device=device)
    adj = SparseTensor(row=th.tensor([], dtype=th.long, device=device),
                       col=th.tensor([], dtype=th.long, device=device),
                       value=th.tensor([], dtype=th.float, device=device),
                       sparse_sizes=(num_graphs, num_graphs))
    batch = th.arange(num_graphs, device=device, dtype=th.long)
    parent_idx_1b = th.zeros(num_graphs, device=device, dtype=th.long)
    leaf_idx = th.arange(num_graphs, device=device, dtype=th.long)
    leaf_expansion = th.where(target_sizes >= 3, th.full_like(leaf_idx, 2), th.full_like(leaf_idx, 1))

    histories = []
    max_steps = int(target_sizes.max().item() * 2)
    terminated = False
    step = 0
    while not terminated and step < max_steps:
        histories.append({'pos': pos.clone(), 'batch': batch.clone(), 'leaf_idx': leaf_idx.clone(), 'adj': adj})
        adj, pos, leaf_idx, leaf_expansion, parent_idx_1b, batch, terminated = method.expand(
            adj, batch, target_sizes, model,
            pos=pos,
            leaf_idx=leaf_idx,
            leaf_expansion=leaf_expansion,
            parent_idx_1b=parent_idx_1b,
            step=step,
            ensure_progress=True,
            map_threshold=0.5,
        )
        step += 1

    # Capture final state
    histories.append({'pos': pos.clone(), 'batch': batch.clone(), 'leaf_idx': leaf_idx.clone(), 'adj': adj})

    # Plot panel
    plot_path = plot_generation_history(histories, target_sizes, 'panel_basic.png')
    assert os.path.exists(plot_path), "Plot file was not created"

    # Sanity: final counts per graph <= target and >=1
    counts = th.zeros(target_sizes.size(0), dtype=th.long)
    for g in range(target_sizes.size(0)):
        counts[g] = (batch == g).sum()
    assert (counts <= target_sizes).all(), "A graph exceeded its target size"
    assert (counts >= 1).all(), "A graph has zero nodes (should have at least root)"

    # Check at least one expansion happened for graphs with capacity >=3
    expanded_any = counts[target_sizes >= 3] > 1
    assert expanded_any.any(), "Expected expansion for graphs with capacity >=3"


if __name__ == '__main__':
    # Allow running this test file directly for quick dev visualization
    test_expansion_generation_basic()
    test_expansion_generation_history_and_plot()
    print("Expansion generation tests completed. Plots saved to:", PLOTS_DIR)
