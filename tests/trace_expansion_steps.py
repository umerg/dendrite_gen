"""Trace expansion predictions step-by-step during sampling.

Usage:
    conda run -n NEURO2 python tests/trace_expansion_steps.py \
        --checkpoint full_run_outs/neuron_run_1_30K.pt \
        --gt-dir /Volumes/Seagate/neurons_v1/train
"""
from __future__ import annotations
import argparse, sys, torch as th, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loading import load_swc_graph
from utils.tmd import compute_tmd_mixed, tmd_conditioning_dim


def build_model_and_method(checkpoint_path: str, device: str = "cpu"):
    """Build model and method from checkpoint metadata."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parent.parent / "config"),
        version_base=None,
    )
    cfg = compose(config_name="small_trees_run")

    # TMD conditioning knobs (tmd_in_dim is derived, matching main.py).
    tmd_hidden_dim = getattr(cfg.model, "tmd_hidden_dim", 0)
    tmd_filtrations = list(getattr(cfg.model, "tmd_filtrations", ("path", "height", "rho")))
    tmd_bins = int(getattr(cfg.model, "tmd_bins", 16))
    tmd_in_dim = tmd_conditioning_dim(tmd_filtrations, tmd_bins) if tmd_hidden_dim > 0 else 0

    # Build model
    from graph_generation.model.egnn_so2 import SO2_EGNN_Network
    model = SO2_EGNN_Network(
        n_layers=cfg.model.num_layers,
        feats_dim=cfg.model.feats_dim,
        pos_dim=3,
        m_dim=cfg.model.m_dim,
        edge_embedding_nums=[2],
        edge_embedding_dims=[4],
        edge_attr_dim=1,
        dropout=cfg.model.dropout,
        norm_feats=cfg.model.norm_feats,
        global_linear_attn_every=cfg.model.global_linear_attn_every,
        global_linear_attn_heads=cfg.model.global_linear_attn_heads,
        global_linear_attn_dim_head=cfg.model.global_linear_attn_dim_head,
        num_global_tokens=cfg.model.num_global_tokens,
        offset_head_hidden=cfg.model.offset_head_hidden,
        tmd_in_dim=tmd_in_dim,
        tmd_hidden_dim=tmd_hidden_dim,
        so2_axis=cfg.model.so2_axis,
    )

    # Build method + diffusion
    from graph_generation.diffusion.basic import DenoisingDiffusionModel
    diffusion = DenoisingDiffusionModel(num_steps=cfg.diffusion.num_steps)

    from graph_generation.method.expansion import Expansion
    method = Expansion(diffusion=diffusion)

    # Load checkpoint — EMA1 (beta=1) is a pass-through wrapper, actual weights are under "model"
    ckpt = th.load(checkpoint_path, map_location=device)
    state = ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    method.to(device)

    return model, method, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    model, method, cfg = build_model_and_method(args.checkpoint, device)

    # Monkey-patch expand to log details
    import graph_generation.method.expansion as exp_mod
    from torch_scatter import scatter

    _orig_expand = exp_mod.Expansion.expand
    expand_log = []

    MAX_TRACE_STEPS = 30  # Hard cap to avoid CPU runaway

    def _logging_expand(self, adj_reduced, batch_reduced, target_size, model_arg, **kwargs):
        step = kwargs.get("step", -1)

        # Call original
        result = _orig_expand(self, adj_reduced, batch_reduced, target_size, model_arg, **kwargs)
        # Force termination after MAX_TRACE_STEPS
        if step >= MAX_TRACE_STEPS:
            result = (*result[:7], True)  # set terminated=True
        adj, pos, leaf_idx, leaf_exp, pidx, batch, lmask, terminated = result

        n_nodes = pos.size(0)
        n_leaves = leaf_idx.numel()
        n_expand = (leaf_exp == 2).sum().item() if n_leaves > 0 else 0
        n_stop = (leaf_exp == 1).sum().item() if n_leaves > 0 else 0

        # Count old terminal leaves (is_leaf but not in leaf_idx_next)
        total_leaves_in_graph = lmask.sum().item()
        old_terminal = total_leaves_in_graph - n_leaves

        num_graphs = int(target_size.numel())
        nc = scatter(th.ones_like(batch, dtype=target_size.dtype), batch, dim=0, dim_size=num_graphs)
        ratios = (nc / target_size.float().clamp_min(1)).tolist()

        expand_log.append({
            "step": step,
            "n_nodes": n_nodes,
            "n_new_leaves": n_leaves,
            "old_terminal": int(old_terminal),
            "expand": n_expand,
            "stop": n_stop,
            "ratios": ratios,
            "terminated": terminated,
        })
        return result

    exp_mod.Expansion.expand = _logging_expand

    # Load GT graphs — pick only the smallest one for fast CPU debugging
    gt_dir = Path(args.gt_dir)
    gt_files = sorted([f for f in gt_dir.glob("*.csv.swc") if not f.name.startswith("._")])
    # Sort by size, start with smallest
    gt_with_size = [(f, load_swc_graph(f).number_of_nodes()) for f in gt_files]
    gt_with_size.sort(key=lambda x: x[1])

    for fi, (f, _) in enumerate(gt_with_size):
        G = load_swc_graph(f)
        root = G.graph.get("root", 0)
        k = G.degree[root]
        n = G.number_of_nodes()

        target = th.tensor([n], device=device)
        nrc = th.tensor([k], device=device)
        _uhat = model.uhat.detach().cpu().numpy().reshape(-1) if getattr(model, "uhat", None) is not None else np.array([0., 0., 1.])
        _fils = list(getattr(cfg.model, "tmd_filtrations", ("path", "height", "rho")))
        _bins = int(getattr(cfg.model, "tmd_bins", 16))
        tmd = th.tensor(compute_tmd_mixed(G, filtrations=_fils, n_bins=_bins, uhat=_uhat), dtype=th.float32).unsqueeze(0).to(device)

        expand_log.clear()
        # Temporarily limit max_steps to avoid runaway expansion on CPU
        old_max = None
        with th.no_grad():
            graphs = method.sample_graphs(target, model, tmd=tmd, num_root_children=nrc)

        gen_n = graphs[0].number_of_nodes()
        print(f"\n{'='*75}")
        print(f"Graph {fi}: {f.name} | GT N={n}, K={k} | Generated N={gen_n}")
        print(f"{'='*75}")
        print(f"{'step':>4} {'N':>5} {'new_L':>6} {'old_T':>6} {'expand':>7} {'stop':>5} "
              f"{'exp%':>5} {'ratio':>7} {'term':>5}")
        print(f"{'-'*4:>4} {'-'*5:>5} {'-'*6:>6} {'-'*6:>6} {'-'*7:>7} {'-'*5:>5} "
              f"{'-'*5:>5} {'-'*7:>7} {'-'*5:>5}")

        for entry in expand_log:
            r = entry["ratios"][0] if entry["ratios"] else 0
            nl = entry["n_new_leaves"]
            exp_pct = (entry["expand"] / nl * 100) if nl > 0 else 0
            print(
                f"{entry['step']:>4} {entry['n_nodes']:>5} {nl:>6} "
                f"{entry['old_terminal']:>6} {entry['expand']:>7} {entry['stop']:>5} "
                f"{exp_pct:>4.0f}% {r:>7.3f} {str(entry['terminated']):>5}"
            )
            sys.stdout.flush()

        sys.stdout.flush()


if __name__ == "__main__":
    main()
