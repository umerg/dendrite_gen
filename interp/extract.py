"""Teacher-forced activation extraction at fixed flow times.

For every reduction item of the chosen splits, runs the real training forward
(``Expansion.teacher_forced_diagnostics``) with t pinned to each value in ``--t`` and a
per-batch seed, records every site, and writes an ``interp.store`` directory per (t, split).

Example::

    python -m interp.extract --config parity_neurons_uncond --ckpt /path/step_218500.pt \
        --which ema --splits val,test --t 0 --out /itet-stor/.../interp/uncond

``--which random`` needs no checkpoint (random-init control). ``--data-dir`` overrides
``dataset.data_dir``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch_geometric.data import Batch

from graph_generation.method.helpers import select_training_leaf_indices

from .hooks import (ActivationRecorder, AttnAblation, InputCapture, all_sites, fixed_flow_time,
                    keyed_flow_noise)
from .load import build_method_and_model, compose_cfg, load_split, tmd_vector, uhat_of
from .reduction_ids import build_items
from .store import split_dir
from .targets import final_node_targets

ORD_SLICE = slice(4, 4 + 23)   # x columns of the 23-wide child-ordinal one-hot (after pos(3), is_leaf)
POS_DIM = 3


def default_sparse_sites(model) -> list[str]:
    n = len(model.mpnn_layers)
    return ["in"] + [f"L{i}" for i in range(1, n, 2)]


def _row_groups(item, rng: np.random.Generator, sparse_frac: float):
    """Per-item row selection (item-local indices): dense = frontier ∪ root children."""
    n = int(item.num_nodes)
    frontier = item.new_leaf_mask_from_next.numpy().astype(bool)
    rootch = (item.parent_idx_1b.numpy() == 1)  # parent is local node 0 (the root)
    dense = frontier | rootch
    other = np.nonzero(~dense)[0]
    take = other[rng.random(other.size) < sparse_frac]
    sparse = dense.copy()
    sparse[take] = True
    tier = np.full(n, "", dtype=object)
    tier[take] = "B2"
    tier[rootch] = "B1"
    tier[frontier] = "A"
    return dense, sparse, frontier, rootch, tier


def extract_split(method, model, cfg, graphs, split: str, ts: list[float], out_root: Path,
                  meta: dict, seed: int = 0, batch_items: int = 64, sparse_frac: float = 0.2,
                  device: str = "cpu", sparse_sites: list[str] | None = None,
                  dtype: str = "float16", ablate: str = "none", mean_batches: int = 20,
                  overwrite: bool = False) -> None:
    """Extract one split at every t. ``ablate`` in {none, zero, mean, graphmean} applies
    ``AttnAblation`` to every global block (P6 route test); "mean" first estimates the
    dataset-mean attn2 read at each t from ``mean_batches`` batches."""
    if model.training:
        raise RuntimeError("extract_split needs the model in eval mode (dropout would noise activations).")
    from graph_generation.diffusion import VFlowMatchingModel
    velocity = isinstance(method.diffusion, VFlowMatchingModel)
    items, per = build_items(graphs, cfg)
    sites = all_sites(model)
    node_sites = [s for s in sites if not s.startswith("G")]
    g_sites = [s for s in sites if s.startswith("G")]
    sparse_sites = sparse_sites or default_sparse_sites(model)
    dense_sites = [s for s in node_sites if s not in sparse_sites]
    F = int(model.feats_dim)
    np_dtype = np.dtype(dtype)
    m_tok = int(model.global_tokens.shape[0])

    # Final-tree targets (t-independent), shared by every t directory.
    uhat = uhat_of(cfg)
    tf = []
    for info in per:
        df = final_node_targets(info["parent"], info["depth"], info["pos"], info["is_apical"], uhat)
        df.insert(0, "neuron", info["neuron"])
        df["neuron_id"] = info["neuron_id"]
        tf.append(df)
    targets_final = pd.concat(tf, ignore_index=True)
    # Per-neuron conditioning-style TMD (for P3/P4 targets; computed for every arm) and metadata.
    neuron_tmd = np.stack([tmd_vector(G, cfg) for G in graphs]).astype(np.float32)
    neuron_meta = pd.DataFrame(dict(
        neuron=[p["neuron"] for p in per], neuron_id=[p["neuron_id"] for p in per],
        cell_class=[p["cell_class"] for p in per], n_levels=[p["n_levels"] for p in per],
        n_nodes=[len(p["parent"]) for p in per]))

    # Row selection is fixed before any forward, so every memmap can be preallocated.
    rng = np.random.default_rng(seed)
    groups = [_row_groups(it, rng, sparse_frac) for it in items]
    n_dense = int(sum(g[0].sum() for g in groups))
    n_sparse = int(sum(g[1].sum() for g in groups))
    batches = [list(range(i, min(i + batch_items, len(items)))) for i in range(0, len(items), batch_items)]

    model.to(device)
    if hasattr(method, "to"):
        method.to(device)
    for t0 in ts:
        d = split_dir(out_root, t0, split)
        if (d / "meta.json").exists() and not overwrite:
            raise FileExistsError(f"{d} already holds an extraction (ablate={ablate}); use a different --out "
                                  "or --overwrite.")
        (d / "acts").mkdir(parents=True, exist_ok=True)
        (d / "graph_acts").mkdir(parents=True, exist_ok=True)
        mm = {}
        for s in node_sites:
            n = n_sparse if s in sparse_sites else n_dense
            mm[s] = np.lib.format.open_memmap(d / "acts" / f"{s}.npy", mode="w+", dtype=np_dtype, shape=(n, F))
        gmm = {s: np.lib.format.open_memmap(d / "graph_acts" / f"{s}.npy", mode="w+", dtype=np_dtype,
                                            shape=(len(items), m_tok, F)) for s in g_sites}
        pool = {s: np.lib.format.open_memmap(d / "graph_acts" / f"pool_{s}.npy", mode="w+", dtype=np_dtype,
                                             shape=(len(items), F)) for s in node_sites}
        overflow = {s: 0 for s in [*node_sites, *g_sites]}

        def put(arr, key, sl, val):
            v = val.cpu().numpy().astype(np_dtype)
            overflow[key] += int((~np.isfinite(v)).sum())
            arr[sl] = v

        index_rows, graph_rows = [], []
        r_dense = r_sparse = 0
        tic = time.time()
        abl_means = None
        if ablate == "mean":
            rec_abl = AttnAblation(model, mode="record")
            with fixed_flow_time(method.diffusion, t0), rec_abl, th.no_grad():
                step = max(1, len(batches) // mean_batches)   # evenly spread, not the first neurons
                for bi, idxs in enumerate(batches[::step][:mean_batches]):
                    b_ = Batch.from_data_list([items[i] for i in idxs]).to(device)
                    with keyed_flow_noise(_leaf_keys(b_, seed, t0)):
                        method.teacher_forced_diagnostics(b_, model)
            abl_means = rec_abl.recorded_means()
        for bi, idxs in enumerate(batches):
            batch = Batch.from_data_list([items[i] for i in idxs]).to(device)
            outs = []
            h = model.register_forward_hook(lambda m, a, o: outs.append(o))
            try:
                abl = (AttnAblation(model, mode=ablate, means=abl_means) if ablate != "none"
                       else contextlib.nullcontext())
                with fixed_flow_time(method.diffusion, t0), abl, keyed_flow_noise(_leaf_keys(batch, seed, t0)), \
                        ActivationRecorder(model, sites, dtype=th.float32, device=device) as rec, \
                        InputCapture(model) as cap, th.no_grad():
                    th.manual_seed(_batch_seed(seed, t0, bi))  # nothing else should draw; belt and braces
                    diag = method.teacher_forced_diagnostics(batch, model)
            finally:
                h.remove()
            acts = {s: v[0] for s, v in rec.pop().items()}
            x = cap.last["x"]
            parent = cap.last["parent_idx"]
            e_t = x[:, -2].clone()
            t_col = x[:, -1]
            ord_onehot = x[:, ORD_SLICE]
            slot = th.where(ord_onehot.sum(1) > 0, ord_onehot.argmax(1), th.full_like(parent, -1))
            ev = outs[-1]["expansion_pred"].reshape(-1)
            # implied clean expansion: velocity model → e_t + (1-t)·v ; data-prediction → output itself
            e1 = e_t + (1.0 - t_col) * ev if velocity else ev.clone()
            pos_err = th.full_like(e_t, float("nan"))
            leaf = diag["leaf_idx_train"]
            # e_t / e1 only mean something on training leaves (the frontier); NaN elsewhere
            not_leaf = th.ones_like(e_t, dtype=th.bool)
            not_leaf[leaf] = False
            e_t[not_leaf] = float("nan")
            e1[not_leaf] = float("nan")
            pos_err[leaf] = (diag["cp"] - diag["c0"]).pow(2).sum(1)

            ptr = batch.ptr.cpu().numpy()
            bvec = batch.batch
            for s in node_sites:  # graph mean-pool over all nodes
                a = acts[s]
                sums = a.new_zeros((len(idxs), a.size(1))).index_add_(0, bvec, a)
                cnt = th.bincount(bvec, minlength=len(idxs)).clamp_min(1).unsqueeze(1)
                put(pool[s], s, slice(idxs[0], idxs[-1] + 1), sums / cnt)
            for s in g_sites:
                put(gmm[s], s, slice(idxs[0], idxs[-1] + 1), acts[s].reshape(len(idxs), m_tok, F))

            dense_g = np.concatenate([groups[i][0] for i in idxs])
            sparse_g = np.concatenate([groups[i][1] for i in idxs])
            nd, ns = int(dense_g.sum()), int(sparse_g.sum())
            dg, sg = th.from_numpy(dense_g).to(device), th.from_numpy(sparse_g).to(device)
            for s in node_sites:
                if s in sparse_sites:
                    put(mm[s], s, slice(r_sparse, r_sparse + ns), acts[s][sg])
                else:
                    put(mm[s], s, slice(r_dense, r_dense + nd), acts[s][dg])

            # index rows (every sparse-group row; dense rows are a subset)
            row_d = np.full(sparse_g.size, -1, dtype=np.int64)
            row_d[dense_g] = np.arange(r_dense, r_dense + nd)
            row_s = np.full(sparse_g.size, -1, dtype=np.int64)
            row_s[sparse_g] = np.arange(r_sparse, r_sparse + ns)
            cpu = lambda v: v.detach().cpu().numpy()  # noqa: E731
            par, slot_n, e_t_n, e1_n, perr_n = map(cpu, (parent, slot, e_t, e1, pos_err))
            for k, i in enumerate(idxs):
                it, lo, hi = items[i], ptr[k], ptr[k + 1]
                keep = np.nonzero(groups[i][1])[0]
                gl = lo + keep
                rootchild = groups[i][3][keep]
                index_rows.append(pd.DataFrame(dict(
                    neuron=int(it.neuron), level=int(it.level), n_levels=int(it.n_levels), item=i,
                    node=it.fine_idx.numpy()[keep], local=keep, tier=groups[i][4][keep],
                    is_frontier=groups[i][2][keep], is_root_child=rootchild,
                    slot=slot_n[gl], t=float(t0), e_t=e_t_n[gl], e1_implied=e1_n[gl], pos_err=perr_n[gl],
                    row_dense=row_d[gl - ptr[0]], row_sparse=row_s[gl - ptr[0]],
                )))
                graph_rows.append(dict(
                    item=i, neuron=int(it.neuron), neuron_id=per[int(it.neuron)]["neuron_id"],
                    level=int(it.level), n_levels=int(it.n_levels), n_nodes=int(hi - lo),
                    n_frontier=int(groups[i][2].sum()), cell_class=per[int(it.neuron)]["cell_class"],
                ))
            r_dense += nd
            r_sparse += ns
            if bi % 50 == 0:
                print(f"[extract {split} t={t0:g}] batch {bi + 1}/{len(batches)} {time.time() - tic:.0f}s", flush=True)

        assert r_dense == n_dense and r_sparse == n_sparse, (r_dense, n_dense, r_sparse, n_sparse)
        for a in [*mm.values(), *gmm.values(), *pool.values()]:
            a.flush()
        pd.concat(index_rows, ignore_index=True).to_parquet(d / "index.parquet")
        pd.DataFrame(graph_rows).to_parquet(d / "graph_index.parquet")
        targets_final.to_parquet(d / "targets_final.parquet")
        np.save(d / "neuron_tmd.npy", neuron_tmd)
        neuron_meta.to_parquet(d / "neuron_meta.parquet")
        (d / "meta.json").write_text(json.dumps({
            **meta, "t": float(t0), "split": split, "seed": seed, "batch_items": batch_items,
            "sparse_frac": sparse_frac, "sites_dense": dense_sites, "sites_sparse": sparse_sites,
            "sites_graph": g_sites, "n_dense": n_dense, "n_sparse": n_sparse, "n_items": len(items),
            "neuron_ids": [p["neuron_id"] for p in per], "dtype": dtype, "ablate": ablate,
            "overflow": {k: v for k, v in overflow.items() if v},
        }, indent=1, default=str))
        if any(overflow.values()):
            print(f"[extract {split} t={t0:g}] WARNING: {dtype} overflow at sites "
                  f"{sorted(k for k, v in overflow.items() if v)} — rerun with --dtype float32 "
                  "if these sites are needed.", flush=True)
        print(f"[extract {split} t={t0:g}] done: {n_dense} dense / {n_sparse} sparse rows, "
              f"{len(items)} items in {time.time() - tic:.0f}s -> {d}", flush=True)


def _leaf_keys(batch, seed: int, t0: float) -> np.ndarray:
    """(seed, neuron, level, node) per training leaf, in the forward's leaf order.

    t is deliberately *not* in the key: every flow time uses the same prior draw for a node, so
    a t-sweep compares the same noise sample at different interpolation points."""
    leaf = select_training_leaf_indices(batch).cpu()
    b = batch.batch.cpu()[leaf]
    return np.stack([np.full(len(leaf), seed),
                     batch.neuron.cpu()[b].numpy(), batch.level.cpu()[b].numpy(),
                     batch.fine_idx.cpu()[leaf].numpy()], axis=1)


def _batch_seed(seed: int, t0: float, bi: int) -> int:
    return (int(seed) * 1_000_003 + int(round(t0 * 1000)) * 10_007 + bi) % (2**31 - 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--which", default="ema", help="ema | raw | random[:seed]")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--max-neurons", type=int, default=None, help="per split (debug)")
    ap.add_argument("--t", default="0", help="comma-separated flow times (analyses use t=0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-items", type=int, default=64)
    ap.add_argument("--sparse-frac", type=float, default=0.2)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting an existing extraction")
    ap.add_argument("--ablate", default="none", choices=["none", "zero", "mean", "graphmean"],
                    help="ablate the tokens->nodes global read in every global block (P6)")
    args = ap.parse_args(argv)

    device = ("cuda" if th.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    overrides = [f"dataset.data_dir={args.data_dir}"] if args.data_dir else []
    cfg = compose_cfg(args.config, overrides)
    which, _, init_seed = args.which.partition(":")
    pool = load_split(cfg, "val", max_n=2)
    method, model, meta = build_method_and_model(cfg, args.ckpt, which=which,
                                                 seed=int(init_seed or 0), pool=pool)
    meta["config"] = args.config
    ts = [float(v) for v in args.t.split(",")]
    for split in args.splits.split(","):
        graphs = load_split(cfg, split, max_n=args.max_neurons)
        extract_split(method, model, cfg, graphs, split, ts, Path(args.out), meta, seed=args.seed,
                      batch_items=args.batch_items, sparse_frac=args.sparse_frac, device=device,
                      dtype=args.dtype, ablate=args.ablate, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
