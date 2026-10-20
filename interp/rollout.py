"""Generation-time interventions (methodology §11b): a standalone rollout driver.

``generate`` mirrors the generation block of ``Trainer._evaluate_rollout``: target size = GT
node count, root children = GT root degree, the GT neuron's TMD for the TMD arm, then
``method.sample_graphs`` and positions × ``pos_scale_factor``. Batches are fixed and
size-sorted (no permutation), so every condition sees the same batches.

Lineage-keyed noise (``keyed=True``). The sampler draws its priors C [L,3] and e [L,1] over
all frontier leaves of the batch, so once an intervention changes how many leaves exist,
every later draw shifts. Here every node gets a 64-bit key, root = H(seed, neuron),
child = H(parent key, child rank), and each leaf's prior (and the root's random frame angle)
is a pure function of its key. An unedited lineage then sees identical noise in every
condition: differences there come from the model (global attention, graph LayerNorm).

``RolloutState`` is updated once per expand step and read by editors / ablation masks:
``owner`` (spawn slot of the soma child a node descends from; slot 0 = model apical, -1 = soma),
``frontier`` (bool over nodes), ``t`` (flow time of the current network call).

E3 target selection (``select_dmin``): at each expand step, every graph without a target picks
the basal frontier node (owner ≥ 1, depth ≥ select_dmin) with the smallest lineage key. Before
the first edit every condition is identical (keyed noise), so all conditions pick the same node;
its key is written to ``G.graph["target_key"]``. Masks ``target`` (the node, while it is on the
frontier) and ``target_lineage`` (the node and, later, its frontier descendants).

CLI (one experiment, many conditions, one job)::

    python -m interp.rollout --config parity_neurons_uncond --ckpt /.../step_218500.pt \\
        --exp e0 --split val --max-neurons 256 --seeds 0,1 --out /.../rollouts
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import torch as th

from . import hooks as H

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_C1, _C2, _C3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


# ----------------------------------------------------------------------------- keyed randomness

def splitmix(x) -> np.ndarray:
    """SplitMix64 finaliser, vectorised over uint64 (wraps mod 2^64)."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _C1
        z = (z ^ (z >> np.uint64(30))) * _C2
        z = (z ^ (z >> np.uint64(27))) * _C3
        return z ^ (z >> np.uint64(31))


def child_keys(parent_keys, rank) -> np.ndarray:
    return splitmix(np.asarray(parent_keys, np.uint64) ^ splitmix(np.asarray(rank, np.uint64) + np.uint64(1)))


def uniforms(keys, n: int, stream: int = 0) -> np.ndarray:
    """[len(keys), n] uniforms in (0, 1), a pure function of (key, stream, column)."""
    keys = np.asarray(keys, np.uint64)
    cols = splitmix(np.arange(n, dtype=np.uint64) + np.uint64(1000 * (stream + 1)))
    bits = splitmix(keys[:, None] ^ cols[None, :]) >> np.uint64(11)
    return (bits.astype(np.float64) + 0.5) * 2.0 ** -53


def normals(keys, n: int) -> np.ndarray:
    """[len(keys), n] standard normals (Box–Muller on keyed uniforms)."""
    m = (n + 1) // 2
    u1, u2 = uniforms(keys, m, 1), uniforms(keys, m, 2)
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)], axis=1)
    return z[:, :n]


# ----------------------------------------------------------------------------- per-rollout state

@dataclass
class RolloutState:
    """Node bookkeeping for one ``sample_graphs`` call, grown in step with the sampler."""
    keys: np.ndarray                       # uint64 [N]
    owner: np.ndarray                      # int64 [N]; -1 = soma
    depth: np.ndarray                      # int64 [N]
    device: str = "cpu"
    t: float = 0.0
    step: int = -1
    frontier_t: th.Tensor | None = None    # bool [N]
    owner_t: th.Tensor | None = None       # long [N]
    batch_t: th.Tensor | None = None       # long [N]
    n_degenerate_frames: list = field(default_factory=list)
    select_dmin: int | None = None         # E3: enable target selection at this generated depth
    under: np.ndarray | None = None        # bool [N]: target or its descendant
    targets: dict = field(default_factory=dict)        # graph j -> global node index
    target_step: dict = field(default_factory=dict)    # graph j -> expand step of selection
    is_target_t: th.Tensor | None = None
    under_t: th.Tensor | None = None

    @classmethod
    def new(cls, seed: int, graph_keys, device="cpu") -> "RolloutState":
        g = np.asarray(graph_keys, dtype=np.uint64)
        keys = splitmix(splitmix(np.uint64(seed) + np.uint64(7)) ^ splitmix(g))
        n = len(g)
        return cls(keys=keys, owner=np.full(n, -1, np.int64), depth=np.zeros(n, np.int64), device=device,
                   under=np.zeros(n, bool))

    def grow(self, parent_idx: np.ndarray) -> np.ndarray:
        """Register nodes appended since the last call. ``parent_idx`` covers all nodes (0-based,
        -1 = root). Children of one parent are created contiguously (``Expansion.expand``), so a
        child's rank is its offset within its parent's run. Returns the new node indices."""
        N_old, N = len(self.keys), len(parent_idx)
        new = np.arange(N_old, N)
        if len(new) == 0:
            return new
        par = parent_idx[new]
        if (par < 0).any() or (par >= N_old).any():
            raise RuntimeError("new nodes must have already-registered parents")
        start = np.r_[True, par[1:] != par[:-1]]
        run0 = np.maximum.accumulate(np.where(start, np.arange(len(new)), 0))
        rank = np.arange(len(new)) - run0
        par_is_root = parent_idx[par] < 0
        self.keys = np.concatenate([self.keys, child_keys(self.keys[par], rank)])
        self.owner = np.concatenate([self.owner, np.where(par_is_root, rank, self.owner[par])])
        self.depth = np.concatenate([self.depth, self.depth[par] + 1])
        self.under = np.concatenate([self.under, self.under[par]])
        return new

    def select(self, batch: np.ndarray, leaf_idx: np.ndarray):
        """E3: give each graph without a target the basal frontier node at depth ≥ select_dmin
        with the smallest lineage key."""
        if self.select_dmin is None or len(leaf_idx) == 0:
            return
        cand = leaf_idx[(self.owner[leaf_idx] >= 1) & (self.depth[leaf_idx] >= self.select_dmin)]
        for j in np.unique(batch[cand]):
            j = int(j)
            if j in self.targets:
                continue
            c = cand[batch[cand] == j]
            n = int(c[np.argmin(self.keys[c])])
            self.targets[j], self.target_step[j] = n, self.step
            self.under[n] = True

    # masks for editors / ablations (torch, on the model device)
    def mask(self, which: str) -> th.Tensor:
        o, f = self.owner_t, self.frontier_t
        if which == "all":
            return th.ones_like(f)
        if which == "basal":
            return o >= 1
        if which == "basal_frontier":
            return (o >= 1) & f
        if which == "apical":
            return o == 0
        if which == "frontier":
            return f
        if which == "target":
            return f & self.is_target_t
        if which == "target_lineage":
            return f & self.under_t
        raise ValueError(which)


@contextlib.contextmanager
def track(method, model, state: RolloutState, keyed: bool = True):
    """Keep ``state`` in step with ``method.sample_graphs`` and (optionally) key its noise.

    Wraps, for the duration: ``compute_local_bases_for_leaves`` as seen by the expansion module
    (registers new nodes; keys the shared random frame angle of degenerate parents),
    ``method.diffusion.sample`` (publishes masks; keys the C/e priors) and a forward pre-hook
    on ``model`` (publishes the current flow time t = x[0, -1]).
    """
    import graph_generation.method.expansion as E

    orig_bases = E.compute_local_bases_for_leaves
    orig_sample = method.diffusion.sample

    def bases(pos, parent_idx, leaf_parent_idx, uhat, *a, **kw):
        state.grow(parent_idx.detach().cpu().numpy())
        state.step += 1
        if not keyed:
            return orig_bases(pos, parent_idx, leaf_parent_idx, uhat, *a, **kw)
        par = _degenerate_parents(pos, parent_idx, leaf_parent_idx, uhat)
        theta = th.as_tensor(uniforms(state.keys[par], 1, stream=5)[:, 0], dtype=pos.dtype, device=pos.device)
        state.n_degenerate_frames.append(len(par))
        with _serve(th, "rand", [theta] if len(par) else []):
            return orig_bases(pos, parent_idx, leaf_parent_idx, uhat, *a, **kw)

    def sample(*, leaf_idx, batch, **kw):
        N = len(state.keys)
        dev = batch.device
        f = th.zeros(N, dtype=th.bool, device=dev)
        f[leaf_idx] = True
        state.frontier_t, state.batch_t = f, batch
        state.owner_t = th.as_tensor(state.owner, device=dev)
        if state.select_dmin is not None:
            state.select(batch.detach().cpu().numpy(), leaf_idx.detach().cpu().numpy())
            it = th.zeros(N, dtype=th.bool, device=dev)
            if state.targets:
                it[th.as_tensor(list(state.targets.values()), device=dev)] = True
            state.is_target_t, state.under_t = it, th.as_tensor(state.under, device=dev)
        if not keyed:
            return orig_sample(leaf_idx=leaf_idx, batch=batch, **kw)
        z = normals(state.keys[leaf_idx.detach().cpu().numpy()], 4)
        P0 = kw["P_0"]
        draws = [th.as_tensor(z[:, :3], dtype=P0.dtype, device=dev), th.as_tensor(z[:, 3:], dtype=P0.dtype, device=dev)]
        with _serve(th, "randn", draws):
            return orig_sample(leaf_idx=leaf_idx, batch=batch, **kw)

    def pre(m, args, kwargs):
        x = kwargs.get("x", args[0] if args else None)
        state.t = float(x[0, -1])

    E.compute_local_bases_for_leaves = bases
    method.diffusion.sample = sample
    h = model.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield state
    finally:
        E.compute_local_bases_for_leaves = orig_bases
        del method.diffusion.sample
        h.remove()


@contextlib.contextmanager
def _serve(mod, name: str, values: list):
    """Replace ``mod.<name>`` so its next len(values) calls return ``values`` (shapes checked);
    later calls fall through to the original."""
    orig = getattr(mod, name)
    queue = list(values)

    def fake(*size, **kw):
        if not queue:
            return orig(*size, **kw)
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, th.Size)) else tuple(size)
        v = queue.pop(0)
        if tuple(v.shape) != shape:
            raise RuntimeError(f"keyed {name}: expected shape {tuple(v.shape)}, got {shape}")
        return v.clone()

    setattr(mod, name, fake)
    try:
        yield
    finally:
        setattr(mod, name, orig)
        if queue:
            raise RuntimeError(f"keyed {name}: {len(queue)} prepared draw(s) were not consumed")


def _degenerate_parents(pos, parent_idx, leaf_parent_idx, uhat, eps: float = 1e-8) -> np.ndarray:
    """Sorted unique parents whose leaves hit the random shared-frame fallback in
    ``helpers.compute_local_bases_for_leaves`` (same test, same order as ``torch.unique``)."""
    gp = parent_idx[leaf_parent_idx]
    v_in = th.zeros((len(leaf_parent_idx), 3), device=pos.device, dtype=pos.dtype)
    has = gp >= 0
    if has.any():
        s = has.nonzero(as_tuple=False).flatten()
        v_in[s] = pos[leaf_parent_idx[s]] - pos[gp[s]]
    for s in (~has).nonzero(as_tuple=False).flatten().tolist():
        p = int(leaf_parent_idx[s])
        for c in (parent_idx == p).nonzero(as_tuple=False).flatten().tolist():
            if (pos[c] - pos[p]).norm() > eps:
                v_in[s] = pos[c] - pos[p]
                break
    u = uhat.view(1, -1).to(pos)
    perp = v_in - (v_in * u).sum(-1, keepdim=True) * u
    deg = perp.norm(dim=-1) <= eps
    return th.unique(leaf_parent_idx[deg]).cpu().numpy() if deg.any() else np.zeros(0, np.int64)


# ----------------------------------------------------------------------------- editors

class AddDirection:
    """``SiteEditor`` fn adding α·v to the nodes selected by ``state.mask(target)``."""

    def __init__(self, state: RolloutState, v, alpha: float, target: str = "all"):
        self.state, self.alpha, self.target = state, float(alpha), target
        self.v = th.as_tensor(np.asarray(v, dtype=np.float32))

    def __call__(self, feats: th.Tensor):
        if self.alpha == 0.0:
            return None
        m = self.state.mask(self.target).to(feats.device)
        if m.numel() != feats.size(0):
            raise RuntimeError(f"mask over {m.numel()} nodes, features over {feats.size(0)}")
        return feats + self.alpha * m.unsqueeze(-1).to(feats.dtype) * self.v.to(feats)


# ----------------------------------------------------------------------------- generation

def gt_inputs(graphs: list[nx.Graph], cfg, model) -> dict:
    """Per-GT-graph sampler inputs, exactly as ``Trainer._evaluate_rollout`` builds them."""
    from .load import tmd_vector

    out = dict(size=np.array([len(g) for g in graphs]),
               nrc=np.array([g.degree[g.graph["root"]] if "root" in g.graph else 2 for g in graphs]),
               tmd=None, cls=None)
    if getattr(model, "tmd_hidden_dim", 0) > 0:
        out["tmd"] = np.stack([tmd_vector(g, cfg) for g in graphs]).astype(np.float32)
    if getattr(model, "class_hidden_dim", 0) > 0:
        out["cls"] = np.array([int(g.graph["cell_class"]) for g in graphs], dtype=np.int64)
    return out


def generate(method, model, graphs, cfg, seed: int, *, inputs: dict | None = None, batch_items: int = 64,
             keyed: bool = True, interventions=(), device: str = "cpu", tmd_override=None,
             select_dmin: int | None = None) -> list[nx.Graph]:
    """Generate one neuron per GT graph. ``interventions``: callables ``state -> context manager``
    entered around every batch (editors, ablations). Returns graphs in GT order with ``graph``
    attrs gt_index, neuron_id, cell_class, cap_hit (+ target_key / target_step with
    ``select_dmin``) and node attrs ``owner`` and ``key`` (lineage key)."""
    inputs = inputs or gt_inputs(graphs, cfg, model)
    if tmd_override is not None:
        inputs = {**inputs, "tmd": np.asarray(tmd_override, np.float32)}
    psf = getattr(cfg.dataset, "pos_scale_factor", None)
    order = np.argsort(inputs["size"], kind="stable")
    out: list[nx.Graph | None] = [None] * len(graphs)
    for b0 in range(0, len(order), batch_items):
        idx = order[b0:b0 + batch_items]
        th.manual_seed(int(seed) * 100_003 + b0)   # unkeyed path: reproducible per batch
        state = RolloutState.new(seed, idx, device=device)
        state.select_dmin = select_dmin
        kw = dict(target_size=th.as_tensor(inputs["size"][idx], device=device),
                  num_root_children=th.as_tensor(inputs["nrc"][idx], device=device))
        if inputs["tmd"] is not None:
            kw["tmd"] = th.as_tensor(inputs["tmd"][idx], device=device)
        if inputs["cls"] is not None:
            kw["cell_class"] = th.as_tensor(inputs["cls"][idx], device=device)
        with contextlib.ExitStack() as es, th.no_grad():
            es.enter_context(track(method, model, state, keyed=keyed))
            for make in interventions:
                es.enter_context(make(state))
            preds = method.sample_graphs(model=model, **kw)
        batch = state.batch_t.cpu().numpy() if state.batch_t is not None else np.arange(len(idx))
        for j, (gi, G) in enumerate(zip(idx, preds)):
            nodes = np.flatnonzero(batch == j) if state.batch_t is not None else np.array([j])
            own, keys = state.owner[nodes], state.keys[nodes]
            if len(own) != G.number_of_nodes():
                raise RuntimeError("node bookkeeping out of step with sample_graphs")
            for n in G.nodes:
                G.nodes[n]["owner"] = int(own[n])
                G.nodes[n]["key"] = int(keys[n])
                if psf is not None:
                    G.nodes[n]["pos"] = G.nodes[n]["pos"] * float(psf)
            if select_dmin is not None:
                tg = state.targets.get(j)
                G.graph.update(target_key=int(state.keys[tg]) if tg is not None else None,
                               target_step=state.target_step.get(j))
            G.graph.update(root=0, gt_index=int(gi), neuron_id=graphs[gi].graph.get("neuron_id"),
                           cell_class=graphs[gi].graph.get("cell_class"),
                           cap_hit=bool(G.number_of_nodes() >= (method.max_tree_size or 10 ** 9)))
            out[gi] = G
    return out


# ----------------------------------------------------------------------------- conditions

def make_editor(model, site: str, fn):
    return lambda state: H.SiteEditor(model, site, fn(state))


def make_ablation(model, mode: str, blocks=None, means=None, target: str | None = None):
    def mk(state):
        return H.AttnAblation(model, blocks=blocks, mode=mode, means=means, t_fn=lambda: state.t,
                              node_mask=(lambda: state.mask(target)) if target else None)
    return mk


GROWTH_DMIN = 5   # E3: target = basal frontier node at generated depth >= 5


def load_dirs(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def conditions(exp: str, model, args, dirs: dict | None, means: dict | None) -> list[dict]:
    """Condition grid for one experiment. Each: dict(name, interventions=[...], spec=...)."""
    G = H.global_block_ids(model)
    C = []
    if exp == "e0":
        for mode in ("mean", "graphmean", "zero"):
            C.append(dict(name=f"abl_{mode}", spec=dict(ablate=mode, blocks=G),
                          interventions=[make_ablation(model, mode, G, means)]))
        for blk in ([1, 3], [5, 7], [9, 11]):
            C.append(dict(name=f"abl_mean_b{''.join(map(str, blk))}", spec=dict(ablate="mean", blocks=blk),
                          interventions=[make_ablation(model, "mean", blk, means)]))
        return C
    sites = args.sites.split(",")
    alphas = [float(a) for a in args.alphas.split(",")]
    kinds = args.kinds.split(",")
    names = args.dir_names.split(",") if args.dir_names else None
    targets = args.masks.split(",") if exp == "growth" else ["all"]
    for s in sites:
        avail = sorted({k.split("/")[1] for k in dirs if k.startswith(f"{s}/") and k.endswith("/v")})
        for dn in (names or avail):
            for kind in kinds:
                if f"{s}/{dn}/{kind}" not in dirs:     # controls exist for the main direction only
                    continue
                v = dirs[f"{s}/{dn}/{kind}"]
                for a in alphas:
                    for tg in targets:
                        if a == 0:
                            continue
                        C.append(dict(name=f"{s}_{dn}_{kind}_a{a:g}_{tg}",
                                      spec=dict(site=s, dir=dn, kind=kind, alpha=a, target=tg,
                                                norm=float(np.linalg.norm(v)),
                                                sigma=float(dirs.get(f"{s}/sigma", np.nan))),
                                      interventions=[make_editor(model, s, lambda st, v=v, a=a, tg=tg:
                                                                 AddDirection(st, v, a, tg))]))
    return C


def _save(path: Path, graphs, meta):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"pred_graphs": graphs, "meta": meta}, f)
    tmp.replace(path)


def main(argv=None):
    from .load import build_method_and_model, compose_cfg, load_split

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--which", default="ema", help="ema | raw | random")
    ap.add_argument("--exp", required=True, choices=["intact", "e0", "class", "growth"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-neurons", type=int, default=256, help="Phase A subset size (random, fixed seed 0)")
    ap.add_argument("--subset", default="phaseA", choices=["phaseA", "rest", "all"],
                    help="phaseA = the --max-neurons subset; rest = every other neuron of the split (Phase B)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--batch-items", type=int, default=64)
    ap.add_argument("--no-keyed", action="store_true", help="plain sampler noise (no lineage keys)")
    ap.add_argument("--dirs", default=None, help="directions npz from interp.analyses.steer_dirs")
    ap.add_argument("--sites", default="L5")
    ap.add_argument("--alphas", default="1")
    ap.add_argument("--kinds", default="v", help="v,neg is expressed by negative alphas; controls: random,onmanifold")
    ap.add_argument("--dir-names", default=None, help="subset of direction names (default: all in the npz)")
    ap.add_argument("--masks", default="target_lineage", help="growth: node sets to edit (target_lineage,target)")
    ap.add_argument("--growth-dmin", type=int, default=GROWTH_DMIN, help="growth: min generated depth of the target")
    ap.add_argument("--tmd-class", type=int, default=None,
                    help="TMD arm yardstick: condition on TMDs of real neurons of this class (cycled)")
    ap.add_argument("--max-tree-size", type=int, default=None, help="override method.max_tree_size (tests)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    cfg = compose_cfg(a.config)
    graphs = load_split(cfg, a.split, data_dir=a.data_dir)
    method, model, meta = build_method_and_model(cfg, a.ckpt, which=a.which, pool=graphs[:2])
    model.to(a.device)
    if a.max_tree_size:
        method.max_tree_size = a.max_tree_size
    if a.subset != "all" and a.max_neurons and len(graphs) > a.max_neurons:
        phase_a = np.sort(np.random.default_rng(0).choice(len(graphs), a.max_neurons, replace=False))
        keep = phase_a if a.subset == "phaseA" else np.setdiff1d(np.arange(len(graphs)), phase_a)
        graphs = [graphs[i] for i in keep]
    inputs = gt_inputs(graphs, cfg, model)
    tmd_override, tag = None, ""
    if a.tmd_class is not None:
        if inputs["tmd"] is None:
            raise SystemExit("--tmd-class needs the TMD arm")
        pool = [g for g in load_split(cfg, "train", data_dir=a.data_dir) if g.graph.get("cell_class") == a.tmd_class]
        donors = np.random.default_rng(1).choice(len(pool), len(graphs), replace=len(pool) < len(graphs))
        from .load import tmd_vector
        tmd_override = np.stack([tmd_vector(pool[d], cfg) for d in donors])
        tag = f"_tmdclass{a.tmd_class}"

    out = Path(a.out) / (a.exp + tag + ("" if a.subset == "phaseA" else f"_{a.subset}"))
    out.mkdir(parents=True, exist_ok=True)
    dirs = load_dirs(a.dirs) if a.dirs else None
    seeds = [int(s) for s in a.seeds.split(",")]
    run_meta = dict(meta, config=a.config, split=a.split, subset=a.subset, growth_dmin=a.growth_dmin, n=len(graphs), keyed=not a.no_keyed,
                    batch_items=a.batch_items, neuron_ids=[g.graph["neuron_id"] for g in graphs], dirs=a.dirs)
    gen = dict(inputs=inputs, batch_items=a.batch_items, keyed=not a.no_keyed, device=a.device,
               tmd_override=tmd_override, select_dmin=a.growth_dmin if a.exp == "growth" else None)

    # intact first; for e0 it also records the per-(block, t) mean global read
    means = None
    need_means = a.exp == "e0"
    mpath = out / "attn2_means.pt"
    for s in seeds:
        p = out / f"intact_seed{s}.pkl"
        if p.exists() and (not need_means or mpath.exists()):
            continue
        rec = H.AttnAblation(model, mode="record")
        mk = [lambda st, rec=rec: _rec_ctx(rec, st)] if need_means and s == seeds[0] else []
        t0 = time.time()
        G = generate(method, model, graphs, cfg, s, interventions=mk, **gen)
        _save(p, G, dict(run_meta, condition="intact", seed=s, spec={}, seconds=time.time() - t0))
        if mk:
            th.save({k: v.cpu() for k, v in rec.recorded_means().items()}, mpath)
        print(f"[rollout] intact seed {s}: {time.time() - t0:.0f}s", flush=True)
    if need_means:
        means = {k: v.to(a.device) for k, v in th.load(mpath, weights_only=False).items()}  # (block, tkey)
        for i in H.global_block_ids(model):   # per-block fallback = mean over t keys
            ks = [k for k in means if isinstance(k, tuple) and k[0] == i]
            means.setdefault(i, th.stack([means[k] for k in ks]).mean(0))
    if a.exp == "intact":
        return
    conds = conditions(a.exp, model, a, dirs, means)
    (out / "conditions.json").write_text(json.dumps([dict(name=c["name"], spec=c["spec"]) for c in conds], indent=1))
    for c in conds:
        for s in seeds:
            p = out / f"{c['name']}_seed{s}.pkl"
            if p.exists():
                continue
            t0 = time.time()
            G = generate(method, model, graphs, cfg, s, interventions=c["interventions"], **gen)
            _save(p, G, dict(run_meta, condition=c["name"], seed=s, spec=c["spec"], seconds=time.time() - t0))
            print(f"[rollout] {c['name']} seed {s}: {time.time() - t0:.0f}s", flush=True)


@contextlib.contextmanager
def _rec_ctx(rec: H.AttnAblation, state: RolloutState):
    rec.t_fn = lambda: state.t
    with rec:
        yield


if __name__ == "__main__":
    main()
