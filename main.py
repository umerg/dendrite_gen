import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch as th
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from utils.data_loading import nx_graph_to_adj_pos, load_swc_graphs_from_dir
from utils.tmd import compute_tmd_mixed, tmd_conditioning_dim

import graph_generation as gg


def get_expansion_items(cfg: DictConfig, train_graphs, diffusion=None):

    # Train Dataset
    reduction_type = getattr(cfg.reduction, "type", "cherry")
    if reduction_type == "depth":
        factory_cls = gg.depth_reduction.DepthReductionFactory
    elif reduction_type == "cherry":
        factory_cls = gg.reduction.ReductionFactory
    else:
        raise ValueError(f"Unknown reduction type '{reduction_type}'. Expected 'cherry' or 'depth'.")

    factory_kwargs = dict(
        mode=cfg.reduction.mode,
        cherry_p=cfg.reduction.cherry_p,
        ensure_progress=cfg.reduction.ensure_progress,
        root=cfg.reduction.root,
        contract_root=cfg.reduction.contract_root,
    )
    if hasattr(cfg.reduction, "weighted_reduction"):
        factory_kwargs["weighted_reduction"] = cfg.reduction.weighted_reduction

    red_factory = factory_cls(**factory_kwargs) # initialised cherry/depth reduction factory
    pos_scale_factor = getattr(cfg.dataset, "pos_scale_factor", None)
    # TMD conditioning knobs (single source of truth; tmd_in_dim is derived at model init).
    tmd_hidden_dim = getattr(cfg.model, "tmd_hidden_dim", 0)
    tmd_filtrations = list(getattr(cfg.model, "tmd_filtrations", ("path", "height", "rho")))
    tmd_bins = int(getattr(cfg.model, "tmd_bins", 16))
    uhat = np.asarray(getattr(cfg.model, "so2_axis", (0.0, 0.0, 1.0)), dtype=float).reshape(3)
    # RBF edge-distance kernel knobs (default OFF; ranges are in pos_scale_factor-normalized units).
    rbf_k = int(getattr(cfg.model, "rbf_k", 0))
    rbf_gamma = float(getattr(cfg.model, "rbf_gamma", 10.0))
    rbf_rho_max = float(getattr(cfg.model, "rbf_rho_max", 5.0))
    rbf_du_max = float(getattr(cfg.model, "rbf_du_max", 3.0))
    print(f"Extracting adjacency and position matrices for {len(train_graphs)} training graphs...")
    adjs = []
    poses = []
    tmds = []
    for G in train_graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        if pos_scale_factor is not None:
            P = P / float(pos_scale_factor)
        adjs.append(A)
        poses.append(P)
        tmds.append(compute_tmd_mixed(G, filtrations=tmd_filtrations, n_bins=tmd_bins, uhat=uhat))
    if pos_scale_factor is not None:
        print(f"Positions scaled by 1/{pos_scale_factor} (offsets now ~unit scale).")
    print("Extraction done.")

    print("Creating training reduction sequences...")
    # When depth reduction is deterministic, precompute all sequences once
    if reduction_type == "depth":
        train_dataset = gg.data.PrecomputedRedDataset(
            adjs=adjs, poses=poses, tmds=tmds, red_factory=red_factory,
        )
        num_workers = 0  # data is precomputed, no worker computation needed
    else:
        train_dataset = gg.data.InfiniteRandRedDataset(
            adjs=adjs, poses=poses, tmds=tmds, red_factory=red_factory,
        )
        # InfiniteRandRedDataset is always infinite -> enable dataloader workers.
        num_workers = min(mp.cpu_count(), cfg.training.max_num_workers)
    print("Training reduction sequences created.")

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=Batch.from_data_list,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    # Edge attribute carries a single label category (initial expansion label).
    edge_embedding_nums = [2]
    edge_embedding_dims = [4]
    edge_attr_dim = 1

    # Model
    print(f"Initializing model: {cfg.model.name}...")
    # features = 2 if cfg.diffusion.name == "discrete" else 1
    # tmd_in_dim is DERIVED from the conditioning filtration set x bins (never hand-set),
    # so the embedding width and the model's input projection can never desync.
    tmd_in_dim = tmd_conditioning_dim(tmd_filtrations, tmd_bins) if tmd_hidden_dim > 0 else 0
    _stale_tmd_in_dim = getattr(cfg.model, "tmd_in_dim", None)
    if tmd_hidden_dim > 0 and _stale_tmd_in_dim not in (None, 0) and int(_stale_tmd_in_dim) != tmd_in_dim:
        raise ValueError(
            f"cfg.model.tmd_in_dim={_stale_tmd_in_dim} disagrees with the derived conditioning "
            f"dim {tmd_in_dim} (= {len(tmd_filtrations)} filtrations x {tmd_bins}^2). "
            "tmd_in_dim is derived from tmd_filtrations/tmd_bins — remove it from the config "
            "or fix tmd_filtrations/tmd_bins."
        )
    if tmd_hidden_dim > 0:
        print(f"TMD conditioning ON: filtrations={list(tmd_filtrations)}, bins={tmd_bins} -> tmd_in_dim={tmd_in_dim}")
    if rbf_k > 0:
        print(f"RBF edge features ON: k={rbf_k}, gamma={rbf_gamma}, rho in [0,{rbf_rho_max}], du in [+-{rbf_du_max}]")
    if cfg.model.name == "egnn":
        model = gg.model.SO2_EGNN_Network(
            n_layers=cfg.model.num_layers,
            feats_dim=cfg.model.feats_dim,
            pos_dim=3,
            m_dim=cfg.model.m_dim,
            edge_embedding_nums=edge_embedding_nums,
            edge_embedding_dims=edge_embedding_dims,
            edge_attr_dim=edge_attr_dim,
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
            rbf_k=rbf_k,
            rbf_gamma=rbf_gamma,
            rbf_rho_max=rbf_rho_max,
            rbf_du_max=rbf_du_max,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    print("Model initialized.")

    # Diffusion - Currently one shot
    # if cfg.diffusion.name == "discrete":
    #     diffusion = gg.diffusion.sparse.DiscreteGraphDiffusion(
    #         self_conditioning=cfg.diffusion.self_conditioning,
    #         num_steps=cfg.diffusion.num_steps,
    #     )
    # elif cfg.diffusion.name == "edm":
    #     diffusion = gg.diffusion.sparse.EDM(
    #         self_conditioning=cfg.diffusion.self_conditioning,
    #         num_steps=cfg.diffusion.num_steps,
    #     )
    # else:
    #     raise ValueError(f"Unknown diffusion name: {cfg.diffusion.name}")

    # Method — the live path is the diffusion-wrapped Expansion.
    method_name = cfg.method.name
    if method_name != "expansion":
        raise ValueError(f"Unknown method name: {method_name}")
    if diffusion is None:
        raise ValueError("Expansion requires a diffusion block (basic | edm | flow | flow_v).")
    expansion_loss_weight = getattr(cfg.method, "expansion_loss_weight", 1.0)
    use_size_ratio = getattr(cfg.method, "use_size_ratio", True)
    method = gg.method.Expansion(
        diffusion=diffusion,
        expansion_loss_weight=expansion_loss_weight,
        use_size_ratio=use_size_ratio,
        max_tree_size=getattr(cfg.method, "max_tree_size", 500),
    )

    return {
        "train_dataloader": train_dataloader,
        "method": method,
        "model": model,
        "pos_scale_factor": pos_scale_factor,
    }


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debugging:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Fix random seeds
    random.seed(0)
    np.random.seed(0)
    th.manual_seed(0)

    # Graphs
    if cfg.dataset.load:  # load from SWC directory structure: data_dir/{train,val,test}
        print("Loading dataset from SWC files...")
        data_root = Path(cfg.dataset.data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_root}")
        train_graphs = load_swc_graphs_from_dir(data_root / "train")
        validation_graphs = load_swc_graphs_from_dir(data_root / "val")
        test_graphs = load_swc_graphs_from_dir(data_root / "test")
        print(f"Loaded {len(train_graphs)} train graphs, "
              f"{len(validation_graphs)} validation graphs, "
              f"{len(test_graphs)} test graphs.")

    elif cfg.dataset.name in ["tree_synthetic"]: # retaining synthetic graphs dataset if required for testing
        graph_generator = (
            gg.data.generate_tree_graphs
        )

        train_graphs = graph_generator(
            num_graphs=cfg.dataset.train_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=0,
        )
        # validation_graphs = graph_generator(
        #     num_graphs=cfg.dataset.val_size,
        #     min_size=cfg.dataset.min_size,
        #     max_size=cfg.dataset.max_size,
        #     seed=1,
        # )
        validation_graphs = train_graphs[:8]  # TODO remove

        test_graphs = graph_generator(
            num_graphs=cfg.dataset.test_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=2,
        )
    elif cfg.dataset.name == "deterministic_synth":  # zero-conditional-entropy probe dataset
        train_graphs = gg.data.generate_deterministic_trees(
            num_graphs=cfg.dataset.train_size, seed=0,
        )
        validation_graphs = gg.data.generate_deterministic_trees(
            num_graphs=cfg.dataset.val_size, seed=1,
        )
        test_graphs = gg.data.generate_deterministic_trees(
            num_graphs=cfg.dataset.test_size, seed=2,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

    def _ensure_root(graphs):
        for G in graphs:
            root = G.graph.get("root", None)
            if root is None or root not in G.nodes:
                G.graph["root"] = next(iter(G.nodes))

    _ensure_root(train_graphs)
    _ensure_root(validation_graphs)
    _ensure_root(test_graphs)

    diffusion_cfg = OmegaConf.select(cfg, "diffusion")
    diffusion = None
    if diffusion_cfg is not None:
        diffusion_name = getattr(diffusion_cfg, "name", None)
        if diffusion_name == "basic":
            diffusion = gg.diffusion.DenoisingDiffusionModel(
                num_steps=diffusion_cfg.num_steps,
            )
        elif diffusion_name == "edm":
            diffusion = gg.diffusion.EDMDiffusionModel(
                num_steps=diffusion_cfg.num_steps,
            )
        elif diffusion_name in ("flow", "flow_v"):
            _prior_std_pos = getattr(diffusion_cfg, "prior_std_pos", None)
            _flow_cls = (
                gg.diffusion.VFlowMatchingModel if diffusion_name == "flow_v"
                else gg.diffusion.FlowMatchingModel
            )
            diffusion = _flow_cls(
                num_steps=diffusion_cfg.num_steps,
                prior_std=getattr(diffusion_cfg, "prior_std", 1.0),
                time_dist=getattr(diffusion_cfg, "time_dist", "uniform"),
                beta_a=getattr(diffusion_cfg, "beta_a", 2.0),
                beta_b=getattr(diffusion_cfg, "beta_b", 1.0),
                sigma_min=getattr(diffusion_cfg, "sigma_min", 0.0),
                prior_std_pos=(list(_prior_std_pos) if _prior_std_pos is not None else None),
            )
        else:
            raise ValueError(f"Unknown diffusion name: {diffusion_name}")

    # Method
    if cfg.method.name != "expansion":
        raise ValueError(f"Unknown method name: {cfg.method.name}")
    method_items = get_expansion_items(cfg, train_graphs, diffusion=diffusion)
    method_items = defaultdict(lambda: None, method_items)
    # verbose logging of method details
    print(f"Using method: {cfg.method.name}")
    print(f"Method details: {method_items['method']}")
    print(f"Model details: {method_items['model']}")

    # Trainer
    th.set_float32_matmul_precision("high")
    trainer = gg.training.Trainer(
        model=method_items["model"],
        method=method_items["method"],
        train_dataloader=method_items["train_dataloader"],
        train_graphs=train_graphs,
        validation_graphs=validation_graphs,
        test_graphs=test_graphs,
        cfg=cfg,
        pos_scale_factor=method_items.get("pos_scale_factor"),
    )
    if cfg.testing:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
