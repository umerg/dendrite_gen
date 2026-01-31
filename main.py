import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import hydra
import networkx as nx
import numpy as np
import torch as th
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from utils.data_loading import nx_graph_to_adj_pos, load_swc_graphs_from_dir
from utils.tmd_paper_embedding_utils import compute_tmd_global_embedding_paper

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
    print(f"Extracting adjacency and position matrices for {len(train_graphs)} training graphs...")
    adjs = []
    poses = []
    tmds = []
    for G in train_graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        adjs.append(A)
        poses.append(P)
        tmds.append(compute_tmd_global_embedding_paper(G))
    print("Extraction done.")

    print("Creating training reduction sequences...")
    train_dataset = gg.data.InfiniteRandRedDataset(
        adjs=adjs,
        poses=poses,
        tmds=tmds,
        red_factory=red_factory,
    ) # support only for infinite random reduction dataset for expansion
    print("Training reduction sequences created.")

    # Dataloader
    is_mp = cfg.reduction.num_red_seqs < 0  # if infinite dataset
    num_workers = min(mp.cpu_count(), cfg.training.max_num_workers) * is_mp
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True, 
        collate_fn=Batch.from_data_list,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    # check if we augment the graph with extra edges
    edge_embedding_nums = [2]
    edge_embedding_dims = [4]
    edge_attr_dim = 1  # initial edge attribute dimension - label category
    if cfg.method.name == "expansion_augmented":
        edge_embedding_nums = [3]
        edge_embedding_dims = [4]
        print(f"Using augmented expansion with edge embeddings: nums {edge_embedding_nums}, dims {edge_embedding_dims}")

    # Model
    print(f"Initializing model: {cfg.model.name}...")
    # features = 2 if cfg.diffusion.name == "discrete" else 1
    tmd_in_dim = getattr(cfg.model, "tmd_in_dim", 0)
    tmd_hidden_dim = getattr(cfg.model, "tmd_hidden_dim", 0)
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
            # so2_axis=cfg.model.so2_axis,
        )
    elif cfg.model.name == "egnn_simple":
        model = gg.model.SO2_EGNN_Sparse_Network_Simple(
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
            # so2_axis=cfg.model.so2_axis,
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

    # Method
    method = None
    method_name = cfg.method.name
    if diffusion is not None:
        if method_name != "expansion":
            raise ValueError(
                f"Diffusion-based runs require method 'expansion', got '{method_name}'."
            )
        expansion_loss_weight = getattr(cfg.method, "expansion_loss_weight", 1.0)
        method = gg.method.Expansion(
            diffusion=diffusion,
            deterministic_expansion=cfg.method.deterministic_expansion,
            red_threshold=cfg.reduction.red_threshold,
            expansion_loss_weight=expansion_loss_weight,
        )
    elif method_name == "expansion":
        method = gg.method.Expansion_OneShot(
            deterministic_expansion=cfg.method.deterministic_expansion,
            leaf_noise_sigma=cfg.method.leaf_noise_sigma,
            leaf_noise_clip=cfg.method.leaf_noise_clip,
            sibling_loss_weight=cfg.method.sibling_loss_weight,
            use_sibling_matching=cfg.method.use_sibling_matching,
            use_geo_lr_mask=cfg.method.use_geo_lr_mask,
            use_radial_distance=cfg.method.use_radial_distance,
            debug=cfg.debugging,
            debug_max_batches=cfg.debugging_max_batches,
            debug_dir=cfg.debugging_dir,
        )  # expansion with one-shot generation at every step
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    return {
        "train_dataloader": train_dataloader,
        "method": method,
        "model": model,
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

    # keep only largest connected component for train graphs - IS THIS REDUNDANT? TODO
    train_graphs = [
        G.subgraph(max(nx.connected_components(G), key=len)) for G in train_graphs
    ]
    _ensure_root(train_graphs)

    # Metrics
    validation_metrics = [
        gg.metrics.NodeNumDiff(),
        gg.metrics.NodeDegree(),
        gg.metrics.ClusteringCoefficient(),
        gg.metrics.OrbitCount(),
        gg.metrics.Spectral(),
        gg.metrics.Wavelet(),
        gg.metrics.Ratio(),
        gg.metrics.Uniqueness(),
        gg.metrics.Novelty(),
    ]

    if "tree" in cfg.dataset.name: # retaining tree-specific metrics
        validation_metrics += [
            gg.metrics.ValidTree(),
            gg.metrics.UniqueNovelValidTree(),
        ]

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
        else:
            raise ValueError(f"Unknown diffusion name: {diffusion_name}")

    # Method
    if cfg.method.name in ("expansion", "expansion_augmented", "expansion_0ed"):
        method_items = get_expansion_items(cfg, train_graphs, diffusion=diffusion)
    else:
        raise ValueError(f"Unknown method name: {cfg.method.name}")
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
        metrics=validation_metrics,
        cfg=cfg,
    )
    if cfg.testing:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
