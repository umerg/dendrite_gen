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
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from utils.data_loading import nx_graph_to_adj_pos, load_swc_graphs_from_dir

import graph_generation as gg


def get_expansion_items(cfg: DictConfig, train_graphs):

    # Train Dataset
    red_factory = gg.reduction.ReductionFactory(
        mode=cfg.reduction.mode,
        cherry_p=cfg.reduction.cherry_p,
        ensure_progress=cfg.reduction.ensure_progress,
        root=cfg.reduction.root,
        contract_root=cfg.reduction.contract_root,
    ) # initialised cherry reduction factory

    adjs = []
    poses = []
    for G in train_graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        adjs.append(A)
        poses.append(P)

    train_dataset = gg.data.InfiniteRandRedDataset(
        adjs=adjs,
        poses=poses,
        red_factory=red_factory,
    ) # support only for infinite random reduction dataset for expansion

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

    # Model

    # features = 2 if cfg.diffusion.name == "discrete" else 1
    if cfg.model.name == "egnn": # PARAMS NOT DECIDED TODO
        model = gg.model.SO2_EGNN_Sparse_Network(
            n_layers=cfg.model.num_layers,
            feats_dim=cfg.model.feats_dim,
            pos_dim=3,
            m_dim=cfg.model.m_dim,
            dropout=cfg.model.dropout,
            norm_feats=cfg.model.norm_feats,
            global_linear_attn_every=cfg.model.global_linear_attn_every,
            global_linear_attn_heads=cfg.model.global_linear_attn_heads,
            global_linear_attn_dim_head=cfg.model.global_linear_attn_dim_head,
            num_global_tokens=cfg.model.num_global_tokens,
            offset_head_hidden=cfg.model.offset_head_hidden,
            # so2_axis=cfg.model.so2_axis,
            use_global_fallback_frames=cfg.model.use_global_fallback_frames,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

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
    method = gg.method.Expansion_OneShot(
        deterministic_expansion=cfg.method.deterministic_expansion,
        red_threshold=cfg.reduction.red_threshold,
        leaf_noise_sigma=cfg.method.leaf_noise_sigma,
        leaf_noise_clip=cfg.method.leaf_noise_clip,
        sibling_loss_weight=cfg.method.sibling_loss_weight,
        debug=cfg.debugging,
        debug_max_batches=cfg.debugging_max_batches,
        debug_dir=cfg.debugging_dir,
    ) # expansion with one-shot generation at every step

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
        data_root = Path(cfg.dataset.data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_root}")
        train_graphs = load_swc_graphs_from_dir(data_root / "train")
        validation_graphs = load_swc_graphs_from_dir(data_root / "val")
        test_graphs = load_swc_graphs_from_dir(data_root / "test")

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

    # keep only largest connected component for train graphs - IS THIS REDUNDANT? TODO
    train_graphs = [
        G.subgraph(max(nx.connected_components(G), key=len)) for G in train_graphs
    ]

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

    # Method
    if cfg.method.name == "expansion":
        method_items = get_expansion_items(cfg, train_graphs)
    else:
        raise ValueError(f"Unknown method name: {cfg.method.name}")
    method_items = defaultdict(lambda: None, method_items)

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
