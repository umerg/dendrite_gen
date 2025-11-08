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

from utils.proprocessing import nx_graph_to_adj_pos

import graph_generation as gg


def get_expansion_items(cfg: DictConfig, train_graphs):

    # Train Dataset
    red_factory = gg.reduction.ReductionFactory(
        mode=cfg.reduction.mode,
        cherry_p=cfg.reduction.cherry_p,
        ensure_progress=cfg.reduction.ensure_progress,
        root=cfg.reduction.root,
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=Batch.from_data_list,
        num_workers=min(mp.cpu_count(), cfg.training.max_num_workers) * is_mp,
        multiprocessing_context="spawn" if is_mp else None,
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
    if cfg.dataset.load: # needs to change TODO
        with open(Path("./data") / f"{cfg.dataset.name}.pkl", "rb") as f:
            dataset = pickle.load(f)

        train_graphs = dataset["train"]
        validation_graphs = dataset["val"]
        test_graphs = dataset["test"]

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
        validation_graphs = graph_generator(
            num_graphs=cfg.dataset.val_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=1,
        )
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
