"""On-disk activation store.

Layout of one extraction ``<root>/t<t>/<split>/``::

    meta.json                 provenance (ckpt, step, which, seed, t, sites, git sha)
    index.parquet             one row per stored node row (neuron, level, node, tier, ...)
    graph_index.parquet       one row per reduction item (neuron, level, n_levels, ...)
    acts/<site>.npy           [n_rows_site, feats] float16 memmap (dense sites: tier A+B1;
                              sparse sites: tier A+B1+B2 — see ``index.parquet.row_<group>``)
    graph_acts/<G site>.npy   [n_items, m_tokens, feats] float16
    graph_acts/pool_<site>.npy  [n_items, feats] mean over all nodes of the item
    targets_final.parquet     per (neuron, node) final-tree targets (interp.targets)

Two row groups: ``dense`` (frontier + root-child rows, every site) and ``sparse``
(dense rows + a fixed random subsample of other context rows, a reduced site list).
``index.parquet`` has ``row_dense`` / ``row_sparse`` columns (-1 = not stored there).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def split_dir(root: str | Path, t: float, split: str) -> Path:
    return Path(root) / f"t{t:g}" / split


@dataclass
class Store:
    path: Path
    meta: dict
    index: pd.DataFrame
    graph_index: pd.DataFrame
    _gcache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def light(cls, path: str | Path) -> "Store":
        """Meta only (no index tables) — enough for ``acts`` / ``graph_acts`` given row ids.
        Used by parallel workers, which receive the row ids from the parent."""
        p = Path(path)
        return cls(path=p, meta=json.loads((p / "meta.json").read_text()), index=None, graph_index=None)

    @classmethod
    def open(cls, root: str | Path, t: float | None = None, split: str | None = None) -> "Store":
        p = Path(root) if t is None else split_dir(root, t, split)
        return cls(
            path=p,
            meta=json.loads((p / "meta.json").read_text()),
            index=pd.read_parquet(p / "index.parquet"),
            graph_index=pd.read_parquet(p / "graph_index.parquet"),
        )

    def group_of(self, site: str) -> str:
        return "dense" if site in self.meta["sites_dense"] else "sparse"

    def acts(self, site: str, rows: pd.DataFrame | None = None, dtype=np.float32) -> np.ndarray:
        """Activations of ``site`` for ``rows`` (a subset of ``index``; default all rows stored there)."""
        n_bad = self.meta.get("overflow", {}).get(site, 0)
        if n_bad:
            raise ValueError(f"Site {site} has {n_bad} non-finite values ({self.meta.get('dtype')} overflow "
                             "at extraction); re-extract with --dtype float32.")
        mm = np.load(self.path / "acts" / f"{site}.npy", mmap_mode="r")
        col = f"row_{self.group_of(site)}"
        if rows is None:
            rows = self.index[self.index[col] >= 0]
        r = rows[col].to_numpy()
        if (r < 0).any():
            raise ValueError(f"{(r < 0).sum()} requested rows are not stored at site {site}.")
        return np.asarray(mm[r], dtype=dtype)

    def graph_acts(self, name: str, dtype=np.float32) -> np.ndarray:
        """``G{i}`` token states [items, m, F] or ``pool_<site>`` mean-pooled node feats [items, F]."""
        key = name[len("pool_"):] if name.startswith("pool_") else name
        if self.meta.get("overflow", {}).get(key, 0):
            raise ValueError(f"{name}: {self.meta.get('dtype')} overflow at extraction; re-extract with --dtype float32.")
        key = (name, np.dtype(dtype).str)
        if key not in self._gcache:  # read each graph-level array from disk once per Store
            self._gcache[key] = np.load(self.path / "graph_acts" / f"{name}.npy", mmap_mode="r").astype(dtype)
        return self._gcache[key]

    def neuron_tmd(self) -> np.ndarray:
        return np.load(self.path / "neuron_tmd.npy")

    def neuron_meta(self) -> pd.DataFrame:
        return pd.read_parquet(self.path / "neuron_meta.parquet")

    def targets_final(self) -> pd.DataFrame:
        return pd.read_parquet(self.path / "targets_final.parquet")
