# data.py
import numpy as np
import scipy as sp
import torch as th
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor

class ReducedGraphData(Data):
    """
    Minimal payload for training the expansion model.
    Required fields:
      - adj:               fine adjacency (SparseTensor or torch/scipy -> SparseTensor)
      - adj_reduced:       coarse adjacency (next level)
      - expansion_matrix:  fine->coarse membership (n x m, binary)
      - node_expansion:    per coarse node cluster sizes (length m)
      - reduction_level:   current level (int)
      - target_size:       n (fine node count), for conditioning / bookkeeping
    """
    def __init__(self, **kwargs):
        super().__init__()
        if not kwargs:
            return

        # Minimal x to satisfy PyG (not used by model)
        n = kwargs["adj"].shape[0] if hasattr(kwargs["adj"], "shape") else None
        if n is None and isinstance(kwargs["adj"], SparseTensor):
            n = kwargs["adj"].size(0)
        if n is None:
            raise ValueError("adj must have a known node dimension")
        super().__init__(x=th.zeros(n))

        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, (int, np.integer)):
                value = th.tensor(int(value), dtype=th.long)
            elif isinstance(value, np.ndarray):
                value = th.from_numpy(value).to(th.float32 if value.dtype.kind == "f" else th.long)
            elif sp.sparse.issparse(value) or isinstance(value, sp.sparse.sparray):
                value = SparseTensor.from_scipy(value).to(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            elif isinstance(value, th.Tensor) or isinstance(value, SparseTensor):
                pass
            else:
                raise ValueError(f"Unsupported type {type(value)} for key {key}")
            setattr(self, key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Keep block-diagonal concatenation for sparse tensors
        if isinstance(value, SparseTensor):
            return (0, 1)
        return super().__cat_dim__(key, value, *args, **kwargs)
