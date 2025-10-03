# data.py
import numpy as np
import scipy as sp
import torch as th
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor

class ReducedGraphData(Data):
    """
    Required fields:
      - adj:               adjacency of the current reduced graph (SparseTensor)
      - pos:               node positions (FloatTensor shape [N,3])
      - leaf_idx:          indices of leaf nodes in this graph (LongTensor shape [L])
      - leaf_mask:         boolean mask for leaf nodes (BoolTensor shape [N])
      - leaf_expansion:    labels for those leaves (LongTensor in {1,2}, shape [L])
      - reduction_level:   current level (int)
      - target_size:       n (node count of this graph), for bookkeeping
    """
    def __init__(self, **kwargs):
        super().__init__()
        if not kwargs:
            return

        # Use position matrix directly as x, EGNN expects positions as node features
        pos = kwargs.get("pos", None)
        if isinstance(pos, np.ndarray):
            x = th.from_numpy(pos).to(th.float32)
        elif isinstance(pos, th.Tensor):
            x = pos if pos.dtype.is_floating_point else pos.float()
        else:
            raise ValueError("pos must be a numpy array or torch tensor")
        super().__init__(x=x)

        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, (int, np.integer)):
                value = th.tensor(int(value), dtype=th.long)
            elif isinstance(value, np.ndarray):
                # keep booleans as bool tensors, floats as float32, others as long
                if value.dtype == np.bool_:
                    value = th.from_numpy(value).to(th.bool)
                else:
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

    def __inc__(self, key, value, *args, **kwargs):
        # Offset indices correctly when batching
        if key == "leaf_idx":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)
