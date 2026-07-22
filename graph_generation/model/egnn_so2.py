import os
import time
from pathlib import Path
import logging
log = logging.getLogger(__name__)
import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# types

from typing import Optional, List, Union
from torch_scatter import scatter_add, scatter_max

# pytorch geometric

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.nn import LayerNorm as PygLayerNorm
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PygLayerNorm = nn.LayerNorm  # fallback
    PYG_AVAILABLE = False
    
    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

from .egnn_pytorch import *

from graph_generation.method.helpers import (
    compute_branch_angles_parent_centric,
    assign_branch_angles_to_edges,
    assign_parent_scalar_to_edges,
)

# global linear attention

# --- Pad/unpad helpers for batched attention ---

def _pad_to_batch(flat: torch.Tensor, batch_ids: torch.Tensor):
    """Convert flat [sum_N, D] + batch_ids [sum_N] -> padded [B, N_max, D] + mask [B, N_max].

    Handles non-contiguous graph assignment (sampling path) via argsort.
    Returns (padded, mask, sorted_indices, counts) for unpadding.
    """
    device = flat.device
    B = int(batch_ids.max().item()) + 1
    N = flat.size(0)
    D = flat.size(-1)

    # Sort by graph ID to make contiguous
    sorted_indices = torch.argsort(batch_ids, stable=True)
    sorted_feats = flat[sorted_indices]
    sorted_batch = batch_ids[sorted_indices]

    # Per-graph counts
    _, counts = torch.unique_consecutive(sorted_batch, return_counts=True)
    N_max = int(counts.max().item())

    # Fast path: all graphs same size (e.g. global tokens) — just reshape
    if int(counts.min().item()) == N_max:
        padded = sorted_feats.reshape(B, N_max, D)
        mask = torch.ones(B, N_max, dtype=torch.bool, device=device)
        return padded, mask, sorted_indices, counts

    # Build within-graph offsets: offset[i] = i - start_of_its_graph
    starts = torch.zeros(B, device=device, dtype=torch.long)
    starts[1:] = counts.cumsum(0)[:-1]
    offsets = torch.arange(N, device=device) - starts[sorted_batch]

    # Scatter into padded tensor
    padded = flat.new_zeros(B, N_max, D)
    padded[sorted_batch, offsets] = sorted_feats

    # Build mask: True where valid
    mask = torch.arange(N_max, device=device).unsqueeze(0) < counts.unsqueeze(1)

    return padded, mask, sorted_indices, counts


def _unpad_from_batch(padded: torch.Tensor, counts: torch.Tensor,
                      sorted_indices: torch.Tensor, original_N: int):
    """Convert padded [B, N_max, D] back to flat [sum_N, D] in original order."""
    B, N_max, D = padded.shape
    device = padded.device

    # Rebuild within-graph offsets (same logic as _pad_to_batch)
    sorted_batch = torch.arange(B, device=device).repeat_interleave(counts)
    starts = torch.zeros(B, device=device, dtype=torch.long)
    starts[1:] = counts.cumsum(0)[:-1]
    total = int(counts.sum().item())
    offsets = torch.arange(total, device=device) - starts[sorted_batch]

    # Fast path: all graphs same size — just reshape
    if int(counts.min().item()) == N_max:
        flat_sorted = padded.reshape(total, D)
    else:
        flat_sorted = padded[sorted_batch, offsets]

    # Invert sort to restore original element order
    flat_original = flat_sorted.new_empty(original_N, D)
    flat_original[sorted_indices] = flat_sorted

    return flat_original


class Attention_Sparse(Attention):
    def __init__(self, **kwargs):
        """ Wraps the attention class to operate with pytorch-geometric inputs. """
        super(Attention_Sparse, self).__init__(**kwargs)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None: 
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1: 
            x, context = map(lambda t: rearrange(t, 'h d -> () h d'), (x, context))
            return self.forward(x, context, mask=None).squeeze() # get rid of batch dim
        # multiple examples in batch - do block-sparse by dense loop
        else:
            x_list = []
            aux_count = 0
            for bi,n_idxs in zip(*batch_uniques):
                x_list.append( 
                    self.sparse_forward(
                        x[aux_count:aux_count+n_idxs], 
                        context[aux_count:aux_count+n_idxs],
                        batch_uniques = (bi.unsqueeze(-1), n_idxs.unsqueeze(-1)) 
                    ) 
                )
                aux_count += int(n_idxs.item())
            return torch.cat(x_list, dim=0)

    @torch.no_grad()
    def _unique_counts(self, b: torch.Tensor):
        # returns (unique_ids, counts)
        return torch.unique(b, return_counts=True)

    def batched_forward(self, q: torch.Tensor, kv: torch.Tensor,
                        *, q_batch: torch.Tensor, kv_batch: torch.Tensor):
        """Single-kernel batched attention with padding and masking.
        Replaces the per-graph Python loop with one fused Attention.forward() call.
        """
        # Pad queries and KVs into [B, N_max, D] tensors
        q_padded, _q_mask, q_sort, q_counts = _pad_to_batch(q, q_batch)
        kv_padded, kv_mask, _kv_sort, _kv_counts = _pad_to_batch(kv, kv_batch)

        # Call base Attention.forward — mask is KV-side [B, N_kv_max]
        # (prevents queries from attending to padding positions in kv)
        out_padded = super().forward(q_padded, kv_padded, mask=kv_mask)

        # Unpad back to flat [sum_q, D] in original element order
        return _unpad_from_batch(out_padded, q_counts, q_sort, q.size(0))

    def sparse_forward_with_separate_batches(self,
                                           q: torch.Tensor,
                                           kv: torch.Tensor,
                                           *,
                                           q_batch: Optional[torch.Tensor] = None,
                                           kv_batch: Optional[torch.Tensor] = None,
                                           mask: Optional[torch.Tensor] = None):
        """Block-sparse over graphs with possibly different per-graph sizes for q vs kv.
        Expects q.shape == [sum_q, d], kv.shape == [sum_kv, d].
        If no batches are given, assumes a single graph and does dense.
        """
        assert (q_batch is None) == (kv_batch is None), "Pass both q_batch and kv_batch or neither."
        if q_batch is None:
            # single-graph fast path
            q_ = rearrange(q, 'n d -> () n d')
            kv_ = rearrange(kv, 'm d -> () m d')
            return super().forward(q_, kv_, mask=None).squeeze(0)

        return self.batched_forward(q, kv, q_batch=q_batch, kv_batch=kv_batch)


class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = PygLayerNorm(dim)
        self.norm_queries = PygLayerNorm(dim)
        self.attn1 = Attention_Sparse(dim=dim, heads=heads, dim_head=dim_head)
        self.attn2 = Attention_Sparse(dim=dim, heads=heads, dim_head=dim_head)
        self.ff_norm_x = PygLayerNorm(dim)
        self.ff_x = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ff_norm_q = PygLayerNorm(dim)
        self.ff_q = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self,
                x: torch.Tensor,
                queries: torch.Tensor,
                *,
                x_batch: Optional[torch.Tensor] = None,
                q_batch: Optional[torch.Tensor] = None):
        # norm
        x_n = self.norm_seq(x, batch=x_batch) if x_batch is not None else self.norm_seq(x)
        q_n = self.norm_queries(queries, batch=q_batch) if q_batch is not None else self.norm_queries(queries)
        # ISAB step 1: tokens ← nodes
        induced = self.attn1.sparse_forward_with_separate_batches(q_n, x_n, q_batch=q_batch, kv_batch=x_batch)
        # ISAB step 2: nodes ← tokens
        out = self.attn2.sparse_forward_with_separate_batches(x_n, induced, q_batch=x_batch, kv_batch=q_batch)
        # residuals
        x = x + out
        queries = queries + induced
        # FFN on both (stabilizes training)
        x_ = self.ff_norm_x(x, batch=x_batch) if x_batch is not None else self.ff_norm_x(x)
        x = x + self.ff_x(x_)
        q_ = self.ff_norm_q(queries, batch=q_batch) if q_batch is not None else self.ff_norm_q(queries)
        queries = queries + self.ff_q(q_)
        return x, queries


# define pytorch-geometric equivalents
# Main edits in the code below for SO(3) => SO(2) equivariance: UGEDIT

class SO2_EGNN(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and 
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        soft_edge = 0,
        norm_feats = False,
        norm_coors = False,
        norm_coors_scale_init = 1e-2,
        update_feats = True,
        update_coors = False, 
        dropout = 0.,
        coor_weights_clamp_value = None, 
        aggr = "add",
        # UGEDIT
        so2_axis=(0., 0., 1.), # axis of rotation for SO(2) equivariance, default z-axis
        anisotropic=False, # not implemented yet - two MLPs for xy and z separately UGEDIT
        # NEW
        add_local_angles: bool = True,
        angle_weighted_mean: bool = True,
        rbf_k: int = 0,
        rbf_gamma: float = 10.0,
        rbf_rho_max: float = 5.0,
        rbf_du_max: float = 3.0,
                 eps: float = 1e-8,
                 **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(SO2_EGNN, self).__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None

        # SO(2) axis unit vector UGEDITS
        u = torch.tensor(so2_axis, dtype=torch.float32)
        self.register_buffer('uhat', u / (u.norm() + 1e-8))
        self.anisotropic = anisotropic

        # NEW knobs
        self.add_local_angles = add_local_angles
        self.angle_weighted_mean = angle_weighted_mean
        self.rbf_k = rbf_k
        self.rbf_gamma = rbf_gamma
        if self.rbf_k > 0:
            class RBF(nn.Module):
                def __init__(self, k, lo, hi, gamma):
                    super().__init__()
                    self.register_buffer('mu', torch.linspace(lo, hi, k))
                    self.gamma = gamma
                def forward(self, s):
                    return torch.exp(-self.gamma * (s - self.mu.view(1, -1))**2)
            # Ranges are threaded from config/data stats (pos_scale_factor-normalized units).
            self.rbf_rho = RBF(self.rbf_k, 0.0, rbf_rho_max, self.rbf_gamma)   # rho in [0, rho_max]
            self.rbf_du  = RBF(self.rbf_k, -rbf_du_max, rbf_du_max, self.rbf_gamma)  # du in [-du_max, du_max]
        self.eps = eps

        # base edge scalars: rho, du, optionally local angles (+ option fourier)
        base_scalar_dim = (rbf_k if rbf_k > 0 else 1) * 2  # rho, du
        if self.add_local_angles:
            base_scalar_dim += 3                               # cosψ, sinψ, cosθ
        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + base_scalar_dim + (feats_dim * 2) # features for both nodes connected by edge
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), 
                                         nn.Sigmoid()
        ) if soft_edge else None # soft edge irrelevant for take one UGEDIT

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1. 
        self.node_norm = PygLayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        # COORS
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None # false for us

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None, 
                angle_data: List = None,  size: Size = None,
                parent_idx: OptTensor = None,
                pre_geom: Optional[dict] = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (2, n_edges)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        
        # If geometry was precomputed (static coordinates), reuse it.
        if pre_geom is not None:
            rel_coors = pre_geom['rel_coors']
            r_perp = pre_geom['r_perp']
            rho = pre_geom['rho']
            du = pre_geom['du']
            cospsi_edge = pre_geom.get('cospsi_edge')
            sinpsi_edge = pre_geom.get('sinpsi_edge')
            cos_theta_edge = pre_geom.get('cos_theta_edge')
        else:
            src, dst = edge_index
            rel_coors = coors[dst] - coors[src]  # (E, 3)
            # axis-aware decomposition
            du   = (rel_coors @ self.uhat)                                    # (E, )
            r_par = du[:, None] * self.uhat                                  # (E, 3)
            r_perp = rel_coors - r_par                                       # (E, 3)
            rho  = r_perp.norm(dim=-1, keepdim=True).clamp_min(self.eps)     # (E, 1)
            du = du[:, None]                                                 # (E, 1)
            cospsi_edge = None
            sinpsi_edge = None
            cos_theta_edge = None
            if self.add_local_angles:
                if parent_idx is None:
                    raise ValueError("parent_idx must be provided when add_local_angles=True.")
                cospsi_node, sinpsi_node, cos_theta_node = compute_branch_angles_parent_centric(
                    coors, parent_idx, self.uhat, eps=self.eps
                )
                cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(
                    edge_index, parent_idx, cospsi_node, sinpsi_node
                )
                cos_theta_edge = assign_parent_scalar_to_edges(
                    edge_index, parent_idx, cos_theta_node
                )

        if self.fourier_features > 0: # switched off for us
            rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        # --- Build edge features (rho, du, optional angles) ---
        base_feats = []
        if self.rbf_k > 0:
            rho_feat = self.rbf_rho(rho)
            du_feat  = self.rbf_du(du)
        else:
            rho_feat, du_feat = rho, du
        base_feats.extend([rho_feat, du_feat])

        if self.add_local_angles:
            if cospsi_edge is None or sinpsi_edge is None or cos_theta_edge is None:
                raise ValueError("Branch angles were not computed; check parent_idx input.")
            base_feats.extend([cospsi_edge, sinpsi_edge, cos_theta_edge])

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr] + base_feats, dim=-1)
        else:
            edge_attr_feats = torch.cat(base_feats, dim=-1)

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                                           coors=coors, rel_coors=rel_coors,
                                                           batch=batch)
        return torch.cat([coors_out, hidden_out], dim=-1)


    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp( torch.cat([x_i, x_j, edge_attr], dim=-1) )
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        
        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_weights = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp(min = -clamp_value, max = clamp_value)

            # normalize if needed
            # only isotropic version for now UGEDITS
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_weights * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp( torch.cat([hidden_feats, m_i], dim = -1) )
            hidden_out = kwargs["x"] + hidden_out
        else: 
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__) 


class SO2_EGNN_Network(nn.Module):
    r"""Sample GNN model architecture that uses the EGNN-Sparse
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """

    @staticmethod
    def _make_global_tokens(global_tokens_param: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (tokens, tokens_batch) where we tile m learned tokens for each graph in the batch."""
        # batch is shape [N], values 0..B-1
        device = batch.device
        B = int(batch.max().item()) + 1 if batch.numel() else 0
        if B == 0:
            return torch.empty(0, global_tokens_param.size(-1), device=device), torch.empty(0, dtype=torch.long, device=device)
        m, d = global_tokens_param.shape
        tokens = global_tokens_param.unsqueeze(0).expand(B, m, d).reshape(B * m, d)
        tokens_batch = torch.arange(B, device=device).repeat_interleave(m)
        return tokens, tokens_batch

    def _iter_egnn_layers(self):
        for L in self.mpnn_layers:
            if isinstance(L, nn.ModuleList):  # [ATTN, EGNN]
                yield L[1]
            else:
                yield L
    def __init__(self, n_layers, feats_dim, 
                 pos_dim = 3,
                 edge_attr_dim = 0, 
                 m_dim = 16,
                 tmd_in_dim = 0,
                 tmd_hidden_dim = 0,
                 num_classes = 0,
                 class_hidden_dim = 0,
                 fourier_features = 0,
                 soft_edge = 0,
                 embedding_nums=[], 
                 embedding_dims=[],
                 edge_embedding_nums=[], 
                 edge_embedding_dims=[],
                 update_coors=False, 
                 update_feats=True, 
                 norm_feats=True, 
                 norm_coors=False,
                 norm_coors_scale_init = 1e-2, 
                 dropout=0.,
                 coor_weights_clamp_value=None, 
                 aggr="add",
                 global_linear_attn_every = 0,
                 global_linear_attn_heads = 8,
                 global_linear_attn_dim_head = 64,
                 num_global_tokens = 4,
                 recalc=0 ,
                 # SO(2) knobs for layers
                 so2_axis=(0.,0.,1.),
                 add_local_angles=True,
                 angle_weighted_mean=True,
                 rbf_k=0,
                 rbf_gamma=10.0,
                 rbf_rho_max=5.0,
                 rbf_du_max=3.0,
                 eps=1e-8,
                 # offset head
                 LR_offset_head=False,
                 offset_head_hidden=128,
                 # root-child ordinal criterion: "first_edge" (legacy) | "axial_extent" (apical=ordinal 0)
                 root_child_order="first_edge",
                 ):
        super().__init__()

        self.n_layers         = n_layers
        # Read by Expansion.expand() at sampling: in "axial_extent" mode the per-step ordinal
        # correction is skipped for root children (their index-order/spawn-slot ordinal is kept,
        # so slot 0 = apical stays fixed, matching the training-time apical flag).
        self.root_child_order = root_child_order

        # Embeddings? solve here
        self.embedding_nums   = embedding_nums
        self.embedding_dims   = embedding_dims
        self.emb_layers       = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers     = nn.ModuleList()

        # instantiate point and edge embedding layers

        for i in range( len(self.embedding_dims) ):
            self.emb_layers.append(nn.Embedding(num_embeddings = embedding_nums[i],
                                                embedding_dim  = embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range( len(self.edge_embedding_dims) ):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim  = edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers      = nn.ModuleList()
        self.feats_dim        = feats_dim
        self.pos_dim          = pos_dim
        self.edge_attr_dim    = edge_attr_dim
        self.m_dim            = m_dim
        self.tmd_in_dim       = int(tmd_in_dim)
        self.tmd_hidden_dim   = int(tmd_hidden_dim)
        self.num_classes      = int(num_classes)
        self.class_hidden_dim = int(class_hidden_dim)
        self.fourier_features = fourier_features
        self.soft_edge        = soft_edge
        self.norm_feats       = norm_feats
        self.norm_coors       = norm_coors
        self.norm_coors_scale_init = norm_coors_scale_init
        self.update_feats     = update_feats
        self.update_coors     = update_coors
        self.dropout          = dropout
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc           = recalc
        self.eps              = eps

        # axis buffer needed for decode
        u = torch.tensor(so2_axis, dtype=torch.float32)
        self.register_buffer('uhat', u / (u.norm() + 1e-8))

        if self.tmd_hidden_dim < 0 or self.tmd_in_dim < 0:
            raise ValueError("tmd_in_dim and tmd_hidden_dim must be >= 0.")
        if self.tmd_hidden_dim > 0 and self.tmd_in_dim == 0:
            raise ValueError("tmd_in_dim must be > 0 when tmd_hidden_dim > 0.")
        if self.class_hidden_dim < 0 or self.num_classes < 0:
            raise ValueError("num_classes and class_hidden_dim must be >= 0.")
        if self.class_hidden_dim > 0 and self.num_classes == 0:
            raise ValueError("num_classes must be > 0 when class_hidden_dim > 0.")
        # Both conditioners reserve part of feats_dim; their combined width must fit.
        if self.tmd_hidden_dim + self.class_hidden_dim > self.feats_dim:
            raise ValueError("tmd_hidden_dim + class_hidden_dim cannot exceed feats_dim.")

        self.tmd_mlp = None
        if self.tmd_hidden_dim > 0:
            self.tmd_mlp = nn.Sequential(
                nn.Linear(self.tmd_in_dim, self.tmd_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.tmd_hidden_dim, self.tmd_hidden_dim),
            )

        # Cell-type conditioning: a one-hot(num_classes) fed through a single Linear
        # (== a learned embedding, but the one-hot stays explicit in the assembly).
        self.class_lin = None
        if self.class_hidden_dim > 0:
            self.class_lin = nn.Linear(self.num_classes, self.class_hidden_dim)

        # basis-coef head
        self.LR_offset_head = LR_offset_head
        if self.LR_offset_head:
            # Two-class gated offset heads; each is a 2-layer MLP producing (dx, dy, dz, expansion)
            self.offset_head_class0 = nn.Sequential(
                nn.Linear(self.feats_dim, offset_head_hidden),
                nn.SiLU(),
                nn.Linear(offset_head_hidden, 4),
            )
            self.offset_head_class1 = nn.Sequential(
                nn.Linear(self.feats_dim, offset_head_hidden),
                nn.SiLU(),
                nn.Linear(offset_head_hidden, 4),
            )
        else:
            self.offset_head = nn.Sequential(
                nn.Linear(self.feats_dim, offset_head_hidden),
                nn.SiLU(),
                nn.Linear(offset_head_hidden, 4),
            )
        # global attn irrelevant for take one UGEDIT
        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, self.feats_dim))
        
        # instantiate layers
        for i in range(n_layers):
            layer = SO2_EGNN(
                feats_dim = feats_dim,
                pos_dim = pos_dim,
                edge_attr_dim = edge_attr_dim,
                m_dim = m_dim,
                fourier_features = fourier_features,
                soft_edge = soft_edge,
                norm_feats = norm_feats,
                norm_coors = norm_coors,
                norm_coors_scale_init = norm_coors_scale_init,
                update_feats = update_feats,
                update_coors = update_coors,
                dropout = dropout,
                coor_weights_clamp_value = coor_weights_clamp_value,
                so2_axis = so2_axis,
                add_local_angles = add_local_angles,
                angle_weighted_mean = angle_weighted_mean,
                rbf_k = rbf_k,
                rbf_gamma = rbf_gamma,
                rbf_rho_max = rbf_rho_max,
                rbf_du_max = rbf_du_max,
                eps = eps,
            )

            # global attention case
            is_global_layer = self.has_global_attn and ((i + 1) % self.global_linear_attn_every) == 0
            if is_global_layer:
                attn_layer = GlobalLinearAttention_Sparse(dim=self.feats_dim, 
                                                   heads = global_linear_attn_heads, 
                                                   dim_head = global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([attn_layer, layer]))  # [ATTN, EGNN]
            # normal case
            else: 
                self.mpnn_layers.append(layer)
            

    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0,
                parent_idx: Optional[torch.Tensor] = None,
                tmd: Optional[torch.Tensor] = None,
                cell_class: Optional[torch.Tensor] = None,
                pre_geom: Optional[dict] = None):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        x = embedd_token(x, self.embedding_dims, self.emb_layers) # identity if no embedding layers
        # Per-graph conditioners (TMD structure vector and/or cell-type label) each
        # reserve a slice of feats_dim: embed per-graph -> broadcast by `batch` -> append
        # AFTER the real node features (so the LR_offset_head's fixed-index read below is
        # undisturbed). Fixed order: tmd, then class.
        reserved = self.tmd_hidden_dim + self.class_hidden_dim
        if reserved > 0:
            num_graphs = int(batch.max().item()) + 1
            cond_embs = []
            if self.tmd_hidden_dim > 0:
                if tmd is None:
                    raise ValueError("tmd must be provided when tmd_hidden_dim > 0.")
                if tmd.dim() != 2:
                    raise ValueError("tmd must be a 2D tensor of shape (B, tmd_in_dim).")
                if tmd.size(-1) != self.tmd_in_dim:
                    raise ValueError("tmd last dim must match tmd_in_dim.")
                tmd = tmd.to(device=x.device, dtype=x.dtype)
                tmd_emb = self.tmd_mlp(tmd)
                if tmd_emb.size(0) != num_graphs:
                    raise ValueError("tmd batch size must be == number of graphs in batch.")
                cond_embs.append(tmd_emb)
            if self.class_hidden_dim > 0:
                if cell_class is None:
                    raise ValueError("cell_class must be provided when class_hidden_dim > 0.")
                cell_class = cell_class.reshape(-1).to(device=x.device)
                if cell_class.numel() != num_graphs:
                    raise ValueError("cell_class must have one entry per graph in the batch.")
                class_onehot = torch.nn.functional.one_hot(
                    cell_class.long(), self.num_classes
                ).to(dtype=x.dtype)
                class_emb = self.class_lin(class_onehot)
                cond_embs.append(class_emb)

            coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
            expected_feats = self.feats_dim - reserved
            if feats.size(1) != expected_feats:
                raise ValueError(
                    "Input feature dim does not match feats_dim - tmd_hidden_dim - class_hidden_dim."
                )
            for emb in cond_embs:
                feats = torch.cat([feats, emb[batch]], dim=-1)
            x = torch.cat([coors, feats], dim=-1)

        class_feature = None
        if self.LR_offset_head:
            class_feature_idx = self.pos_dim + 1  # second entry of the input feature vector
            if x.size(1) <= class_feature_idx:
                raise ValueError("Expected at least two feature channels to extract class indicator.")
            class_feature = x[:, class_feature_idx].clone()

        # regulates wether to embedd edges each layer
        edges_need_embedding = True  
        # Precompute static geometry if coordinates will remain fixed (update_coors False in all layers)
        # If pre_geom was passed in (precomputed externally), skip internal computation.
        if pre_geom is None:
            static_coords = all((not getattr(L, 'update_coors', True)) for L in self._iter_egnn_layers()) # bit redundant for now as all layers same, but future-proof
            if parent_idx is not None and static_coords:
                _t0_geom = time.perf_counter()
                pre_geom = self._compute_static_so2_geometry(x[:, :self.pos_dim], edge_index, parent_idx) # we will be precomputing in current set-up
                log.debug("[SO2_EGNN_Network.forward N=%d E=%d] static_geometry=%.4fs", x.size(0), edge_index.size(1), time.perf_counter() - _t0_geom)

        _t0_mpnn = time.perf_counter()
        for i,layer in enumerate(self.mpnn_layers):
            
            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                if edge_attr is not None:
                    edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers) # identity if no embedding layers
                edges_need_embedding = False # embedd edges only once unless recalc (later)

            # pass layers
            is_global_layer = self.has_global_attn and ((i + 1) % self.global_linear_attn_every) == 0
            if isinstance(layer, nn.ModuleList):  # global block: [ATTN, EGNN]
                # (a) build tokens per graph
                tokens, tokens_batch = self._make_global_tokens(self.global_tokens, batch)

                # (b) run ISAB on features only
                coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
                feats, _ = layer[0](feats, tokens, x_batch=batch, q_batch=tokens_batch) # global attn step before every mpnn layer

                # (c) merge and continue with EGNN
                x = torch.cat([coors, feats], dim=-1)
                x = layer[1](x, edge_index, edge_attr, batch=batch, size=bsize,
                             parent_idx=parent_idx, pre_geom=pre_geom)
            else:
                # regular EGNN layer
                x = layer(x, edge_index, edge_attr, batch=batch, size=bsize,
                          parent_idx=parent_idx, pre_geom=pre_geom)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i%self.recalc == 0) and not (i == len(self.mpnn_layers)-1)) :
                edge_index, edge_attr, _ = recalc_edge(x) # returns attr, idx, any_other_info
                edges_need_embedding = True
            
        if x.is_cuda:
            torch.cuda.synchronize()
        log.debug(
            "[SO2_EGNN_Network.forward N=%d E=%d n_layers=%d] mpnn_total=%.4fs",
            x.size(0), edge_index.size(1), self.n_layers, time.perf_counter() - _t0_mpnn,
        )

        # decode per-node parent-relative offsets
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        N = coors.size(0)
        if parent_idx is None:
            raise ValueError("parent_idx is required to decode parent-relative offsets.")

        # head
        _t0_head = time.perf_counter()
        if not self.LR_offset_head:
            offset_state = self.offset_head(feats)
        else:
            if class_feature is None:
                raise ValueError("Class feature was not captured; cannot route to multi-head offset decoder.")
            class_mask = (class_feature > 0.5).unsqueeze(-1)  # True -> head 1, False -> head 0
            head0 = self.offset_head_class0(feats)
            head1 = self.offset_head_class1(feats)
            offset_state = torch.where(class_mask, head1, head0)  # apply class-specific decoder head
        
        rel_pred = offset_state[:, :3]
        expansion_pred = offset_state[:, 3:4]

        log.debug(
            "[SO2_EGNN_Network.forward N=%d] offset_head=%.4fs",
            x.size(0), time.perf_counter() - _t0_head,
        )
        return {"node_state": x, "rel_pred": rel_pred, "expansion_pred": expansion_pred}

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))

    def _compute_static_so2_geometry(self, coors: torch.Tensor, edge_index: torch.Tensor, parent_idx: torch.Tensor) -> dict:
        """Precompute SO(2) geometric quantities reused across layers when coordinates are static.
        Returns dict with per-edge and per-node frames & angle features.
        """
        src, dst = edge_index
        rel_coors = coors[dst] - coors[src]                   # (E,3)
        du = (rel_coors @ self.uhat)                          # (E,)
        r_par = du[:, None] * self.uhat
        r_perp = rel_coors - r_par
        rho = r_perp.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # (E,1)
        du = du[:, None]

        if parent_idx is None:
            raise ValueError("parent_idx must be provided for branch angle computation; received None.")

        cospsi_node, sinpsi_node, cos_theta_node = compute_branch_angles_parent_centric(
            coors, parent_idx, self.uhat, eps=self.eps
        )
        cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(
            edge_index, parent_idx, cospsi_node, sinpsi_node
        )
        cos_theta_edge = assign_parent_scalar_to_edges(
            edge_index, parent_idx, cos_theta_node
        )

        return {
            'rel_coors': rel_coors,
            'r_perp': r_perp,
            'rho': rho,
            'du': du,
            'cospsi_edge': cospsi_edge,
            'sinpsi_edge': sinpsi_edge,
            'cos_theta_edge': cos_theta_edge,
            'cospsi_node': cospsi_node,
            'sinpsi_node': sinpsi_node,
            'cos_theta_node': cos_theta_node,
        }
