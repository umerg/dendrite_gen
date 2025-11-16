import os
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

# axis-aligned fallback helper
def _axis_aligned_fallback_basis(uhat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a unit axis uhat that must be ±x, ±y or ±z, return a fixed orthonormal
    basis (e1, e2) spanning the plane perpendicular to uhat.
    Mapping (right-handed):
      x -> (e1=y, e2=z)
      y -> (e1=z, e2=x)
      z -> (e1=x, e2=y)
    """
    dev = uhat.device
    abs_u = uhat.abs()
    basis = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], device=dev)
    # check axis-aligned (ignore sign)
    if not any(torch.allclose(abs_u, b, atol=1e-5) for b in basis):
        raise AssertionError("SO2 axis must be aligned with ±x, ±y, or ±z.")

    # choose e1 according to which coordinate is 1 in abs(uhat)
    if torch.allclose(abs_u, basis[0], atol=1e-5):       # x-axis
        e1 = torch.tensor([0.,1.,0.], device=dev)
    elif torch.allclose(abs_u, basis[1], atol=1e-5):     # y-axis
        e1 = torch.tensor([0.,0.,1.], device=dev)
    else:                                                # z-axis
        e1 = torch.tensor([1.,0.,0.], device=dev)

    # right-handed e2
    e2 = torch.cross(uhat, e1, dim=-1)
    # both are unit already, but be safe numerically
    e1 = e1 / (e1.norm() + 1e-8)
    e2 = e2 / (e2.norm() + 1e-8)
    return e1, e2

# --- NEW: per-node SO(2) in-plane frames built from projected parent direction ---
def _build_so2_frames_from_parents(
        coors: torch.Tensor,
        uhat: torch.Tensor,
        parent_idx: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        
        """Return E: [N,2,3] with per-node in-plane frame (e1,e2).
        e1 = normalized projection of parent->node onto plane ⟂ uhat (fallback to axis-aligned basis if degenerate/root)
        e2 = uhat × e1
        """
        N = coors.size(0)
        device = coors.device

        # parent vectors v_t = x_t - x_parent(t); roots marked as -1
        has_parent = parent_idx >= 0
        parent = parent_idx.clamp(min=0)
        v = coors - coors[parent]
        v = torch.where(has_parent.view(-1,1), v, torch.zeros_like(v))

        # project to plane ⟂ u
        du_p = (v @ uhat).unsqueeze(-1)                   # [N, 1] 
        v_perp = v - du_p * uhat                          # [N, 3]
        v_norm = v_perp.norm(dim=-1, keepdim=True)        # [N, 1]

        # deterministic global in-plane fallback from the axis-aligned helper
        fb_e1, fb_e2 = _axis_aligned_fallback_basis(uhat)
        fb_e1 = fb_e1.expand_as(v_perp)  # [N, 3]
        fb_e2 = fb_e2.expand_as(v_perp)  # [N, 3]

        need_fb = (~has_parent) | (v_norm.squeeze(-1) <= 1e-6)
        e1 = torch.where(
            need_fb.view(-1, 1),
            fb_e1,                # global in-plane basis
            v_perp / (v_norm + eps),
        )
        e2 = torch.cross(uhat.expand_as(e1), e1, dim=-1)  # [N, 3]

        E = torch.stack([e1, e2], dim=1)  # [N,2,3]
        return E

# global linear attention

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

        # multi-graph path
        (uq, _), (uk, _) = self._unique_counts(q_batch), self._unique_counts(kv_batch)
        # assume the same set of graph ids exists in both batches
        assert torch.equal(uq, uk), "q_batch and kv_batch must index the same set of graphs in order."

        outs = []
        for gid in uq.tolist():
            q_sel = (q_batch == gid)
            k_sel = (kv_batch == gid)
            q_g = q[q_sel]
            kv_g = kv[k_sel]
            q_g = rearrange(q_g, 'n d -> () n d')
            kv_g = rearrange(kv_g, 'm d -> () m d')
            out_g = super().forward(q_g, kv_g, mask=None).squeeze(0)
            outs.append(out_g)
        return torch.cat(outs, dim=0)
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None: 
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1: 
            x, context = map(lambda t: rearrange(t, 'h d -> () h d'), (x, context))
            return self.forward(x, context, mask=None).squeeze() # get rid of batch dim
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
            return torch.cat(x_list, dim=0)


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

class SO2_EGNN_Sparse(MessagePassing):
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
        eps: float = 1e-8,
        # DEBUG: force use of global fallback frames instead of per-parent frames
        use_global_fallback_frames: bool = False,
        **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(SO2_EGNN_Sparse, self).__init__(**kwargs)
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

        # NEW: assert axis-aligned and cache axis-aligned fallback basis
        fb_e1, fb_e2 = _axis_aligned_fallback_basis(self.uhat)
        self.register_buffer('fallback_e1', fb_e1)
        self.register_buffer('fallback_e2', fb_e2)

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
            # TODO: thread these ranges from config/data stats
            rho_max, du_max = 5.0, 3.0
            self.rbf_rho = RBF(self.rbf_k, 0.0, rho_max, self.rbf_gamma)
            self.rbf_du  = RBF(self.rbf_k, -du_max, du_max, self.rbf_gamma)
        self.eps = eps
        # debug flag
        self.use_global_fallback_frames = use_global_fallback_frames

        # base edge scalars: rho, du, u_i, optionally cosφ/sinφ (+ option fourier)
        base_scalar_dim = (rbf_k if rbf_k > 0 else 1) * 2 + 1  # rho, du, u_i
        if self.add_local_angles:
            base_scalar_dim += 2                               # cosφ, sinφ
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
            row, col = edge_index
            r_perp = pre_geom['r_perp']
            rho = pre_geom['rho']
            du = pre_geom['du']
            u_i = pre_geom['u_i']
            cosphi = pre_geom.get('cosphi', None)
            sinphi = pre_geom.get('sinphi', None)
            have_angles = cosphi is not None and sinphi is not None
        else:
            src, dst = edge_index

            rel_coors = coors[dst] - coors[src]  # (E, 3)
            # axis-aware decomposition
            du   = (rel_coors @ self.uhat)                                    # (E, )
            r_par = du[:, None] * self.uhat                                  # (E, 3)
            r_perp = rel_coors - r_par                                       # (E, 3)
            rho  = r_perp.norm(dim=-1, keepdim=True).clamp_min(self.eps)     # (E, 1)
            du = du[:, None]                                                 # (E, 1)
            u_i = (coors[dst] @ self.uhat)[:, None]                          # (E, 1)
            have_angles = False  # will be set after computing angles

        if self.fourier_features > 0: # switched off for us
            rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        # --- Build edge features (rho, du, u_i, optional angles) ---
        base_feats = []
        if self.rbf_k > 0:
            rho_feat = self.rbf_rho(rho)
            du_feat  = self.rbf_du(du)
        else:
            rho_feat, du_feat = rho, du
        base_feats.extend([rho_feat, du_feat, u_i])

        if self.add_local_angles:
            if pre_geom is not None and have_angles:
                base_feats.extend([cosphi, sinphi])
            else:
                if parent_idx is None:
                    raise ValueError("parent_idx must be provided for local angle computation; received None.")
                
                # Compute local angles on-the-fly (coordinates may be changing if update_coors=True)
                E = _build_so2_frames_from_parents(coors, self.uhat, parent_idx, eps=self.eps)  # [N,2,3]
                if self.use_global_fallback_frames:
                    # override all frames with global fallback basis
                    N = coors.size(0)
                    e1_all = self.fallback_e1.expand(N, 3)
                    e2_all = self.fallback_e2.expand(N, 3)
                    E = torch.stack([e1_all, e2_all], dim=1)

                # choose frame based on hierarchy and define outward direction consistently
                parent_mask = (src == parent_idx[dst])               # [E]
                s = torch.where(parent_mask, 1.0, -1.0).unsqueeze(-1)  # parent edge: keep r=j->i; child edge: flip to i->child [E,1]
                r_out = s * rel_coors                                 # [E,3]

                # in-plane unit direction of r_out
                du_out = (r_out @ self.uhat)                                   # [E]
                r_out_par = du_out[:, None] * self.uhat                        # [E,3]
                r_out_perp = r_out - r_out_par                                 # [E,3]
                d2 = r_out_perp / (r_out_perp.norm(dim=-1, keepdim=True) + 1e-8)  # [E,3]


                # select frame: parent’s for parent edge; receiver’s for child edge
                E_parent = E[src]    # [E,2,3]
                E_recv   = E[dst]    # [E,2,3]
                E_use = torch.where(parent_mask.view(-1,1,1), E_parent, E_recv)
                e1 = E_use[:,0,:]
                e2 = E_use[:,1,:]

                cosphi = (d2 * e1).sum(dim=-1, keepdim=True)
                sinphi = (d2 * e2).sum(dim=-1, keepdim=True)

                base_feats.extend([cosphi, sinphi])

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


class SO2_EGNN_Sparse_Network(nn.Module):
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
                 eps=1e-8,
                 # offset head
                 add_offset_head=True,
                 offset_head_hidden=128,
                 # DEBUG: force using global fallback frames everywhere
                 use_global_fallback_frames: bool = False,
                 ):
        super().__init__()

        self.n_layers         = n_layers 

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
        self.use_global_fallback_frames = use_global_fallback_frames

        # axis buffer needed for decode
        u = torch.tensor(so2_axis, dtype=torch.float32)
        self.register_buffer('uhat', u / (u.norm() + 1e-8))

        # cache axis-aligned fallback basis for decode path
        fb_e1, fb_e2 = _axis_aligned_fallback_basis(self.uhat)
        self.register_buffer('fallback_e1', fb_e1)
        self.register_buffer('fallback_e2', fb_e2)

        # basis-coef head
        self.add_offset_head = add_offset_head
        if self.add_offset_head:
            self.offset_head = nn.Sequential(
                nn.Linear(self.feats_dim, offset_head_hidden), nn.SiLU(),
                nn.Linear(offset_head_hidden, offset_head_hidden), nn.SiLU(),
                nn.Linear(offset_head_hidden, 4)  # 3 for offset, 1 for expansion state
            )

        # global attn irrelevant for take one UGEDIT
        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, self.feats_dim))
        
        # instantiate layers
        for i in range(n_layers):
            layer = SO2_EGNN_Sparse(
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
                eps = eps,
                use_global_fallback_frames = use_global_fallback_frames,
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
                parent_idx: Optional[torch.Tensor] = None):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """

        # NODES - Embedd each dim to its target dimensions:
        x = embedd_token(x, self.embedding_dims, self.emb_layers) # identity if no embedding layers
        # regulates wether to embedd edges each layer
        edges_need_embedding = True  
        # Precompute static geometry if coordinates will remain fixed (update_coors False in all layers)
        pre_geom = None
        static_coords = all((not getattr(L, 'update_coors', True)) for L in self._iter_egnn_layers()) # bit redundant for now as all layers same, but future-proof
        if parent_idx is not None and static_coords:
            pre_geom = self._compute_static_so2_geometry(x[:, :self.pos_dim], edge_index, parent_idx) # we will be precomputing in current set-up

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
            
        # decode per-node parent-relative offsets
        if not self.add_offset_head:
            return {"node_state": x, "rel_pred": torch.zeros(x.size(0), 3, device=x.device, dtype=x.dtype)}

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        N = coors.size(0)
        if parent_idx is None:
            raise ValueError("parent_idx is required to decode parent-relative offsets.")

        # Use precomputed node frames if available
        if pre_geom is not None:
            e1 = pre_geom['e1_node']
            e2 = pre_geom['e2_node']
        else:
            raise NotImplementedError("Decoding with dynamic coordinates not implemented yet.")
            # # OLD LOGIC - WRONG: takes current nodes frame instead of parent's
            # parent = parent_idx.clamp(min=-1)
            # has_parent = parent >= 0
            # v = coors - coors[parent.clamp(min=0)]
            # v = torch.where(has_parent.view(-1,1), v, torch.zeros_like(v))
            # du_p = (v @ self.uhat).unsqueeze(-1)
            # v_perp = v - du_p * self.uhat
            # v_norm = v_perp.norm(dim=-1, keepdim=True)

            # # NEW fixed fallback
            # need_fb = (~has_parent) | (v_norm.squeeze(-1) <= 1e-6)
            # e1 = torch.where(
            #     need_fb.view(-1,1),
            #     self.fallback_e1.expand_as(v_perp),
            #     v_perp / (v_norm + 1e-8)
            # )
            # e2 = torch.cross(self.uhat.expand_as(e1), e1, dim=-1)

        # head
        abce = self.offset_head(feats)
        a, b, c, e = abce[:,0:1], abce[:,1:2], abce[:,2:3], abce[:,3:4] # basis coeffs + expansion state
        rel_pred = a*e1 + b*e2 + c*self.uhat
        expansion_pred = e

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
        u_i = (coors[dst] @ self.uhat)[:, None]
        
        # compute local angles
        if parent_idx is None:
            raise ValueError("parent_idx must be provided for local angle computation; received None.")

        # build per-node frames once
        E = _build_so2_frames_from_parents(coors, self.uhat, parent_idx, eps=self.eps)  # [N,2,3]
        if self.use_global_fallback_frames:
            N_nodes = coors.size(0)
            e1_all = self.fallback_e1.expand(N_nodes, 3)
            e2_all = self.fallback_e2.expand(N_nodes, 3)
            E = torch.stack([e1_all, e2_all], dim=1)

            # choose frame based on hierarchy and define outward direction consistently
            parent_mask = (src == parent_idx[dst])               # [E]
            r_out = rel_coors                                 # [E,3]
        else:
            # choose frame based on hierarchy and define outward direction consistently
            parent_mask = (src == parent_idx[dst])               # [E]
            s = torch.where(parent_mask, 1.0, -1.0).unsqueeze(-1)  # parent edge: keep r=j->i; child edge: flip to i->child [E,1]
            r_out = s * rel_coors                                 # [E,3]
        
        # in-plane unit direction of r_out
        du_out = (r_out @ self.uhat)                                   # [E]
        r_out_par = du_out[:, None] * self.uhat                        # [E,3]
        r_out_perp = r_out - r_out_par                                 # [E,3]
        d2 = r_out_perp / (r_out_perp.norm(dim=-1, keepdim=True) + self.eps)  # [E,3]


        # select frame: parent’s for parent edge; receiver’s for child edge
        E_src = E[src]    # [E,2,3]
        E_dst   = E[dst]    # [E,2,3]
        E_use = torch.where(parent_mask.view(-1,1,1), E_src, E_dst)
        e1 = E_use[:,0,:]
        e2 = E_use[:,1,:]

        cosphi = (d2 * e1).sum(dim=-1, keepdim=True)
        sinphi = (d2 * e2).sum(dim=-1, keepdim=True)

        # parent frames for decoding step
        if self.use_global_fallback_frames:
            # simply broadcast global fallback frames for all nodes
            e1_node = self.fallback_e1.expand(coors.size(0), 3)
            e2_node = self.fallback_e2.expand(coors.size(0), 3)
        else:
            parent_frames = E[parent_idx.clamp(min=0)]  # [N,2,3]
            need_fb = (parent_idx < 0).view(-1, 1, 1)   # [N,1,1]
            fallback_frames = torch.stack(
                [self.fallback_e1, self.fallback_e2], dim=0
            ).unsqueeze(0)                               # [1,2,3]
            E_parent = torch.where(need_fb, fallback_frames, parent_frames)  # [N,2,3]
            e1_node = E_parent[:,0,:]
            e2_node = E_parent[:,1,:]

        # ---- DEBUG GEOMETRY VISUALIZATION (optional) ----
        if os.environ.get("GEOM_DEBUG", "0") == "1":
            try:
                import random
                from utils.debug_helpers import plot_geometry_debug
                N = coors.size(0)
                # prefer nodes with at least one incident edge for richer visualization
                incident = ((src == dst) == False)  # dummy to ensure tensor exists
                deg_counts = torch.zeros(N, dtype=torch.long, device=coors.device)
                deg_counts.scatter_add_(0, src, torch.ones_like(src))
                deg_counts.scatter_add_(0, dst, torch.ones_like(dst))
                candidates = (deg_counts > 0).nonzero(as_tuple=False).flatten().tolist()
                if not candidates:
                    candidates = list(range(N))
                node_id = random.choice(candidates)
                parent_id = int(parent_idx[node_id].item()) if parent_idx[node_id] >= 0 else None

                incoming_mask = (dst == node_id)
                outgoing_mask = (src == node_id)
                incoming_src = src[incoming_mask]
                outgoing_dst = dst[outgoing_mask]

                e1_self = E[node_id,0]
                e2_self = E[node_id,1]
                e1_par = E[parent_id,0] if parent_id is not None else None
                e2_par = E[parent_id,1] if parent_id is not None else None

                edge_vecs_in = []
                edge_decomp_in = []
                angles_in = []
                for p in incoming_src.tolist():
                    v = coors[node_id] - coors[p]
                    du_local = torch.dot(v, self.uhat)
                    r_par_local = du_local * self.uhat
                    r_perp_local = v - r_par_local
                    edge_vecs_in.append(v.detach().cpu())
                    edge_decomp_in.append((r_par_local.detach().cpu(), r_perp_local.detach().cpu()))
                    d2 = r_perp_local / (r_perp_local.norm() + 1e-8)
                    c = torch.dot(d2, e1_self).item()
                    s_ang = torch.dot(d2, e2_self).item()
                    angles_in.append((c, s_ang))

                edge_vecs_out = []
                edge_decomp_out = []
                angles_out = []
                for c_idx in outgoing_dst.tolist():
                    v = coors[c_idx] - coors[node_id]
                    du_local = torch.dot(v, self.uhat)
                    r_par_local = du_local * self.uhat
                    r_perp_local = v - r_par_local
                    edge_vecs_out.append(v.detach().cpu())
                    edge_decomp_out.append((r_par_local.detach().cpu(), r_perp_local.detach().cpu()))
                    d2 = r_perp_local / (r_perp_local.norm() + 1e-8)
                    c = torch.dot(d2, e1_self).item()
                    s_ang = torch.dot(d2, e2_self).item()
                    angles_out.append((c, s_ang))

                out_dir = Path(os.getcwd()) / "geometry_debug"
                out_path = plot_geometry_debug(
                    pos=coors.detach().cpu(),
                    node_id=node_id,
                    parent_id=parent_id,
                    neighbor_in=incoming_src.tolist(),
                    neighbor_out=outgoing_dst.tolist(),
                    uhat=self.uhat.detach().cpu(),
                    e1_node=e1_self.detach().cpu(),
                    e2_node=e2_self.detach().cpu(),
                    e1_parent=e1_par.detach().cpu() if e1_par is not None else None,
                    e2_parent=e2_par.detach().cpu() if e2_par is not None else None,
                    edge_vecs_in=edge_vecs_in,
                    edge_vecs_out=edge_vecs_out,
                    edge_decomp_in=edge_decomp_in,
                    edge_decomp_out=edge_decomp_out,
                    angles_in=angles_in,
                    angles_out=angles_out,
                    out_dir=out_dir,
                    prefix="geom",
                )
                log.info(f"[GEOM_DEBUG] Saved geometry debug figure: {out_path}")
            except Exception as e:
                import traceback
                log.warning(f"[GEOM_DEBUG] Failed to generate geometry debug plot: {e}\n{traceback.format_exc()}")

        return {
            'rel_coors': rel_coors,
            'r_perp': r_perp,
            'rho': rho,
            'du': du,
            'u_i': u_i,
            'e1_node': e1_node,
            'e2_node': e2_node,
            'cosphi': cosphi,
            'sinphi': sinphi,
        }
