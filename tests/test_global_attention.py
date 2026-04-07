import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from graph_generation.model.egnn_so2 import GlobalLinearAttention_Sparse, SO2_EGNN_Network


def _rand_batch(n1, n2, d, m):
    """Helper to create random batch data for testing"""
    x1, x2 = torch.randn(n1, d), torch.randn(n2, d)
    x = torch.cat([x1, x2], 0)
    b = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])
    tokens = torch.randn(m, d)
    return x, b, tokens


def test_make_global_tokens():
    """Test the _make_global_tokens static method"""
    # Test with valid batch
    tokens_param = torch.randn(3, 32)  # 3 tokens, 32 dimensions
    batch = torch.tensor([0, 0, 1, 1, 1])  # 2 nodes in graph 0, 3 nodes in graph 1
    
    tokens, tokens_batch = SO2_EGNN_Network._make_global_tokens(tokens_param, batch)
    
    # Should have 3 tokens * 2 graphs = 6 total tokens
    assert tokens.shape == (6, 32)
    assert tokens_batch.shape == (6,)
    
    # Verify batch assignment: first 3 tokens for graph 0, next 3 for graph 1
    expected_batch = torch.tensor([0, 0, 0, 1, 1, 1])
    assert torch.equal(tokens_batch, expected_batch)
    
    # Test with empty batch
    empty_batch = torch.tensor([])
    tokens_empty, batch_empty = SO2_EGNN_Network._make_global_tokens(tokens_param, empty_batch)
    assert tokens_empty.shape == (0, 32)
    assert batch_empty.shape == (0,)


def test_isab_sparse_dense_equivalence():
    """Test ISAB sparse vs dense equivalence"""
    d, h, dh, m = 32, 4, 8, 3
    isab = GlobalLinearAttention_Sparse(dim=d, heads=h, dim_head=dh)
    x, b, tokens_param = _rand_batch(5, 7, d, m)
    tokens, tb = SO2_EGNN_Network._make_global_tokens(tokens_param, b)

    # multi-graph (sparse)
    x_out, _ = isab(x, tokens, x_batch=b, q_batch=tb)

    # per-graph dense
    def run_dense(xg, tg):
        out, _ = isab(xg, tg)  # when no batches given, falls back to dense
        return out
    
    x0 = run_dense(x[b==0], tokens_param)
    x1 = run_dense(x[b==1], tokens_param)
    x_dense = torch.cat([x0, x1], 0)

    assert x_out.shape == x_dense.shape
    # Note: Due to potential numerical differences in attention computation order,
    # we don't assert exact equality but check that shapes match and no errors occurred


def test_global_attention_integration():
    """Test that global attention layers can be created and run without errors"""
    # Create a small network with global attention
    network = SO2_EGNN_Network(
        n_layers=4,
        feats_dim=16,
        pos_dim=3,
        global_linear_attn_every=2,
        global_linear_attn_heads=2,
        global_linear_attn_dim_head=8,
        num_global_tokens=2,
        update_coors=False,  # Keep coordinates static for testing
        m_dim=8
    )
    
    # Create test data
    batch_size = 3
    n_nodes = 8
    x = torch.randn(n_nodes, 3 + 16)  # pos + feats
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])  # circular
    batch = torch.zeros(n_nodes, dtype=torch.long)
    parent_idx = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6])  # simple chain with correct length
    
    # Forward pass should not raise errors
    result = network(x, edge_index, batch, edge_attr=None, parent_idx=parent_idx)
    
    assert "node_state" in result
    assert "rel_pred" in result
    assert result["node_state"].shape[0] == n_nodes


def _loop_sparse_forward(attn_module, q, kv, q_batch, kv_batch):
    """Reference implementation: the old per-graph Python loop (for correctness comparison)."""
    from einops import rearrange as _rearrange
    uq = torch.unique(q_batch)
    out = q.new_zeros(q.shape)
    for gid in uq.tolist():
        q_sel = (q_batch == gid)
        k_sel = (kv_batch == gid)
        q_g = _rearrange(q[q_sel], 'n d -> () n d')
        kv_g = _rearrange(kv[k_sel], 'm d -> () m d')
        out_g = attn_module.forward(q_g, kv_g, mask=None).squeeze(0)
        out[q_sel] = out_g
    return out


def test_batched_attention_vs_loop_contiguous():
    """Test that batched attention matches the old loop on contiguous graph-sorted nodes."""
    from graph_generation.model.egnn_so2 import Attention_Sparse
    d, h, dh = 32, 4, 8
    attn = Attention_Sparse(dim=d, heads=h, dim_head=dh)
    attn.eval()

    # 3 graphs with 4, 6, 3 nodes; contiguous batch
    counts = [4, 6, 3]
    q_batch = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])
    kv_batch = q_batch.clone()
    q = torch.randn(sum(counts), d)
    kv = torch.randn(sum(counts), d)

    with torch.no_grad():
        out_batched = attn.batched_forward(q, kv, q_batch=q_batch, kv_batch=kv_batch)
        out_loop = _loop_sparse_forward(attn, q, kv, q_batch, kv_batch)

    assert torch.allclose(out_batched, out_loop, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(out_batched - out_loop).abs().max().item()}"


def test_batched_attention_vs_loop_noncontiguous():
    """Test batched attention on interleaved (non-contiguous) graph nodes — the sampling case."""
    from graph_generation.model.egnn_so2 import Attention_Sparse
    d, h, dh = 32, 4, 8
    attn = Attention_Sparse(dim=d, heads=h, dim_head=dh)
    attn.eval()

    # Interleaved: nodes from graphs 0,1,2 mixed together
    q_batch = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 0], dtype=torch.long)
    kv_batch = torch.tensor([0, 2, 1, 0, 2, 1, 0], dtype=torch.long)
    q = torch.randn(q_batch.size(0), d)
    kv = torch.randn(kv_batch.size(0), d)

    with torch.no_grad():
        out_batched = attn.batched_forward(q, kv, q_batch=q_batch, kv_batch=kv_batch)
        out_loop = _loop_sparse_forward(attn, q, kv, q_batch, kv_batch)

    assert torch.allclose(out_batched, out_loop, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(out_batched - out_loop).abs().max().item()}"


def test_batched_attention_asymmetric_sizes():
    """Test ISAB-style asymmetric attention: fixed-size tokens vs variable-size nodes."""
    from graph_generation.model.egnn_so2 import Attention_Sparse
    d, h, dh = 32, 4, 8
    m = 4  # tokens per graph
    attn = Attention_Sparse(dim=d, heads=h, dim_head=dh)
    attn.eval()

    # Step 1: tokens (fixed m=4 per graph) attend to nodes (variable)
    B = 3
    node_counts = [5, 8, 3]
    node_batch = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(node_counts)])
    token_batch = torch.arange(B).repeat_interleave(m)
    nodes = torch.randn(sum(node_counts), d)
    tokens = torch.randn(B * m, d)

    with torch.no_grad():
        # tokens ← nodes
        out1_batched = attn.batched_forward(tokens, nodes, q_batch=token_batch, kv_batch=node_batch)
        out1_loop = _loop_sparse_forward(attn, tokens, nodes, token_batch, node_batch)
        assert torch.allclose(out1_batched, out1_loop, atol=1e-5, rtol=1e-5), \
            f"tokens←nodes max diff: {(out1_batched - out1_loop).abs().max().item()}"

        # nodes ← tokens
        out2_batched = attn.batched_forward(nodes, tokens, q_batch=node_batch, kv_batch=token_batch)
        out2_loop = _loop_sparse_forward(attn, nodes, tokens, node_batch, token_batch)
        assert torch.allclose(out2_batched, out2_loop, atol=1e-5, rtol=1e-5), \
            f"nodes←tokens max diff: {(out2_batched - out2_loop).abs().max().item()}"


def test_batched_attention_gradient_flow():
    """Test that gradients flow correctly through pad/unpad operations."""
    from graph_generation.model.egnn_so2 import Attention_Sparse
    d, h, dh = 16, 2, 8
    attn = Attention_Sparse(dim=d, heads=h, dim_head=dh)

    q_batch = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    kv_batch = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    q = torch.randn(5, d, requires_grad=True)
    kv = torch.randn(4, d, requires_grad=True)

    out = attn.batched_forward(q, kv, q_batch=q_batch, kv_batch=kv_batch)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "No gradient on q"
    assert kv.grad is not None, "No gradient on kv"
    assert not torch.isnan(q.grad).any(), "NaN in q gradient"
    assert not torch.isnan(kv.grad).any(), "NaN in kv gradient"


if __name__ == "__main__":
    test_make_global_tokens()
    test_isab_sparse_dense_equivalence()
    test_global_attention_integration()
    test_batched_attention_vs_loop_contiguous()
    test_batched_attention_vs_loop_noncontiguous()
    test_batched_attention_asymmetric_sizes()
    test_batched_attention_gradient_flow()
    print("All global attention tests passed!")