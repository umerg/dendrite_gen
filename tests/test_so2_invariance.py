"""Tests that root child ordering is SO(2)-invariant under rotations around uhat."""
import math
import torch as th
import pytest

from graph_generation.method.helpers import (
    _order_root_children_by_uhat,
    compute_root_child_angles,
    compute_geo_lr_mask,
    compute_geo_angle_for_new_leaves,
)


def _angles_close_mod2pi(a: th.Tensor, b: th.Tensor, atol: float = 1e-5):
    """Assert angles are close modulo 2π."""
    diff = (a - b) % (2 * math.pi)
    # diff should be near 0 or near 2π
    diff = th.min(diff, 2 * math.pi - diff)
    assert diff.max().item() < atol, f"Angles differ by up to {diff.max().item()}"


def _rotation_matrix_around_axis(uhat: th.Tensor, angle: float) -> th.Tensor:
    """Rodrigues' rotation: rotate by `angle` radians around unit vector `uhat`."""
    c = math.cos(angle)
    s = math.sin(angle)
    ux, uy, uz = uhat.tolist()
    return th.tensor([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)   ],
    ], dtype=uhat.dtype)


def _make_root_tree(k: int, uhat: th.Tensor):
    """Build a simple tree: root at origin with k children at different positions.

    Returns pos [k+1, 3], parent_idx [k+1].
    """
    # Root at origin
    pos = [th.zeros(3)]
    parent_idx = [-1]

    for i in range(k):
        # Spread children in 3D: vary both perp-plane angle and uhat component
        angle = 2 * math.pi * i / max(k, 1) + 0.3  # offset so not axis-aligned
        r_perp = 1.0 + 0.2 * i
        z_component = -0.5 + i * 0.4  # varying uhat components
        # Build position in a canonical frame, will be rotated in tests
        e1 = th.tensor([1.0, 0.0, 0.0])
        e2 = th.cross(uhat, e1)
        if e2.norm() < 1e-6:
            e1 = th.tensor([0.0, 1.0, 0.0])
            e2 = th.cross(uhat, e1)
        e2 = e2 / e2.norm()
        e1_perp = th.cross(e2, uhat)
        e1_perp = e1_perp / e1_perp.norm()

        child_pos = (
            r_perp * (math.cos(angle) * e1_perp + math.sin(angle) * e2)
            + z_component * uhat
        )
        pos.append(child_pos)
        parent_idx.append(0)  # all children of root

    return th.stack(pos), th.tensor(parent_idx, dtype=th.long)


class TestOrderRootChildrenByUhat:
    """Test _order_root_children_by_uhat directly."""

    def test_k2_matches_z_convention(self):
        """For k=2, child_0 should be the one with lower uhat component."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        # child A at z=2, child B at z=-1
        offsets = th.tensor([
            [1.0, 0.0, 2.0],   # higher z
            [0.5, 0.3, -1.0],  # lower z
        ])
        sorted_idx, fwd0, delta = _order_root_children_by_uhat(offsets, uhat)
        assert sorted_idx[0].item() == 1, "child_0 should be lower z (index 1)"
        assert sorted_idx[1].item() == 0

    def test_k3_ordinals(self):
        """For k=3, ordinals should be 0.0, 0.5, 1.0."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        offsets = th.tensor([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [-1.0, 0.0, 0.0],
        ])
        sorted_idx, _, _ = _order_root_children_by_uhat(offsets, uhat)
        # child with lowest z = index 1 (z=-0.5) should be child_0
        assert sorted_idx[0].item() == 1

    def test_rotation_invariance(self):
        """Rotating offsets around uhat should not change sorted_idx or ordinals."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        offsets = th.tensor([
            [1.0, 0.5, 2.0],
            [-0.3, 1.0, -1.0],
            [0.7, -0.8, 0.3],
        ])
        sorted_ref, _, delta_ref = _order_root_children_by_uhat(offsets, uhat)

        for angle in [0.5, 1.3, math.pi, 2.7, 5.5]:
            R = _rotation_matrix_around_axis(uhat, angle)
            offsets_rot = (R @ offsets.T).T
            sorted_rot, _, delta_rot = _order_root_children_by_uhat(offsets_rot, uhat)
            assert th.equal(sorted_ref, sorted_rot), (
                f"Sorted indices changed under rotation by {angle:.2f}"
            )
            _angles_close_mod2pi(delta_ref, delta_rot)

    def test_tiebreaker_perp_distance(self):
        """When uhat components are equal, child with largest perp distance is child_0."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        offsets = th.tensor([
            [0.5, 0.0, 1.0],  # perp dist = 0.5
            [2.0, 0.0, 1.0],  # perp dist = 2.0, same z
        ])
        sorted_idx, _, _ = _order_root_children_by_uhat(offsets, uhat)
        assert sorted_idx[0].item() == 1, "Tiebreaker: larger perp distance should be child_0"

    def test_degenerate_on_axis(self):
        """When all children are on the uhat axis (zero perp component)."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        offsets = th.tensor([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.5],
        ])
        sorted_idx, fwd0, _ = _order_root_children_by_uhat(offsets, uhat)
        # Should sort by uhat ascending: -1.0, 0.5, 2.0 → indices 1, 2, 0
        assert sorted_idx[0].item() == 1
        assert sorted_idx[1].item() == 2
        assert sorted_idx[2].item() == 0
        assert fwd0.norm().item() < 1e-6, "fwd0 should be zero for degenerate case"


class TestComputeRootChildAngles:
    """Test compute_root_child_angles with full tree structure."""

    def test_rotation_invariance_k2(self):
        """k=2 root children: geo_ordinal unchanged under rotation around uhat."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(2, uhat)
        lr_mask = compute_geo_lr_mask(pos, parent_idx, uhat=uhat)
        ordinal_ref, dt_ref = compute_root_child_angles(pos, parent_idx, uhat, lr_mask)

        for angle in [0.7, math.pi / 3, math.pi, 4.1]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            lr_rot = compute_geo_lr_mask(pos_rot, parent_idx, uhat=uhat)
            ordinal_rot, dt_rot = compute_root_child_angles(pos_rot, parent_idx, uhat, lr_rot)
            th.testing.assert_close(ordinal_ref, ordinal_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)

    def test_rotation_invariance_k3(self):
        """k=3 root children: geo_ordinal unchanged under rotation around uhat."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(3, uhat)
        lr_mask = compute_geo_lr_mask(pos, parent_idx, uhat=uhat)
        ordinal_ref, dt_ref = compute_root_child_angles(pos, parent_idx, uhat, lr_mask)

        for angle in [0.7, math.pi / 3, math.pi, 4.1]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            lr_rot = compute_geo_lr_mask(pos_rot, parent_idx, uhat=uhat)
            ordinal_rot, dt_rot = compute_root_child_angles(pos_rot, parent_idx, uhat, lr_rot)
            th.testing.assert_close(ordinal_ref, ordinal_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)

    def test_rotation_invariance_k5(self):
        """k=5 root children: geo_ordinal unchanged under rotation around uhat."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(5, uhat)
        lr_mask = compute_geo_lr_mask(pos, parent_idx, uhat=uhat)
        ordinal_ref, dt_ref = compute_root_child_angles(pos, parent_idx, uhat, lr_mask)

        for angle in [0.3, 1.5, math.pi, 5.0]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            lr_rot = compute_geo_lr_mask(pos_rot, parent_idx, uhat=uhat)
            ordinal_rot, dt_rot = compute_root_child_angles(pos_rot, parent_idx, uhat, lr_rot)
            th.testing.assert_close(ordinal_ref, ordinal_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)

    def test_k2_matches_lr_convention(self):
        """For k=2, ordinal should match geo_lr_mask: left(True)→0.0, right(False)→1.0."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos = th.tensor([
            [0.0, 0.0, 0.0],   # root
            [1.0, 0.0, -1.0],  # child with lower z → left → ordinal 0.0
            [0.5, 0.5, 2.0],   # child with higher z → right → ordinal 1.0
        ])
        parent_idx = th.tensor([-1, 0, 0])
        lr_mask = compute_geo_lr_mask(pos, parent_idx, uhat=uhat)
        ordinal, _ = compute_root_child_angles(pos, parent_idx, uhat, lr_mask)

        # Node 1 has lower z → should be child_0 → ordinal 0.0
        assert ordinal[1].item() == pytest.approx(0.0, abs=1e-6)
        # Node 2 has higher z → should be child_1 → ordinal 1.0
        assert ordinal[2].item() == pytest.approx(1.0, abs=1e-6)


class TestComputeGeoAngleForNewLeaves:
    """Test compute_geo_angle_for_new_leaves SO(2) invariance."""

    def test_rotation_invariance_root_children(self):
        """Post-diffusion ordinal for root children is SO(2)-invariant."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(3, uhat)
        new_leaf_idx = th.tensor([1, 2, 3], dtype=th.long)  # all root children

        angle_ref, valid_ref = compute_geo_angle_for_new_leaves(
            pos, parent_idx, new_leaf_idx, uhat=uhat,
        )

        for angle in [0.7, math.pi / 3, math.pi, 4.1]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            angle_rot, valid_rot = compute_geo_angle_for_new_leaves(
                pos_rot, parent_idx, new_leaf_idx, uhat=uhat,
            )
            th.testing.assert_close(angle_ref, angle_rot, atol=1e-5, rtol=1e-5)
            assert th.equal(valid_ref, valid_rot)
