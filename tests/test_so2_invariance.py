"""Tests that root child ordering is SO(2)-invariant under rotations around uhat."""
import math
import torch as th
import pytest

from graph_generation.method.helpers import (
    _order_root_children_by_uhat,
    _compute_tree_directions,
    compute_geo_order,
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


class TestComputeGeoOrder:
    """Test compute_geo_order with full tree structure."""

    def test_rotation_invariance_k2(self):
        """k=2 root children: geo_ordinal unchanged under rotation around uhat."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(2, uhat)
        ordinal_ref, dt_ref = compute_geo_order(pos, parent_idx, uhat)

        for angle in [0.7, math.pi / 3, math.pi, 4.1]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            ordinal_rot, dt_rot = compute_geo_order(pos_rot, parent_idx, uhat)
            th.testing.assert_close(ordinal_ref, ordinal_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)

    def test_rotation_invariance_k3(self):
        """k=3 root children: geo_ordinal unchanged under rotation around uhat."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(3, uhat)
        ordinal_ref, dt_ref = compute_geo_order(pos, parent_idx, uhat)

        for angle in [0.7, math.pi / 3, math.pi, 4.1]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            ordinal_rot, dt_rot = compute_geo_order(pos_rot, parent_idx, uhat)
            th.testing.assert_close(ordinal_ref, ordinal_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)

    def test_rotation_invariance_k5(self):
        """k=5 root children: geo_ordinal unchanged under rotation around uhat."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(5, uhat)
        ordinal_ref, dt_ref = compute_geo_order(pos, parent_idx, uhat)

        for angle in [0.3, 1.5, math.pi, 5.0]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            ordinal_rot, dt_rot = compute_geo_order(pos_rot, parent_idx, uhat)
            th.testing.assert_close(ordinal_ref, ordinal_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)

    @pytest.mark.parametrize("k", [11, 12, 13, 14, 15, 16])
    def test_rotation_invariance_large_k(self, k):
        """k=11..16 root children: geo_ordinal stays SO(2)-invariant across the widened
        MAX_CHILDREN=16 one-hot range (guards ranks that previously clamped onto bit 9).

        Children are placed at golden-angle azimuths (not evenly spaced) so no two sit
        exactly opposite — evenly-spaced children would put one on the atan2 branch cut,
        a measure-zero degeneracy that doesn't occur in real morphologies.
        """
        uhat = th.tensor([0.0, 0.0, 1.0])
        golden = 2.39996322972865332  # radians; irrational fraction of 2π -> no azimuth ties
        pos = [th.zeros(3)]
        parent_idx = [-1]
        for i in range(k):
            azim = i * golden
            r_perp = 1.0 + 0.13 * i
            z = -0.5 + 0.37 * i  # strictly increasing, distinct uhat components
            pos.append(th.tensor([r_perp * math.cos(azim), r_perp * math.sin(azim), z]))
            parent_idx.append(0)
        pos = th.stack(pos)
        parent_idx = th.tensor(parent_idx, dtype=th.long)

        ordinal_ref, dt_ref = compute_geo_order(pos, parent_idx, uhat)
        # All k children get distinct ranks 0..k-1 (no collision/clamping in the ordinal).
        root_ordinals = sorted(int(round(o)) for o in ordinal_ref[parent_idx == 0].tolist())
        assert root_ordinals == list(range(k)), f"expected ranks 0..{k-1}, got {root_ordinals}"

        for angle in [0.3, 1.5, math.pi, 5.0]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            ordinal_rot, dt_rot = compute_geo_order(pos_rot, parent_idx, uhat)
            th.testing.assert_close(ordinal_ref, ordinal_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)

    def test_k2_matches_lr_convention(self):
        """For k=2, lowest uhat = child_0 = ordinal 0.0, highest = ordinal 1.0."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos = th.tensor([
            [0.0, 0.0, 0.0],   # root
            [1.0, 0.0, -1.0],  # child with lower z → child_0 → ordinal 0.0
            [0.5, 0.5, 2.0],   # child with higher z → child_1 → ordinal 1.0
        ])
        parent_idx = th.tensor([-1, 0, 0])
        ordinal, _ = compute_geo_order(pos, parent_idx, uhat)

        # Node 1 has lower z → should be child_0 → ordinal 0.0
        assert ordinal[1].item() == pytest.approx(0.0, abs=1e-6)
        # Node 2 has higher z → should be child_1 → ordinal 1.0
        assert ordinal[2].item() == pytest.approx(1.0, abs=1e-6)


class TestAxialExtentChild0Override:
    """axial_extent mode: child0_override / apical_flag pins the apical to ordinal 0
    (overriding the legacy most-negative-first-edge-uhat rule), while remaining
    SO(2)-invariant about uhat."""

    def test_override_forces_child0(self):
        uhat = th.tensor([0.0, 0.0, 1.0])
        offsets = th.tensor([
            [1.0, 0.0, 2.0],    # highest z (legacy would rank last)
            [0.5, 0.3, -1.0],   # lowest z (legacy child_0)
            [0.7, -0.8, 0.3],
        ])
        # Force index 0 (NOT the legacy pick) to be child_0.
        sorted_idx, fwd0, delta = _order_root_children_by_uhat(offsets, uhat, child0_override=0)
        assert sorted_idx[0].item() == 0, "forced child must be ordinal 0"
        assert delta[0].item() == pytest.approx(0.0, abs=1e-6)
        # fwd0 anchors on child 0's perp direction
        perp0 = offsets[0] - (offsets[0] @ uhat) * uhat
        perp0 = perp0 / perp0.norm()
        th.testing.assert_close(fwd0, perp0, atol=1e-5, rtol=1e-5)

    def test_override_rotation_invariance(self):
        uhat = th.tensor([0.0, 0.0, 1.0])
        offsets = th.tensor([
            [1.0, 0.5, 2.0],
            [-0.3, 1.0, -1.0],
            [0.7, -0.8, 0.3],
        ])
        sorted_ref, _, delta_ref = _order_root_children_by_uhat(offsets, uhat, child0_override=0)
        assert sorted_ref[0].item() == 0
        for angle in [0.5, 1.3, math.pi, 2.7]:
            R = _rotation_matrix_around_axis(uhat, angle)
            offsets_rot = (R @ offsets.T).T
            sorted_rot, _, delta_rot = _order_root_children_by_uhat(
                offsets_rot, uhat, child0_override=0,
            )
            assert th.equal(sorted_ref, sorted_rot)
            _angles_close_mod2pi(delta_ref, delta_rot)

    def test_override_degenerate_keeps_apical_first(self):
        """On-axis children (zero perp): forced apical must still be first."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        offsets = th.tensor([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.5],
        ])
        # Legacy would sort by z ascending -> (1, 2, 0); override keeps index 0 first.
        sorted_idx, fwd0, _ = _order_root_children_by_uhat(offsets, uhat, child0_override=0)
        assert sorted_idx[0].item() == 0, "forced apical must stay first in degenerate case"
        assert fwd0.norm().item() < 1e-6

    def test_apical_flag_pins_ordinal0(self):
        """apical_flag threaded through _compute_tree_directions -> compute_geo_order
        gives the flagged root child ordinal 0, overriding the legacy pick."""
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(4, uhat)  # nodes 1..4 at z=-0.5,-0.1,0.3,0.7
        legacy_c0 = 1  # lowest z
        flagged_node = 3  # a different root child
        flag = th.zeros(pos.size(0), dtype=th.bool)
        flag[flagged_node] = True

        dirs = _compute_tree_directions(pos, parent_idx, uhat, apical_flag=flag)
        ordinal, _ = compute_geo_order(pos, parent_idx, uhat, _directions=dirs)
        assert ordinal[flagged_node].item() == pytest.approx(0.0, abs=1e-6)

        # legacy path (no flag) instead assigns ordinal 0 to the lowest-z child
        ordinal_legacy, _ = compute_geo_order(pos, parent_idx, uhat)
        assert ordinal_legacy[legacy_c0].item() == pytest.approx(0.0, abs=1e-6)
        assert ordinal_legacy[flagged_node].item() != pytest.approx(0.0, abs=1e-6)

    def test_apical_flag_rotation_invariance(self):
        # k=5 (not k=4): evenly-spaced children put one exactly opposite the reference,
        # a measure-zero atan2 branch-cut degeneracy (see test_rotation_invariance_large_k).
        uhat = th.tensor([0.0, 0.0, 1.0])
        pos, parent_idx = _make_root_tree(5, uhat)
        flag = th.zeros(pos.size(0), dtype=th.bool)
        flag[3] = True
        dirs = _compute_tree_directions(pos, parent_idx, uhat, apical_flag=flag)
        ord_ref, dt_ref = compute_geo_order(pos, parent_idx, uhat, _directions=dirs)
        for angle in [0.7, math.pi / 3, math.pi, 4.1]:
            R = _rotation_matrix_around_axis(uhat, angle)
            pos_rot = (R @ pos.T).T
            dirs_rot = _compute_tree_directions(pos_rot, parent_idx, uhat, apical_flag=flag)
            ord_rot, dt_rot = compute_geo_order(pos_rot, parent_idx, uhat, _directions=dirs_rot)
            th.testing.assert_close(ord_ref, ord_rot, atol=1e-5, rtol=1e-5)
            _angles_close_mod2pi(dt_ref, dt_rot)


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
