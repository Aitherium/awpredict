"""StructuredAdapter: the treemap renderer — determinism, boxes, degrade rules."""

import numpy as np
import pytest
from awpredict.adapters.obs_adapter import GridAdapter, StructuredAdapter


def _tree(children):
    return {"type": "dir", "size": 1, "children": children}


def test_grid_identity_pass_through():
    g = GridAdapter()
    grid = g.token_grid([[0, 1], [2, 3]])
    assert grid.tolist() == [[0, 1], [2, 3]]
    assert g.cond(None) is None


def test_token_grid_is_deterministic():
    ad = StructuredAdapter(grid=32)
    t = _tree([{"type": "code", "size": 10}, {"type": "data", "size": 5},
               {"type": "code", "size": 3}, {"type": "media", "size": 1}])
    a = ad.token_grid(t)
    b = ad.token_grid(t)
    assert a is not None and np.array_equal(a, b)


def test_boxes_grid_is_byte_identical_to_token_grid():
    ad = StructuredAdapter(grid=32)
    t = _tree([{"type": "code", "size": 10}, {"type": "data", "size": 5},
               {"type": "code", "size": 3}, {"type": "media", "size": 1}])
    g1 = ad.token_grid(t)
    g2, boxes = ad.token_grid_with_boxes(t)
    assert np.array_equal(g1, g2)                     # byte-identical grids
    assert set(boxes) == {0, 1, 2, 3}                 # every child got a box


def test_collapsed_child_has_no_box():
    # On a 2x2 grid, six equal children cannot each hold a cell: the recursion
    # narrows spans to one cell and the left (bigger) child of a 1-cell span
    # collapses to zero width -> no click location.
    ad = StructuredAdapter(grid=2)
    t = _tree([{"type": "code", "size": 1} for _ in range(6)])
    _, boxes = ad.token_grid_with_boxes(t)
    assert len(boxes) < 6                              # some children collapsed
    assert set(boxes) <= set(range(6))


def test_unknown_type_drops_not_coerces():
    ad = StructuredAdapter(grid=16)
    t = _tree([{"type": "madeup", "size": 5}, {"type": "code", "size": 5}])
    grid = ad.token_grid(t)
    vals = set(np.unique(grid).tolist())
    assert ad._other in vals                          # id 15, never a wrong kind
    assert ad._id["code"] in vals                     # 'code' painted with its own id


def test_empty_and_shallow_observations():
    ad = StructuredAdapter(grid=16)
    # an EMPTY dir is a leaf of type 'dir': it paints one uniform block (the fork
    # behaves identically) and — having no children — yields no click boxes.
    empty = ad.token_grid({"type": "dir", "children": []})
    assert empty is not None and int(empty.sum()) == 16 * 16
    assert int(empty[0, 0]) == ad._id["dir"]
    _, boxes = ad.token_grid_with_boxes({"type": "dir", "children": []})
    assert boxes == {}


def test_recursive_leaves_flatten():
    ad = StructuredAdapter(grid=16)
    t = {"type": "dir", "children": [
        {"type": "code", "size": 5, "children": [
            {"type": "data", "size": 2}, {"type": "data", "size": 3}]},
        {"type": "media", "size": 5}]}
    grid = ad.token_grid(t)
    assert grid is not None and grid.sum() > 0


@pytest.mark.parametrize("grid_size", [16, 32, 64])
def test_area_proportional_to_size(grid_size):
    ad = StructuredAdapter(grid=grid_size)
    t = _tree([{"type": "code", "size": 9}, {"type": "media", "size": 1}])
    grid = ad.token_grid(t)
    total = grid.size
    code_share = int((grid == ad._id["code"]).sum()) / total
    assert 0.8 <= code_share <= 0.95                  # ~9/10 of the treemap is code
