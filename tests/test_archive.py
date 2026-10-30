import math
import random

from cathodepilot.archive import (
    Descriptor, MOMEGrid, ParetoSet, does_dominate, front_crowding_distance, value_to_bin,
)


def test_dominance_matches_cpp_definition():
    assert does_dominate([1, 1], [2, 2])
    assert does_dominate([1, 2], [1, 3])
    assert not does_dominate([1, 1], [1, 1])
    assert not does_dominate([1, 3], [2, 2])


def test_value_to_bin_edges_and_rejection():
    assert value_to_bin(0.0, 0.0, 5.0, 10) == 0
    assert value_to_bin(5.0, 0.0, 5.0, 10) == 9  # max maps to the last bin
    assert value_to_bin(2.49, 0.0, 5.0, 10) == 4
    assert value_to_bin(5.01, 0.0, 5.0, 10) is None  # rejected, not clamped
    assert value_to_bin(-0.1, 0.0, 5.0, 10) is None
    assert value_to_bin(math.nan, 0.0, 5.0, 10) is None
    assert value_to_bin(1.0, 5.0, 5.0, 10) is None  # min >= max


def test_pareto_set_rejects_dominated_and_displaces():
    ps = ParetoSet(capacity=10)
    assert ps.insert([2, 2], "a")
    assert not ps.insert([3, 3], "b")
    assert ps.insert([1, 1], "c")
    assert [e.payload for e in ps.entries] == ["c"]
    assert not ps.insert([1, math.inf], "d")
    assert not ps.insert([1, 1, 1], "e")  # dimensionality mismatch


def test_pareto_set_evicts_least_crowded():
    ps = ParetoSet(capacity=3)
    for i, pt in enumerate([[0, 4], [1, 3], [1.1, 2.9], [4, 0]]):
        assert ps.insert(pt, i)
    kept = sorted(e.payload for e in ps.entries)
    assert 0 in kept and 3 in kept  # extremes survive (infinite crowding distance)
    assert len(kept) == 3


def test_crowding_distance_extremes_infinite():
    d = front_crowding_distance([[0, 2], [1, 1], [2, 0]])
    assert d[0] == math.inf and d[2] == math.inf and d[1] == 2.0


def test_grid_insert_and_uniform_cell_selection():
    g = MOMEGrid((Descriptor("density", 0, 5), Descriptor("e_per_atom", -10, 0)), 10, 10)
    assert g.insert(3.5, -4.0, [-4.0, -3.9], "s1").accepted
    assert g.insert(2.3, -3.8, [-3.8, -3.5], "s2").accepted
    assert not g.insert(6.0, -4.0, [-5, -5], "s3").accepted
    assert g.coverage() == 0.02
    rng = random.Random(0)
    picks = {g.select_parent(rng) for _ in range(200)}
    assert picks == {"s1", "s2"}
