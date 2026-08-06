import numpy as np
import pytest

from qmcpy.discrete_distribution import DummySampler
from qmcpy.util import ParameterError


PLACEHOLDER_ERROR = "construction placeholder"


def test_dummy_sampler_constructs_dimension_one():
    sampler = DummySampler(1)

    assert sampler.d == 1
    assert sampler.replications == 1
    assert sampler.no_replications
    assert sampler.mimics == "StdUniform"
    assert sampler.parameters == []


def test_dummy_sampler_constructs_larger_dimensions():
    sampler = DummySampler(3, seed=7)

    assert sampler.d == 3
    assert sampler.replications == 1
    assert sampler.no_replications
    assert np.array_equal(sampler.dvec, np.arange(3))


def test_dummy_sampler_constructs_larger_dimension_with_replications():
    sampler = DummySampler(4, replications=3, seed=7)

    assert sampler.d == 4
    assert sampler.replications == 3
    assert not sampler.no_replications
    assert np.array_equal(sampler.dvec, np.arange(4))


def test_dummy_sampler_direct_sampling_raises_placeholder_error():
    sampler = DummySampler(2)

    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler(8)


def test_dummy_sampler_replicated_direct_sampling_raises_placeholder_error():
    sampler = DummySampler(2, replications=3)

    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler(8)


def test_dummy_sampler_supported_calling_conventions_raise_placeholder_error():
    sampler = DummySampler(2)

    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler(n=4)
    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler(n_min=2, n_max=6)
    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler(n=2, n_min=6)


def test_dummy_sampler_nonzero_n_min_raises_placeholder_error():
    sampler = DummySampler(2)

    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler(n_min=5, n_max=9)


def test_dummy_sampler_rejects_return_binary():
    sampler = DummySampler(2)

    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler(4, return_binary=True)


def test_dummy_sampler_internal_gen_samples_raises_placeholder_error():
    sampler = DummySampler(2)

    with pytest.raises(ParameterError, match=PLACEHOLDER_ERROR):
        sampler._gen_samples(n_min=5, n_max=9, return_binary=False, warn=True)


def test_dummy_sampler_spawn_preserves_relevant_fields():
    sampler = DummySampler(2, replications=3, seed=11)

    spawned = sampler.spawn(s=2, dimensions=[1, 5])

    assert [spawn.d for spawn in spawned] == [1, 5]
    assert [spawn.replications for spawn in spawned] == [3, 3]
    assert all(isinstance(spawn, DummySampler) for spawn in spawned)


def test_dummy_sampler_spawn_without_explicit_replications():
    sampler = DummySampler(2, seed=11)

    spawned = sampler.spawn(s=1, dimensions=4)[0]

    assert spawned.d == 4
    assert spawned.replications == 1
    assert spawned.no_replications


def test_dummy_sampler_limits_are_enforced():
    with pytest.raises(ParameterError, match="dimension greater than dimension limit"):
        DummySampler(10_002)

    sampler = DummySampler(1)
    with pytest.raises(ParameterError, match="n_limit"):
        sampler(n_min=0, n_max=2**32 + 1)
