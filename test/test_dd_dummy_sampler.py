import unittest

import numpy as np

from qmcpy import DummySampler
from qmcpy.util import ParameterError


PLACEHOLDER_ERROR = "construction placeholder"


class TestDummySampler(unittest.TestCase):

    def test_dummy_sampler_constructs_dimension_one(self):
        sampler = DummySampler(1)

        self.assertEqual(sampler.d, 1)
        self.assertEqual(sampler.replications, 1)
        self.assertTrue(sampler.no_replications)
        self.assertEqual(sampler.mimics, "StdUniform")
        self.assertEqual(sampler.parameters, [])

    def test_dummy_sampler_constructs_larger_dimensions(self):
        sampler = DummySampler(3, seed=7)

        self.assertEqual(sampler.d, 3)
        self.assertEqual(sampler.replications, 1)
        self.assertTrue(sampler.no_replications)
        self.assertTrue(np.array_equal(sampler.dvec, np.arange(3)))

    def test_dummy_sampler_constructs_larger_dimension_with_replications(self):
        sampler = DummySampler(4, replications=3, seed=7)

        self.assertEqual(sampler.d, 4)
        self.assertEqual(sampler.replications, 3)
        self.assertFalse(sampler.no_replications)
        self.assertTrue(np.array_equal(sampler.dvec, np.arange(4)))

    def test_dummy_sampler_direct_sampling_raises_placeholder_error(self):
        sampler = DummySampler(2)

        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler(8)

    def test_dummy_sampler_replicated_direct_sampling_raises_placeholder_error(self):
        sampler = DummySampler(2, replications=3)

        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler(8)

    def test_dummy_sampler_supported_calling_conventions_raise_placeholder_error(self):
        sampler = DummySampler(2)

        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler(n=4)
        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler(n_min=2, n_max=6)
        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler(n=2, n_min=6)

    def test_dummy_sampler_nonzero_n_min_raises_placeholder_error(self):
        sampler = DummySampler(2)

        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler(n_min=5, n_max=9)

    def test_dummy_sampler_rejects_return_binary(self):
        sampler = DummySampler(2)

        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler(4, return_binary=True)

    def test_dummy_sampler_internal_gen_samples_raises_placeholder_error(self):
        sampler = DummySampler(2)

        with self.assertRaisesRegex(ParameterError, PLACEHOLDER_ERROR):
            sampler._gen_samples(n_min=5, n_max=9, return_binary=False, warn=True)

    def test_dummy_sampler_spawn_preserves_relevant_fields(self):
        sampler = DummySampler(2, replications=3, seed=11)

        spawned = sampler.spawn(s=2, dimensions=[1, 5])

        self.assertEqual([spawn.d for spawn in spawned], [1, 5])
        self.assertEqual([spawn.replications for spawn in spawned], [3, 3])
        self.assertTrue(all(isinstance(spawn, DummySampler) for spawn in spawned))

    def test_dummy_sampler_spawn_without_explicit_replications(self):
        sampler = DummySampler(2, seed=11)

        spawned = sampler.spawn(s=1, dimensions=4)[0]

        self.assertEqual(spawned.d, 4)
        self.assertEqual(spawned.replications, 1)
        self.assertTrue(spawned.no_replications)

    def test_dummy_sampler_limits_are_enforced(self):
        with self.assertRaisesRegex(ParameterError, "dimension greater than dimension limit"):
            DummySampler(10_002)

        sampler = DummySampler(1)
        with self.assertRaisesRegex(ParameterError, "n_limit"):
            sampler(n_min=0, n_max=2**32 + 1)


if __name__ == "__main__":
    unittest.main()
