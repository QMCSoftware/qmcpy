from qmcpy import Kronecker
from qmcpy.util import ParameterError

import unittest
import warnings

import numpy as np
import numpy.testing as npt


class TestKroneckerConstruction(unittest.TestCase):
    """Unit tests for the additive recurrence defining the Kronecker sequence."""

    def test_matches_additive_recurrence_definition(self):
        gen_vec = 2 ** (np.arange(1, 4) / 4)
        dd = Kronecker(3, generating_vector=gen_vec, randomize=False)
        x = dd.gen_samples(6)
        expected = (np.arange(6)[:, None] * gen_vec[None, :]) % 1
        npt.assert_allclose(x, expected)

    def test_matches_definition_with_replications(self):
        dd = Kronecker(3, generating_vector="SUZUKI", replications=2, seed=7)
        x = dd.gen_samples(5)
        expected = (
            np.arange(5)[:, None] * dd.gen_vec[:, None, :] + dd.shift[:, None, :]
        ) % 1
        self.assertEqual(x.shape, (2, 5, 3))
        npt.assert_allclose(x, expected)

    def test_points_lie_in_unit_cube(self):
        for gen_vec_source in ["CBC", "RICHTMYER", "SUZUKI"]:
            x = Kronecker(4, generating_vector=gen_vec_source, seed=7).gen_samples(32)
            self.assertTrue((x >= 0).all() and (x < 1).all())

    def test_first_point_equals_shift(self):
        shift = [0.1, 0.2, 0.3]
        dd = Kronecker(3, generating_vector="SUZUKI", shift=shift)
        npt.assert_allclose(dd.gen_samples(1)[0], shift)

    def test_unrandomized_sequence_starts_at_origin(self):
        x = Kronecker(3, generating_vector="SUZUKI", randomize=False).gen_samples(1)
        npt.assert_allclose(x, np.zeros((1, 3)))

    def test_n_min_n_max_slices_the_sequence(self):
        dd = Kronecker(3, seed=7)
        npt.assert_allclose(dd.gen_samples(n_min=2, n_max=5), dd.gen_samples(5)[2:])

    def test_subset_dimensions_select_generating_vector_components(self):
        gen_vec = 2 ** (np.arange(1, 4) / 4)
        full = Kronecker(3, generating_vector=gen_vec, randomize=False).gen_samples(4)
        subset = Kronecker(
            [0, 2], generating_vector=gen_vec, randomize=False
        ).gen_samples(4)
        self.assertEqual(subset.shape, (4, 2))
        npt.assert_allclose(subset, full[:, [0, 2]])


class TestKroneckerGeneratingVector(unittest.TestCase):
    """Unit tests for generating vector selection and its reported source."""

    def test_gen_vec_source_labels(self):
        self.assertEqual(Kronecker(3, generating_vector="CBC").gen_vec_source, "CBC")
        self.assertEqual(
            Kronecker(3, generating_vector="RICHTMYER").gen_vec_source, "RICHTMYER"
        )
        self.assertEqual(
            Kronecker(3, generating_vector="SUZUKI").gen_vec_source, "SUZUKI"
        )
        self.assertEqual(
            Kronecker(3, generating_vector=np.array([0.1, 0.2, 0.3])).gen_vec_source,
            "CUSTOM",
        )

    def test_generating_vector_name_is_case_insensitive(self):
        self.assertEqual(Kronecker(3, generating_vector="cbc").gen_vec_source, "CBC")
        self.assertEqual(
            Kronecker(3, generating_vector="richtmyer").gen_vec_source, "RICHTMYER"
        )
        self.assertEqual(
            Kronecker(3, generating_vector="suzuki").gen_vec_source, "SUZUKI"
        )

    def test_suzuki_generating_vector_formula(self):
        d = 5
        dd = Kronecker(d, generating_vector="SUZUKI", randomize=False)
        npt.assert_allclose(dd.gen_vec[0], 2 ** (np.arange(1, d + 1) / (d + 1)))

    def test_richtmyer_generating_vector_formula(self):
        primes = np.array([2, 3, 5, 7, 11])
        dd = Kronecker(5, generating_vector="RICHTMYER", randomize=False)
        npt.assert_allclose(dd.gen_vec[0], np.sqrt(primes) % 1)

    def test_cbc_falls_back_to_richtmyer_beyond_supported_dimension(self):
        with self.assertWarns(RuntimeWarning):
            dd = Kronecker(15, generating_vector="CBC", seed=7)
        self.assertEqual(dd.gen_vec_source, "RICHTMYER")
        self.assertEqual(dd.gen_samples(4).shape, (4, 15))

    def test_cbc_fallback_warning_suppressed_by_warn_false(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dd = Kronecker(15, generating_vector="CBC", seed=7, warn=False)
        self.assertEqual(dd.gen_vec_source, "RICHTMYER")

    def test_generating_vector_above_two_dimensions_raises(self):
        with self.assertRaises(ParameterError):
            Kronecker(3, generating_vector=np.ones((2, 2, 3)))


class TestKroneckerRandomization(unittest.TestCase):
    """Unit tests for randomization aliases and shift validation."""

    def test_randomize_aliases_resolve_to_canonical_values(self):
        for alias in ["TRUE", "true", "SHIFT"]:
            self.assertEqual(Kronecker(2, randomize=alias, seed=7).randomize, "SHIFT")
        for alias in ["FALSE", "NONE", "NO", False]:
            self.assertEqual(Kronecker(2, randomize=alias, seed=7).randomize, "FALSE")

    def test_unrandomized_shift_is_zero(self):
        dd = Kronecker(3, randomize=False)
        npt.assert_allclose(dd.shift, np.zeros((1, 3)))

    def test_invalid_randomize_raises(self):
        with self.assertRaises(AssertionError):
            Kronecker(2, randomize="OWEN")

    def test_shift_requires_randomize_shift(self):
        with self.assertRaises(AssertionError):
            Kronecker(3, randomize=False, shift=[0.1, 0.2, 0.3])

    def test_same_seed_reproduces_shift_and_points(self):
        npt.assert_allclose(
            Kronecker(3, seed=7).gen_samples(8), Kronecker(3, seed=7).gen_samples(8)
        )

    def test_different_seeds_give_different_shifts(self):
        self.assertFalse(
            np.allclose(Kronecker(3, seed=7).shift, Kronecker(3, seed=8).shift)
        )

    def test_replications_use_independent_shifts(self):
        dd = Kronecker(3, replications=2, seed=7)
        self.assertEqual(dd.shift.shape, (2, 3))
        self.assertFalse(np.allclose(dd.shift[0], dd.shift[1]))

    def test_per_replication_shift_is_respected(self):
        shift = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        dd = Kronecker(3, generating_vector="SUZUKI", replications=2, shift=shift)
        npt.assert_allclose(dd.gen_samples(1)[:, 0, :], shift)


class TestKroneckerUnsupportedOptions(unittest.TestCase):
    """Unit tests for options the Kronecker sequence does not support."""

    def test_return_binary_raises(self):
        with self.assertRaises(ParameterError):
            Kronecker(3, seed=7).gen_samples(4, return_binary=True)


class TestKroneckerSpawn(unittest.TestCase):
    """Unit tests for spawning independent Kronecker samplers."""

    def test_spawn_preserves_configuration(self):
        dd = Kronecker(3, generating_vector="SUZUKI", seed=7)
        spawns = dd.spawn(s=2, dimensions=[2, 2])
        self.assertEqual(len(spawns), 2)
        for spawn in spawns:
            self.assertIsInstance(spawn, Kronecker)
            self.assertEqual(spawn.gen_vec_source, "SUZUKI")
            self.assertEqual(spawn.randomize, "SHIFT")
            self.assertEqual(spawn.gen_samples(4).shape, (4, 2))

    def test_spawns_are_independently_randomized(self):
        spawns = Kronecker(3, seed=7).spawn(s=2, dimensions=[3, 3])
        self.assertFalse(np.allclose(spawns[0].shift, spawns[1].shift))

    def test_spawn_requires_unspecified_shift(self):
        dd = Kronecker(3, generating_vector="SUZUKI", shift=[0.1, 0.2, 0.3])
        with self.assertRaises(AssertionError):
            dd.spawn(s=1, dimensions=[3])


class TestKroneckerDiscrepancy(unittest.TestCase):
    """Unit tests for the periodic discrepancy helpers."""

    @staticmethod
    def _default_k_tilde(x, gamma):
        return np.prod(1 + (x * (x - 1) + 1 / 6) * gamma, axis=-1)

    def _pairwise_definition(self, dd, n):
        """Squared periodic discrepancy from its definition over pairwise differences."""
        x = dd.gen_samples(n)
        gamma = np.ones(dd.d)
        k = self._default_k_tilde((x[:, None, :] - x[None, :, :]) % 1, gamma)
        return np.array([k[:m, :m].sum() / m**2 - 1 for m in range(1, n + 1)])

    def test_unshifted_squared_discrepancy_matches_pairwise_definition(self):
        n = 32
        for d in [1, 2, 3, 5]:
            dd = Kronecker(d, seed=7, randomize=False)
            npt.assert_allclose(
                dd._square_periodic_discrepancies(
                    n, (self._default_k_tilde, 1), np.ones(dd.d)
                ),
                self._pairwise_definition(dd, n),
                atol=1e-12,
            )

    @unittest.expectedFailure
    def test_shifted_squared_discrepancy_matches_pairwise_definition(self):
        # Known bug: the implementation keeps the random shift, the definition does not.
        n, d = 32, 3
        dd = Kronecker(d, seed=7)
        npt.assert_allclose(
            dd._square_periodic_discrepancies(
                n, (self._default_k_tilde, 1), np.ones(dd.d)
            ),
            self._pairwise_definition(dd, n),
            atol=1e-12,
        )

    def test_periodic_discrepancy_is_root_of_squared_discrepancy(self):
        dd = Kronecker(2, seed=7)
        n = 8
        gamma = np.ones(dd.d)
        k_tilde = (self._default_k_tilde, 1)
        squared = dd._square_periodic_discrepancies(n, k_tilde, gamma)
        npt.assert_allclose(dd.periodic_discrepancy(n), np.sqrt(squared))

    def test_periodic_discrepancy_shape(self):
        n = 16
        discrep = Kronecker(3, seed=7, randomize=False).periodic_discrepancy(n)
        self.assertEqual(discrep.shape[-1], n)

    def test_unshifted_squared_discrepancy_is_nonnegative(self):
        for d in [1, 2, 3, 5]:
            dd = Kronecker(d, seed=7, randomize=False)
            squared = dd._square_periodic_discrepancies(
                32, (self._default_k_tilde, 1), np.ones(dd.d)
            )
            self.assertTrue((squared >= 0).all())

    @unittest.expectedFailure
    def test_squared_discrepancy_is_nonnegative_when_shifted(self):
        # Known bug: the kernel is evaluated at x_{|a-b|}, which retains the random
        # shift, so the quadratic form loses positive semi-definiteness.
        for d in [1, 3, 5]:
            dd = Kronecker(d, seed=7)
            squared = dd._square_periodic_discrepancies(
                32, (self._default_k_tilde, 1), np.ones(dd.d)
            )
            self.assertTrue((squared >= 0).all())

    @unittest.expectedFailure
    def test_squared_discrepancy_is_invariant_to_the_random_shift(self):
        # Known bug: x_a - x_b = (a-b)*alpha mod 1 cancels the shift, so the
        # discrepancy must not depend on the seed, but it currently does.
        n, d = 32, 3
        gamma = np.ones(d)
        k_tilde = (self._default_k_tilde, 1)
        unshifted = Kronecker(d, seed=7, randomize=False)
        expected = unshifted._square_periodic_discrepancies(n, k_tilde, gamma)
        for seed in [1, 7, 42]:
            shifted = Kronecker(d, seed=seed)
            npt.assert_allclose(
                shifted._square_periodic_discrepancies(n, k_tilde, gamma), expected
            )

    @unittest.expectedFailure
    def test_periodic_discrepancy_is_finite_when_shifted(self):
        # Known bug: negative squared discrepancies make the square root NaN.
        discrep = Kronecker(3, seed=7).periodic_discrepancy(32)
        self.assertTrue(np.isfinite(discrep).all())

    def test_wssd_discrepancy_is_weighted_sum_of_squared_discrepancies(self):
        dd = Kronecker(2, seed=7)
        n = 8
        weights = np.linspace(0.5, 2.0, n)
        npt.assert_allclose(
            dd.wssd_discrepancy(n, weights),
            np.sum(weights * dd.periodic_discrepancy(n) ** 2, axis=-1),
        )

    def test_squared_discrepancy_matches_direct_double_sum(self):
        # Refactor guard only: this expectation is an algebraic restatement of the
        # implementation's cumulative sums, so it cannot detect a wrong formula.
        # Note: passes today because it bakes in the buggy shifted-kernel formula
        # from #612; it will need rewriting once that fix lands (see PR #556/#633).
        dd = Kronecker(2, seed=7)
        n = 8
        gamma = np.ones(dd.d)
        terms = self._default_k_tilde(dd.gen_samples(n), gamma)
        expected = np.array(
            [
                sum(terms[abs(a - b)] for a in range(m) for b in range(m)) / m**2 - 1
                for m in range(1, n + 1)
            ]
        )
        npt.assert_allclose(dd._square_periodic_discrepancies(n, (self._default_k_tilde, 1), gamma), expected)


if __name__ == "__main__":
    unittest.main()
