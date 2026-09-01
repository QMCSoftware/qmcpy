from qmcpy import (
    DigitalNetB2,
    Halton,
    Hammersley,
    IIDStdUniform,
    KorobovLattice,
    LatinHypercube,
    Lattice,
)

from qmcpy.util import ParameterError, ParameterWarning
import qmctoolscl
import os
import unittest
import numpy as np
import numpy.testing as npt
import tempfile
import warnings

class TestDiscreteDistribution(unittest.TestCase):

    def test_size_unsigned_long(self):
        if os.name == "nt":
            self.assertEqual(qmctoolscl.util.get_unsigned_long_size_c(), 4)
        else:
            self.assertEqual(qmctoolscl.util.get_unsigned_long_size_c(), 8)

    def test_size_unsigned_long_long(self):
        self.assertEqual(qmctoolscl.util.get_unsigned_long_long_size_c(), 8)

    def test_abstract_methods(self):
        for d in [3, [1, 3, 5]]:
            dds = [
                Lattice(d, order="natural", seed=7),
                Lattice(d, order="linear", seed=7),
                DigitalNetB2(d, randomize="LMS_DS", order="RADICAL INVERSE", seed=7),
                DigitalNetB2(d, randomize="DS", seed=7),
                DigitalNetB2(d, order="GRAY", seed=7),
                Halton(d, randomize="QRNG", seed=7),
                Halton(d, randomize="Owen", seed=7),
                LatinHypercube(d, replications=None, seed=7),
                LatinHypercube(d, replications=None, seed=7, randomize=False),
                KorobovLattice(d, replications=None, seed=7),
            ]
            for dd in dds:
                for _dd in [dd] + dd.spawn(1):
                    x = _dd.gen_samples(4, warn=False)
                    if _dd.mimics == "StdUniform":
                        self.assertTrue((x > 0).all() and (x < 1).all())
                    pdf = _dd.pdf(_dd.gen_samples(4, warn=False))
                    self.assertEqual(pdf.shape, (4,))
                    self.assertEqual(x.shape, (4, 3))
                    self.assertEqual(x.dtype, np.float64)
                    s = str(_dd)

    def test_spawn(self):
        d = 3
        for dd in [
            IIDStdUniform(d, seed=7),
            Lattice(d, seed=7),
            DigitalNetB2(d, seed=7),
            Halton(d, seed=7, warn=False),
            LatinHypercube(d, replications=None, seed=7),
            LatinHypercube(d, replications=None, seed=7, randomize=False),
            Hammersley(d, seed=7, warn=False),
            KorobovLattice(d, replications=None, seed=7),
        ]:
            s = 3
            for spawn_dim in [4, [1, 4, 6]]:
                spawns = dd.spawn(s=s, dimensions=spawn_dim)
                self.assertEqual(len(spawns), s)
                self.assertTrue(all(type(spawn) == type(dd) for spawn in spawns))
                self.assertTrue(
                    (np.array([spawn.d for spawn in spawns]) == spawn_dim).all()
                )

class TestKorobovLattice(unittest.TestCase):
    """Unit tests for KorobovLattice discrete distribution."""

    def test_gen_samples_shape(self):
        d1 = KorobovLattice(dimension=3, replications=None, seed=7)
        x1 = d1.gen_samples(8, warn=False)
        self.assertEqual(x1.shape, (8, 3))

        d2 = KorobovLattice(dimension=2, replications=5, seed=7)
        x2 = d2.gen_samples(8, warn=False)
        self.assertEqual(x2.shape, (5, 8, 2))

    def test_values_in_unit_cube(self):
        distribution = KorobovLattice(dimension=3, replications=4, seed=11)
        x = distribution.gen_samples(16, warn=False)
        self.assertTrue((x >= 0).all() and (x < 1).all())

    def test_unrandomized_values_seed_7(self):
        # Check the result using precomputed samples
        true_sample = np.array([
            [0.0,   0.0  ],
            [0.125, 0.375],
            [0.25,  0.75 ],
            [0.375, 0.125],
            [0.5,   0.5  ],
            [0.625, 0.875],
            [0.75,  0.25 ],
            [0.875, 0.625],
        ])
        distribution = KorobovLattice(dimension=2, randomize="FALSE", seed=7)
        x = distribution.gen_samples(8, warn=False)
        self.assertTrue((x == true_sample).all())

    def test_rank1_lattice_structure(self):
        # general invariant of a rank-1 lattice: x_{k+1} - x_k = z/n (mod 1)
        # is CONSTANT for every k -- checked without depending on the internal values of a
        distribution = KorobovLattice(dimension=4, randomize="FALSE", seed=7)
        x = distribution.gen_samples(16, warn=False)
        diffs = (x[1:] - x[:-1]) % 1.0
        self.assertTrue(np.allclose(diffs, diffs[0]))

    def test_first_point_is_origin_unrandomized(self):
        distribution = KorobovLattice(dimension=3, randomize="FALSE", seed=7)
        x = distribution.gen_samples(8, warn=False)
        self.assertTrue((x[0] == 0).all())

    def test_n_not_tabulated_raises(self):
        distribution = KorobovLattice(dimension=3, seed=7)
        with self.assertRaises(ParameterError):
            distribution.gen_samples(5, warn=False)   # 5 n'est pas dans la table

    def test_n_min_nonzero_raises(self):
        distribution = KorobovLattice(dimension=3, seed=7)
        with self.assertRaises(ParameterError):
            distribution.gen_samples(n_min=4, n_max=8, warn=False)

    def test_return_binary_raises(self):
        distribution = KorobovLattice(dimension=2, seed=7)
        with self.assertRaises(ParameterError):
            distribution.gen_samples(8, return_binary=True, warn=False)

    def test_warns_by_default_without_randomization(self):
        distribution = KorobovLattice(dimension=2, randomize="FALSE", seed=7)
        with self.assertWarns(ParameterWarning):
            distribution.gen_samples(8)

    def test_no_warning_when_disabled(self):
        distribution = KorobovLattice(dimension=2, randomize="FALSE", seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            distribution.gen_samples(8, warn=False)

    def test_reproducibility_same_seed(self):
        d1 = KorobovLattice(dimension=3, seed=123)
        d2 = KorobovLattice(dimension=3, seed=123)
        x1 = d1.gen_samples(8, warn=False)
        x2 = d2.gen_samples(8, warn=False)
        self.assertTrue((x1 == x2).all())

    def test_spawn_dimension(self):
        d = KorobovLattice(dimension=3, seed=7)
        spawns = d.spawn(s=2, dimensions=[2, 4])
        self.assertEqual(len(spawns), 2)
        self.assertTrue(all(isinstance(s, KorobovLattice) for s in spawns))
        self.assertEqual(spawns[0].d, 2)
        self.assertEqual(spawns[1].d, 4)





class TestLattice(unittest.TestCase):
    """Unit tests for Lattice DiscreteDistribution."""

    def test_gen_samples(self):
        for order in ["natural", "gray"]:
            lattice0123 = Lattice(dimension=4, order=order, randomize=False)
            x0123 = lattice0123.gen_samples(8, warn=False)
            lattice13 = Lattice(dimension=[1, 3], order=order, randomize=False)
            x13 = lattice13.gen_samples(n_min=2, n_max=8)
            self.assertTrue((x0123[2:8, [1, 3]] == x13).all())

    def test_linear_order(self):
        true_sample = np.array(
            [
                [1.0 / 8, 3.0 / 8, 3.0 / 8, 7.0 / 8],
                [3.0 / 8, 1.0 / 8, 1.0 / 8, 5.0 / 8],
                [5.0 / 8, 7.0 / 8, 7.0 / 8, 3.0 / 8],
                [7.0 / 8, 5.0 / 8, 5.0 / 8, 1.0 / 8],
            ]
        )
        distribution = Lattice(dimension=4, randomize=False, order="linear")
        self.assertTrue(
            (
                distribution.gen_samples(n_min=4, n_max=8, warn=False) == true_sample
            ).all()
        )

    def test_natural_order(self):
        true_sample = np.array(
            [
                [1.0 / 8, 3.0 / 8, 3.0 / 8, 7.0 / 8],
                [5.0 / 8, 7.0 / 8, 7.0 / 8, 3.0 / 8],
                [3.0 / 8, 1.0 / 8, 1.0 / 8, 5.0 / 8],
                [7.0 / 8, 5.0 / 8, 5.0 / 8, 1.0 / 8],
            ]
        )
        distribution = Lattice(dimension=4, randomize=False, order="natural")
        self.assertTrue(
            (
                distribution.gen_samples(n_min=4, n_max=8, warn=False) == true_sample
            ).all()
        )

    def test_gray_order(self):
        true_sample = np.array(
            [
                [3.0 / 8, 1.0 / 8, 1.0 / 8, 5.0 / 8],
                [7.0 / 8, 5.0 / 8, 5.0 / 8, 1.0 / 8],
                [5.0 / 8, 7.0 / 8, 7.0 / 8, 3.0 / 8],
                [1.0 / 8, 3.0 / 8, 3.0 / 8, 7.0 / 8],
            ]
        )
        distribution = Lattice(dimension=4, randomize=False, order="gray")
        self.assertTrue(
            (
                distribution.gen_samples(n_min=4, n_max=8, warn=False) == true_sample
            ).all()
        )

    def test_integer_generating_vectors(self):
        distribution = Lattice(
            dimension=4, generating_vector=26, randomize=False, seed=136
        )
        true_sample = np.array(
            [
                [0.125, 0.875, 0.625, 0.375],
                [0.625, 0.375, 0.125, 0.875],
                [0.375, 0.625, 0.875, 0.125],
                [0.875, 0.125, 0.375, 0.625],
            ]
        )
        self.assertTrue(
            (
                distribution.gen_samples(n_min=4, n_max=8, warn=False) == true_sample
            ).all()
        )


class TestDigitalNetB2(unittest.TestCase):
    """Unit tests for DigitalNetB2 DiscreteDistribution.

    Goals:
      - Exercise key branches without relying on doctests/booktests.
      - Keep tests deterministic and platform-stable.
      - Avoid network access (no GitHub/LDData fetches in unit tests).
    """

    def test_basic_default_call_is_deterministic_and_in_unit_cube(self):
        dnb2 = DigitalNetB2(
            2, seed=7
        )  # default randomize="LMS DS", order="RADICAL INVERSE"
        x1 = dnb2(4, warn=False)
        x2 = DigitalNetB2(2, seed=7)(4, warn=False)

        self.assertEqual(x1.shape, (4, 2))
        self.assertTrue(np.isfinite(x1).all())
        self.assertTrue(((x1 >= 0) & (x1 < 1)).all())

        # Determinism given same params + seed (contract)
        npt.assert_array_equal(x1, x2)

    def test_replications_shape_and_determinism(self):
        dnb2 = DigitalNetB2(dimension=3, seed=7, replications=2)
        x = dnb2(4, warn=False)

        self.assertEqual(x.shape, (2, 4, 3))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(((x >= 0) & (x < 1)).all())

        # Determinism for the same seed/params
        x2 = DigitalNetB2(dimension=3, seed=7, replications=2)(4, warn=False)
        npt.assert_array_equal(x, x2)

    def test_ordering_gray_vs_radical_inverse_canonical_small_case(self):
        # These are tiny, canonical “ordering sanity checks” (stable and intentional).
        # We keep them small to avoid brittle large golden arrays.
        dnb2_gray = DigitalNetB2(dimension=2, randomize=False, order="GRAY", seed=7)
        x_gray = dnb2_gray.gen_samples(n_min=2, n_max=4, warn=False)
        x_gray_true = np.array([[0.75, 0.25], [0.25, 0.75]])
        npt.assert_allclose(x_gray, x_gray_true, rtol=0, atol=0)

        dnb2_nat = DigitalNetB2(
            dimension=2, randomize=False, order="RADICAL INVERSE", seed=7
        )
        x_nat = dnb2_nat.gen_samples(n_min=2, n_max=4, warn=False)
        x_nat_true = np.array([[0.25, 0.75], [0.75, 0.25]])
        npt.assert_allclose(x_nat, x_nat_true, rtol=0, atol=0)

    def test_radical_inverse_requires_powers_of_two_bounds(self):
        dnb2 = DigitalNetB2(
            dimension=2, randomize=False, order="RADICAL INVERSE", seed=7
        )
        with self.assertRaises(ParameterError):
            _ = dnb2.gen_samples(n_min=3, n_max=5, warn=False)  # not powers of 2

    def test_deprecated_graycode_emits_warning_and_maps_order(self):
        # graycode=True should map to GRAY and warn.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dnb2 = DigitalNetB2(dimension=2, randomize=False, graycode=True, seed=7)
            self.assertEqual(dnb2.order, "GRAY")
            self.assertTrue(
                any("graycode argument deprecated" in str(x.message) for x in w)
            )

        # graycode=False should map to RADICAL INVERSE and warn.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dnb2 = DigitalNetB2(dimension=2, randomize=False, graycode=False, seed=7)
            self.assertEqual(dnb2.order, "RADICAL INVERSE")
            self.assertTrue(
                any("graycode argument deprecated" in str(x.message) for x in w)
            )

    def test_deprecated_t_lms_emits_warning_and_sets_t(self):
        # IMPORTANT: for default joe_kuo matrices, _t_curr is 32, so t must be >= 32.
        # Use a safe value (63 matches docstring examples).
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dnb2 = DigitalNetB2(dimension=2, seed=7, t_lms=63)
            self.assertEqual(dnb2.t, 63)
            self.assertTrue(
                any("t_lms argument deprecated" in str(x.message) for x in w)
            )

    def test_deprecated_t_max_emits_warning_only(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = DigitalNetB2(dimension=2, seed=7, t_max=999)
            self.assertTrue(any("t_max is deprecated" in str(x.message) for x in w))

    def test_order_normalization_inputs(self):
        # "GRAY CODE" should normalize to "GRAY"
        dnb2 = DigitalNetB2(dimension=2, randomize=False, order="GRAY CODE", seed=7)
        self.assertEqual(dnb2.order, "GRAY")

        # "NATURAL" should normalize to "RADICAL INVERSE"
        dnb2 = DigitalNetB2(dimension=2, randomize=False, order="NATURAL", seed=7)
        self.assertEqual(dnb2.order, "RADICAL INVERSE")

    def test_randomize_mode_coverage_smoke(self):
        # Hit the major randomize branches with small n; assert contracts not exact arrays.
        modes = ["FALSE", "DS", "LMS", "LMS DS", "NUS"]
        for mode in modes:
            dnb2 = DigitalNetB2(dimension=3, seed=7, randomize=mode)
            x = dnb2(4, warn=False)
            self.assertEqual(x.shape, (4, 3))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(((x >= 0) & (x < 1)).all())

    def test_randomize_nus_alpha2_branch_smoke(self):
        # Exercise alpha>1 interlacing + NUS branch (contract asserts only).
        dnb2 = DigitalNetB2(dimension=3, seed=7, randomize="NUS", alpha=2)
        x = dnb2(4, warn=False)
        self.assertEqual(x.shape, (4, 3))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(((x >= 0) & (x < 1)).all())

    def test_warns_when_first_point_origin_without_randomization(self):
        # _gen_samples warns when n_min==0 and randomize in ["FALSE","LMS"] and warn=True
        for mode in ["FALSE", "LMS"]:
            dnb2 = DigitalNetB2(dimension=2, randomize=mode, seed=7)
            with self.assertWarns(Warning):
                _ = dnb2.gen_samples(n_min=0, n_max=2, warn=True)

    def test_generating_matrices_from_local_txt_file_no_network(self):
        # Cover the `isinstance(generating_matrices, str)` .txt parsing path without network.
        #
        # We generate a tiny valid base-2 dnet file on the fly (deterministic, local).
        # Format expected by code:
        #   line0: base (2)
        #   line1: d_limit
        #   line2: n_limit
        #   line3: _t_curr
        #   remaining: rows of ints (d_limit rows, m_max columns)
        #
        # Keep it minimal: d_limit=2, m_max=4 => n_limit=2^4=16, _t_curr=4.
        contents = "\n".join(
            [
                "2",
                "2",
                "16",
                "4",
                "8 8 8 8",
                "9 9 9 9",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tiny_dnet.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(contents)

            dnb2 = DigitalNetB2(
                dimension=2,
                randomize=False,
                generating_matrices=path,
                t=4,  # must satisfy _t_curr <= t <= 64, and here _t_curr=4
                seed=7,
            )
            x = dnb2(8, warn=False)  # 8 points, still small

            self.assertEqual(x.shape, (8, 2))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(((x >= 0) & (x < 1)).all())

    def test_generating_matrices_numpy_array_branch(self):
        # Cover the `isinstance(generating_matrices, np.ndarray)` branch.
        # Provide small positive ints and explicitly set msb to satisfy assertions.
        gen_mats = np.array(
            [
                [1, 2, 3, 1],
                [2, 1, 3, 2],
            ],
            dtype=np.uint64,
        )  # shape (d, m_max)
        # gen_mat_max=3 => _t_curr=2; set t>=2
        dnb2 = DigitalNetB2(
            dimension=2,
            randomize=False,
            generating_matrices=gen_mats,
            msb=False,  # avoid calling conversion routine; still valid branch coverage
            t=2,
            seed=7,
        )
        x = dnb2(4, warn=False)
        self.assertEqual(x.shape, (4, 2))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(((x >= 0) & (x < 1)).all())

    def test_repeated_sampling(self):
        for order in ["GRAY","NATURAL"]:
            for randomize in ["FALSE","LMS DS","LMS","DS","OWEN"]:
                for alpha in [1,2]:
                    replications = 3 if randomize!="FALSE" else 1
                    dnb2 = DigitalNetB2(dimension=5,replications=replications,randomize=randomize,order=order,alpha=alpha)
                    x_full = dnb2(16,warn=False)
                    self.assertEqual(x_full.shape,(replications, 16, 5))
                    self.assertTrue((x_full[:,:4,:]==dnb2(0,4,warn=False)).all())
                    self.assertTrue((x_full[:,4:8,:]==dnb2(4,8)).all())
                    self.assertTrue((x_full[:,8:16,:]==dnb2(8,16)).all())
                    self.assertTrue((x_full[:,4:16,:]==dnb2(4,16)).all())



class TestHammersley(unittest.TestCase):
    """Unit tests for Hammersley discrete distribution."""

    def test_gen_samples_shape(self):
        distribution = Hammersley(dimension=3, seed=7)
        x = distribution.gen_samples(8, warn=False)
        self.assertEqual(x.shape, (8, 3))

    def test_dimension_one(self):
        distribution = Hammersley(dimension=1, seed=7)
        x = distribution.gen_samples(4, warn=False)
        true_sample = np.array([[0.0], [0.25], [0.5], [0.75]])
        self.assertTrue((x == true_sample).all())

    def test_values_in_unit_cube(self):
        distribution = Hammersley(dimension=4, seed=7)
        x = distribution.gen_samples(16, warn=False)
        self.assertTrue((x >= 0).all() and (x < 1).all())

    def test_first_point_is_origin(self):
        distribution = Hammersley(dimension=3, seed=7)
        x = distribution.gen_samples(8, warn=False)
        self.assertTrue((x[0] == 0).all())

    def test_matches_classical_definition(self):
        # t_i = (i/n, phi_p1(i), ..., phi_p_{d-1}(i)) --
        def van_der_corput(i, base):
            f, r, idx = 1.0, 0.0, i
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            return r

        primes = [2, 3, 5]
        n, d = 8, 4
        expected = np.array([
            [i / n] + [van_der_corput(i, p) for p in primes]
            for i in range(n)
        ])
        distribution = Hammersley(dimension=d, seed=7)
        x = distribution.gen_samples(n, warn=False)
        self.assertTrue(np.allclose(x, expected))

    def test_array_dimension_raises(self):
        with self.assertRaises(ParameterError):
            Hammersley(dimension=[1, 3, 5], seed=7)

    def test_dimension_less_than_one_raises(self):
        with self.assertRaises(ParameterError):
            Hammersley(dimension=0, seed=7)

    def test_return_binary_raises(self):
        distribution = Hammersley(dimension=2, seed=7)
        with self.assertRaises(ParameterError):
            distribution.gen_samples(4, return_binary=True, warn=False)

    def test_n_min_nonzero_raises(self):
        distribution = Hammersley(dimension=2, seed=7)
        with self.assertRaises(ParameterError):
            distribution.gen_samples(n_min=4, n_max=8, warn=False)

    def test_warns_by_default(self):
        distribution = Hammersley(dimension=2, seed=7)
        with self.assertWarns(ParameterWarning):
            distribution.gen_samples(8)

    def test_no_warning_when_disabled(self):
        distribution = Hammersley(dimension=2, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            distribution.gen_samples(8, warn=False)

    def test_spawn_dimension(self):
        d = Hammersley(dimension=3, seed=7)
        spawns = d.spawn(s=2, dimensions=[2, 4])
        self.assertEqual(len(spawns), 2)
        self.assertTrue(all(isinstance(s, Hammersley) for s in spawns))
        self.assertEqual(spawns[0].d, 2)
        self.assertEqual(spawns[1].d, 4)

class TestHalton(unittest.TestCase):
    """Unit test for Halton DiscreteDistribution."""

    def test_gen_samples(self):
        h123 = Halton(dimension=4, randomize=False)
        x0123 = h123.gen_samples(8, warn=False)
        h13 = Halton(dimension=[1, 3], randomize=False)
        x13 = h13.gen_samples(n_min=5, n_max=7, warn=False)
        self.assertTrue((x0123[5:7, [1, 3]] == x13).all())

    def test_unrandomized(self):
        x_ur = Halton(dimension=2, randomize=False, warn=False).gen_samples(4, warn=False)
        x_true = np.array(
            [[0, 0], [1.0 / 2, 1.0 / 3], [1.0 / 4, 2.0 / 3], [3.0 / 4, 1.0 / 9]]
        )
        self.assertTrue((x_ur == x_true).all())


class TestLatinHypercube(unittest.TestCase):
    """Unit tests for LatinHypercube DiscreteDistribution."""

    def test_gen_samples_shape(self):
        # replications=None -> squeeze to 2D
        d1 = LatinHypercube(dimension=4, replications=None, seed=7)
        x1 = d1.gen_samples(4, warn=False)
        self.assertEqual(x1.shape, (4, 4))

        # replications=k -> stays 3D
        d2 = LatinHypercube(dimension=2, replications=5, seed=7)
        x2 = d2.gen_samples(3, warn=False)
        self.assertEqual(x2.shape, (5, 3, 2))

    def test_gen_samples_shape_not_randomized(self):
        # Same shape checks as above, but with randomize=False -- this is the
        # exact case that used to be broken: the centered branch previously
        # hardcoded a leading axis of size 1 regardless of `replications`,
        # so replications=5 silently returned shape (1, 3, 2) instead of
        # (5, 3, 2). Regression test for that fix.
        d1 = LatinHypercube(dimension=4, replications=None, seed=7, randomize="False")
        x1 = d1.gen_samples(4, warn=False)
        self.assertEqual(x1.shape, (4, 4))

        d2 = LatinHypercube(dimension=2, replications=5, seed=7, randomize="False")
        x2 = d2.gen_samples(3, warn=False)
        self.assertEqual(x2.shape, (5, 3, 2))

    def test_values_seed_7(self):
        # Regression/reproducibility test: exact values for a fixed seed.
        # SFC64 with a fixed SeedSequence is bit-reproducible, so these
        # values should not change unless the generation algorithm changes.
        true_sample = np.array(
            [
                [0.2379328690962694, 0.2988121617836623, 0.3540711883388259, 0.011804150087474569],
                [0.5717283616874396, 0.8457204749453818, 0.5998019968276497, 0.8653851925864751],
                [0.2975410791646444, 0.719019775383696, 0.9111095799461322, 0.35764889149581724],
                [0.7865408130452101, 0.20797630306113987, 0.16828594953947298, 0.7065550077006459],
            ]
        )
        distribution = LatinHypercube(dimension=4, replications=None, seed=7)
        x = distribution.gen_samples(n_min=0, n_max=4, warn=False)
        self.assertTrue((x == true_sample).all())

    def test_values_seed_7_not_randomized(self):
        # Same seed/shape as test_values_seed_7, but randomize=False: points
        # sit exactly at stratum centers instead of a jittered position.
        true_sample = np.array(
            [
                [0.125, 0.375, 0.375, 0.125],
                [0.625, 0.875, 0.625, 0.875],
                [0.375, 0.625, 0.875, 0.375],
                [0.875, 0.125, 0.125, 0.625],
            ]
        )
        distribution = LatinHypercube(dimension=4, replications=None, seed=7, randomize="False")
        x = distribution.gen_samples(n_min=0, n_max=4, warn=False)
        self.assertTrue((x == true_sample).all())

    def test_values_seed_13_replications(self):
        # We should get the same result if we use the same seed = 13
        true_sample = np.array(
            [
                [[0.5749269005334164, 0.7367635418185489],
                 [0.8027989348020131, 0.09642877089260105],
                 [0.2028827909704837, 0.5170029561963858]],
                [[0.36408554620435707, 0.1218666235293967],
                 [0.8182985171529366, 0.8428148875176115],
                 [0.08025760006653519, 0.5347078626517302]],
            ]
        )
        distribution = LatinHypercube(dimension=2, replications=2, seed=13)
        x = distribution.gen_samples(n_min=0, n_max=3, warn=False)
        self.assertTrue((x == true_sample).all())

    def test_values_seed_13_replications_not_randomized(self):
        true_sample = np.array(
            [
                [[0.5, 0.8333333333333334],
                 [0.8333333333333334, 0.16666666666666666],
                 [0.16666666666666666, 0.5]],
                [[0.5, 0.16666666666666666],
                 [0.8333333333333334, 0.8333333333333334],
                 [0.16666666666666666, 0.5]],
            ]
        )
        distribution = LatinHypercube(dimension=2, replications=2, seed=13, randomize="False")
        x = distribution.gen_samples(n_min=0, n_max=3, warn=False)
        self.assertTrue((x == true_sample).all())

    def test_not_randomized_points_are_stratum_centers(self):
        # Every coordinate must be exactly (k - 0.5) / n for some integer k:
        # the defining property of "centered" (non-jittered) LHS.
        n, d = 20, 4
        distribution = LatinHypercube(dimension=d, replications=3, seed=5, randomize="False")
        x = distribution.gen_samples(n, warn=False)
        centered = x * n + 0.5
        npt.assert_allclose(centered, np.round(centered), atol=0)


    def test_randomize_invalid_value_raises(self):
        with self.assertRaises(ParameterError):
            LatinHypercube(dimension=2, replications=None, seed=1, randomize="banana")

    def test_spawn_preserves_randomize(self):
        # Regression test: _spawn used to silently drop `randomize`, so a
        # spawned child always reverted to the "TRUE" default regardless of
        # the parent's setting.
        parent = LatinHypercube(dimension=2, replications=None, seed=7, randomize="False")
        children = parent.spawn(s=1, dimensions=3)
        self.assertEqual(children[0].randomize, "FALSE")

        parent_true = LatinHypercube(dimension=2, replications=None, seed=7, randomize="True")
        children_true = parent_true.spawn(s=1, dimensions=3)
        self.assertEqual(children_true[0].randomize, "TRUE")

    def test_stratification_property(self):
        # Core LHS invariant: in every dimension, splitting [0,1) into n
        # equal strata must yield exactly one point per stratum.
        n, d = 10, 5
        distribution = LatinHypercube(dimension=d, replications=None, seed=42)
        x = distribution.gen_samples(n, warn=False)
        for j in range(d):
            strata = np.floor(x[:, j] * n).astype(int)
            self.assertEqual(sorted(strata), list(range(n)))

    def test_stratification_property_not_randomized(self):
        # Same invariant must hold when randomize=False: centering within a
        # stratum does not affect which stratum a point falls into.
        n, d = 10, 5
        distribution = LatinHypercube(dimension=d, replications=None, seed=42, randomize="False")
        x = distribution.gen_samples(n, warn=False)
        for j in range(d):
            strata = np.floor(x[:, j] * n).astype(int)
            self.assertEqual(sorted(strata), list(range(n)))

    def test_stratification_property_with_replications(self):
        n, d, reps = 8, 3, 4
        distribution = LatinHypercube(dimension=d, replications=reps, seed=42)
        x = distribution.gen_samples(n, warn=False)
        for r in range(reps):
            for j in range(d):
                strata = np.floor(x[r, :, j] * n).astype(int)
                self.assertEqual(sorted(strata), list(range(n)))

    def test_stratification_property_with_replications_not_randomized(self):
        # This combination (replications > 1, randomize=False) is exactly
        # the one that exposed the shape bug: verifying the stratification
        # invariant here also implicitly re-checks the shape is correct,
        # since a wrong shape would make this loop fail outright.
        n, d, reps = 8, 3, 4
        distribution = LatinHypercube(dimension=d, replications=reps, seed=42, randomize="False")
        x = distribution.gen_samples(n, warn=False)
        for r in range(reps):
            for j in range(d):
                strata = np.floor(x[r, :, j] * n).astype(int)
                self.assertEqual(sorted(strata), list(range(n)))

    def test_values_in_unit_cube(self):
        distribution = LatinHypercube(dimension=4, replications=3, seed=11)
        x = distribution.gen_samples(20, warn=False)
        self.assertTrue((x >= 0).all() and (x < 1).all())

    def test_values_in_unit_cube_not_randomized(self):
        distribution = LatinHypercube(dimension=4, replications=3, seed=11, randomize="False")
        x = distribution.gen_samples(20, warn=False)
        self.assertTrue((x >= 0).all() and (x < 1).all())

    def test_reproducibility_same_seed(self):
        d1 = LatinHypercube(dimension=3, replications=None, seed=123)
        d2 = LatinHypercube(dimension=3, replications=None, seed=123)
        x1 = d1.gen_samples(6, warn=False)
        x2 = d2.gen_samples(6, warn=False)
        self.assertTrue((x1 == x2).all())

    def test_reproducibility_same_seed_not_randomized(self):
        d1 = LatinHypercube(dimension=3, replications=None, seed=123, randomize="False")
        d2 = LatinHypercube(dimension=3, replications=None, seed=123, randomize="False")
        x1 = d1.gen_samples(6, warn=False)
        x2 = d2.gen_samples(6, warn=False)
        self.assertTrue((x1 == x2).all())

    def test_return_binary_raises(self):
        distribution = LatinHypercube(dimension=2, replications=None, seed=7)
        with self.assertRaises(ParameterError):
            distribution.gen_samples(4, return_binary=True, warn=False)

    def test_no_warning_when_disabled(self):
        distribution = LatinHypercube(dimension=2, replications=None, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            distribution.gen_samples(4, warn=False)



if __name__ == "__main__":
    unittest.main()
