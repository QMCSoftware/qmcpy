from qmcpy import *
from qmcpy.util import *
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
            ]
            for dd in dds:
                for _dd in [dd] + dd.spawn(1):
                    x = _dd.gen_samples(4)
                    if _dd.mimics == "StdUniform":
                        self.assertTrue((x > 0).all() and (x < 1).all())
                    pdf = _dd.pdf(_dd.gen_samples(4))
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
        ]:
            s = 3
            for spawn_dim in [4, [1, 4, 6]]:
                spawns = dd.spawn(s=s, dimensions=spawn_dim)
                self.assertEqual(len(spawns), s)
                self.assertTrue(all(type(spawn) == type(dd) for spawn in spawns))
                self.assertTrue(
                    (np.array([spawn.d for spawn in spawns]) == spawn_dim).all()
                )


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


if __name__ == "__main__":
    unittest.main()
