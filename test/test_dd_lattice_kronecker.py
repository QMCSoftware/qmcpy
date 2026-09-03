import inspect
import re
import sys

import numpy as np
import numpy.testing as npt
import pytest

from qmcpy import (
    Kronecker,
    kronecker_vector_search_mobius_transform,
    Lattice,
    lattice_vector_wssd_search,
)

######################################################
# Helper functions
######################################################
def _bern2(x):
    return x * (x - 1) + 1 / 6


def _pkern(x, coord_weights):
    return np.prod(1 + _bern2(x) * coord_weights, axis=-1)


def _direct_disc(points, coord_weights):
    """Evaluate the periodic-kernel definition directly for small prefixes."""
    return np.array(
        [
            _pkern(
                (points[:n, None] - points[None, :n]) % 1, coord_weights
            ).mean()
            - 1
            for n in range(1, len(points) + 1)
        ]
    )


######################################################
# Test class for Lattice and Kronecker methods
######################################################
class TestLatKron(object):

    def test_lat_disc_wssd(self):
        n, coord_weights = 8, np.array([1.0, 0.25])
        lattice = Lattice(2, randomize=False, order="RADICAL_INVERSE")
        expected = _direct_disc(
            lattice.gen_samples(n=n, warn=False), coord_weights
        )

        for actual in (
            lattice.expected_squared_periodic_discrepancies(n),
            lattice.expected_squared_periodic_discrepancies(
                n, coord_weights=coord_weights, kernel=_bern2
            ),
        ):
            assert actual.shape == (n,) and np.isfinite(actual).all()
            npt.assert_allclose(actual, expected, rtol=0, atol=5e-15)

        npt.assert_allclose(
            lattice.wssd(n), np.arange(1, n + 1) @ expected, rtol=0, atol=5e-14
        )
        sample_weights = np.linspace(0.5, 1.5, n)
        npt.assert_allclose(
            lattice.wssd(
                n, coord_weights=coord_weights, sample_weights=sample_weights
            ),
            sample_weights @ expected,
            rtol=0,
            atol=5e-14,
        )

    def test_lat_valid(self):
        lattice = Lattice(2, randomize=False)
        with pytest.raises(ValueError, match="coord_weights"):
            lattice.expected_squared_periodic_discrepancies(8, coord_weights=[1.0])
        with pytest.raises(ValueError, match="coord_weights"):
            lattice.wssd(8, coord_weights=[1.0])
        # sample_weights length must match n_max exactly, both too short and too long
        for bad in (np.ones(7), np.ones(9)):
            with pytest.raises(ValueError, match="sample_weights"):
                lattice.wssd(8, sample_weights=bad)
        with pytest.raises(NotImplementedError, match="linear order"):
            Lattice(2, randomize=False, order="LINEAR").expected_squared_periodic_discrepancies(8)
        with pytest.raises(ValueError, match="n_max must be at least 8"):
            lattice_vector_wssd_search(3, 3)
        with pytest.raises(ValueError, match="candidate pool"):  # d_max > CBC pool, once an infinite loop
            lattice_vector_wssd_search(8, 3)

    def test_lat_wssd_kernel(self):
        # wssd must forward a custom kernel through to expected_squared_periodic_discrepancies
        lattice = Lattice(3, randomize=False, order="RADICAL_INVERSE")
        n = 16
        bern4 = lambda x: x**4 - 2 * x**3 + x**2 - 1 / 30
        npt.assert_allclose(
            lattice.wssd(n, kernel=bern4),
            np.arange(1, n + 1) @ lattice.expected_squared_periodic_discrepancies(n, kernel=bern4),
            rtol=0, atol=5e-14,
        )
        assert not np.isclose(lattice.wssd(n), lattice.wssd(n, kernel=bern4))

    def test_lat_search(self):
        default = lattice_vector_wssd_search(16, 4, None, None)
        npt.assert_array_equal(default, [1, 5, 3, 7])
        # passing the built-in weights and kernel explicitly must reproduce the default
        npt.assert_array_equal(
            lattice_vector_wssd_search(16, 4, np.array([1.0, 0.25, 1 / 9, 1 / 16]), _bern2),
            default,
        )
        # coord_weights[0] and kernel= must each reach the search (both were once ignored)
        bern6 = lambda x: x**6 - 3 * x**5 + 2.5 * x**4 - 0.5 * x**2 + 1 / 42
        w = np.array([1.0, 0.25, 1 / 9, 1 / 16])
        base = lattice_vector_wssd_search(64, 4, w)
        assert not np.array_equal(base, lattice_vector_wssd_search(64, 4, np.array([1e-6, 0.25, 1 / 9, 1 / 16])))
        assert not np.array_equal(base, lattice_vector_wssd_search(64, 4, w, bern6))
        # selection is deterministic and yields a valid generating vector
        v = lattice_vector_wssd_search(2**6, 5, kernel=bern6)
        npt.assert_array_equal(v, lattice_vector_wssd_search(2**6, 5, kernel=bern6))
        assert v[0] == 1 and len(np.unique(v)) == 5 and np.all((v % 2 == 1) & (0 < v) & (v < 2**6))

    def test_import_conventions(self):
        mod = lambda name: inspect.getsource(sys.modules["qmcpy.discrete_distribution." + name])
        # importing qmcpy must not need optional sympy: no top-level `import sympy`
        assert not re.search(r"(?m)^(import sympy|from sympy)\b", mod("kronecker.kronecker_search_methods"))
        # the discrepancy code must not use np.vecmat (a NumPy >= 2.2 only API)
        assert "np.vecmat" not in mod("lattice.lattice")

    def test_kron_disc_wssd(self):
        n = 8
        kronecker = Kronecker(
            2, generating_vector="SUZUKI", randomize="SHIFT", shift=[0.1, 0.2]
        )
        points = (np.arange(n)[:, None] * kronecker.gen_vec[0]) % 1
        sample_weights = np.arange(1, n + 1)
        expected = _direct_disc(points, np.ones(2))
        actual = kronecker.periodic_discrepancy(n) ** 2
        assert actual.shape == (1, n)
        npt.assert_allclose(actual, expected[None], rtol=0, atol=5e-15)
        npt.assert_allclose(
            kronecker.wssd_discrepancy(n, sample_weights),
            [sample_weights @ expected],
            rtol=0,
            atol=5e-14,
        )

        coord_weights, kernel = np.array([1.0, 0.25]), (_pkern, 1)
        expected = _direct_disc(points, coord_weights)
        for actual in (
            kronecker._square_periodic_discrepancies(n, kernel, coord_weights),
            kronecker.periodic_discrepancy(
                n, k_tilde=kernel, gamma=coord_weights
            )
            ** 2,
        ):
            npt.assert_allclose(actual, expected[None], rtol=0, atol=5e-15)
        npt.assert_allclose(
            kronecker.wssd_discrepancy(
                n, sample_weights, k_tilde=kernel, gamma=coord_weights
            ),
            [sample_weights @ expected],
            rtol=0,
            atol=5e-14,
        )

    def test_cbc_mt_fallback(self):
        kronecker = Kronecker(3, generating_vector="CBC_MT", randomize=False)
        assert kronecker.gen_vec_source == "CBC_MT"
        assert kronecker.gen_vec.shape == (1, 3)
        assert np.isfinite(kronecker.gen_vec).all()

        with pytest.warns(RuntimeWarning, match="CBC_MT.*dimension <= 100"):
            fallback = Kronecker(101, generating_vector="CBC_MT", randomize=False)
        assert fallback.gen_vec_source == "RICHTMYER"
        assert fallback.gen_vec.shape == (1, 101)

    def test_kron_search(self):
        n = 8
        coord_weights = np.array([1.0, 0.25, 1 / 9])  # == the j^-2 default for d=3
        # coord_weights also accepts a plain list, not only an ndarray
        vector, wssd, discrepancies, coefficients = (
            kronecker_vector_search_mobius_transform(n, 3, 3, coord_weights=list(coord_weights))
        )
        assert vector.shape == (3,) and discrepancies.shape == (n,)
        assert coefficients.shape == (2, 4)
        assert np.isfinite(vector).all() and np.isfinite(discrepancies).all()
        assert np.all((0 <= vector) & (vector < 1))
        assert 0 < wssd
        npt.assert_allclose(
            wssd, np.arange(1, n + 1) @ discrepancies, rtol=0, atol=5e-14
        )

        points = (np.arange(n)[:, None] * vector) % 1
        npt.assert_allclose(
            discrepancies,
            _direct_disc(points, coord_weights),
            rtol=0,
            atol=5e-15,
        )

        vector, wssd, discrepancies, coefficients = (
            kronecker_vector_search_mobius_transform(
                n_max=n,
                d_max=3,
                searchsize=3,
                kernel=_bern2,
                coord_weights=coord_weights,
                gen_vec_init=1.25,
            )
        )
        assert vector[0] == pytest.approx(0.25) and coefficients.shape == (2, 4)
        npt.assert_allclose(
            wssd, np.arange(1, n + 1) @ discrepancies, rtol=0, atol=5e-14
        )

        vector, wssd, discrepancies, coefficients = (
                    kronecker_vector_search_mobius_transform(
                        n_max=n,
                        d_max=3,
                        searchsize=3,
                        kernel= lambda x: 3 * _bern2(x),
                        coord_weights=coord_weights,
                        gen_vec_init=1.25,
                    )
                )
        assert 0 < wssd

    def test_kron_search_1d(self):
        # d_max == 1 skips the CBC loop; best_wssd was once left unbound (UnboundLocalError)
        n = 16
        vector, wssd, discrepancies, coeff = kronecker_vector_search_mobius_transform(n, 1, 3)
        assert vector.shape == (1,) and discrepancies.shape == (n,) and coeff.shape == (0, 4)
        npt.assert_allclose(wssd, np.arange(1, n + 1) @ discrepancies, rtol=0, atol=5e-14)

    def test_kron_search_no_sympy(self, monkeypatch):
        # the missing-sympy branch must warn (filterable), not print or call input()
        monkeypatch.setitem(sys.modules, "sympy", None)
        with pytest.warns(UserWarning, match="sympy"):
            vector, *_ = kronecker_vector_search_mobius_transform(8, 2, 2)
        assert vector.shape == (2,) and np.isfinite(vector).all()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_max": 8, "d_max": 2, "searchsize": 1}, "searchsize"),
            ({"n_max": 1, "d_max": 2, "searchsize": 2}, "n_max must"),
            ({"n_max": 8, "d_max": 0, "searchsize": 2}, "d_max"),
            (
                {
                    "n_max": 8,
                    "d_max": 3,
                    "searchsize": 2,
                    "coord_weights": np.ones(2),
                },
                "coord_weights",
            ),
        ],
    )
    def test_kron_search_valid(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            kronecker_vector_search_mobius_transform(**kwargs)
