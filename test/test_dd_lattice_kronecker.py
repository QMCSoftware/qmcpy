import numpy as np
import numpy.testing as npt
import pytest

from qmcpy import (
    Kronecker,
    Lattice,
    kronecker_vector_search_mobius_transform,
    lattice_vector_wssd_search,
)

######################################################
# Helper functions
######################################################
def _bernoulli_two(x):
    return x * (x - 1) + 1 / 6


def _periodic_kernel(x, coord_weights):
    return np.prod(1 + _bernoulli_two(x) * coord_weights, axis=-1)


def _direct_squared_discrepancies(points, coord_weights):
    """Evaluate the periodic-kernel definition directly for small prefixes."""
    return np.array(
        [
            _periodic_kernel(
                (points[:n, None] - points[None, :n]) % 1, coord_weights
            ).mean()
            - 1
            for n in range(1, len(points) + 1)
        ]
    )


######################################################
# Test class for Lattice and Kronecker methods
######################################################
class TestLatticeKroneckerMethods(object):

    def test_lattice_discrepancy_and_wssd(self):
        n, coord_weights = 8, np.array([1.0, 0.25])
        lattice = Lattice(2, randomize=False, order="RADICAL_INVERSE")
        expected = _direct_squared_discrepancies(
            lattice.gen_samples(n=n, warn=False), coord_weights
        )

        for actual in (
            lattice.expected_squared_periodic_discrepancies(n),
            lattice.expected_squared_periodic_discrepancies(
                n, coord_weights=coord_weights, kernel=_bernoulli_two
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

    def test_lattice_validation(self):
        lattice = Lattice(2, randomize=False)
        with pytest.raises(ValueError, match="coord_weights"):
            lattice.expected_squared_periodic_discrepancies(8, coord_weights=[1.0])
        with pytest.raises(ValueError, match="coord_weights"):
            lattice.wssd(8, coord_weights=[1.0])
        with pytest.raises(ValueError, match="sample_weights"):
            lattice.wssd(8, sample_weights=np.ones(7))
        with pytest.raises(NotImplementedError, match="linear order"):
            Lattice(2, randomize=False, order="LINEAR").expected_squared_periodic_discrepancies(8)

    def test_lattice_vector_search(self):
        default = lattice_vector_wssd_search(16, 4, None, None)
        explicit = lattice_vector_wssd_search(
            n_max=16,
            d_max=4,
            coord_weights=np.array([1.0, 0.25, 1 / 9, 1 / 16]),
            kernel=_bernoulli_two,
        )
        npt.assert_array_equal(default, np.array([1, 5, 3, 7]))
        npt.assert_array_equal(explicit, default)
        assert default.shape == (4,) and default.dtype.kind in "iu"
        assert len(np.unique(default)) == len(default) and np.all(default % 2 == 1)

    def test_kronecker_discrepancy_and_wssd(self):
        n = 8
        kronecker = Kronecker(
            2, generating_vector="SUZUKI", randomize="SHIFT", shift=[0.1, 0.2]
        )
        points = (np.arange(n)[:, None] * kronecker.gen_vec[0]) % 1
        sample_weights = np.arange(1, n + 1)
        expected = _direct_squared_discrepancies(points, np.ones(2))
        actual = kronecker.periodic_discrepancy(n) ** 2
        assert actual.shape == (1, n)
        npt.assert_allclose(actual, expected[None], rtol=0, atol=5e-15)
        npt.assert_allclose(
            kronecker.wssd_discrepancy(n, sample_weights),
            [sample_weights @ expected],
            rtol=0,
            atol=5e-14,
        )

        coord_weights, kernel = np.array([1.0, 0.25]), (_periodic_kernel, 1)
        expected = _direct_squared_discrepancies(points, coord_weights)
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

    def test_cbc_mobius_fallback(self):
        kronecker = Kronecker(3, generating_vector="CBC_MT", randomize=False)
        assert kronecker.gen_vec_source == "CBC_MT"
        assert kronecker.gen_vec.shape == (1, 3)
        assert np.isfinite(kronecker.gen_vec).all()

        with pytest.warns(RuntimeWarning, match="CBC_MT.*dimension <= 100"):
            fallback = Kronecker(101, generating_vector="CBC_MT", randomize=False)
        assert fallback.gen_vec_source == "RICHTMYER"
        assert fallback.gen_vec.shape == (1, 101)

    def test_kronecker_search(self):
        n = 8
        vector, wssd, discrepancies, coefficients = (
            kronecker_vector_search_mobius_transform(
                n_max=n, d_max=3, searchsize=3
            )
        )
        assert vector.shape == (3,) and discrepancies.shape == (n,)
        assert coefficients.shape == (2, 4)
        assert np.isfinite(vector).all() and np.isfinite(discrepancies).all()
        assert np.all((0 <= vector) & (vector < 1))
        assert 0 < wssd
        npt.assert_allclose(
            wssd, np.arange(1, n + 1) @ discrepancies, rtol=0, atol=5e-14
        )

        coord_weights = np.array([1.0, 0.25, 1 / 9])
        points = (np.arange(n)[:, None] * vector) % 1
        npt.assert_allclose(
            discrepancies,
            _direct_squared_discrepancies(points, coord_weights),
            rtol=0,
            atol=5e-15,
        )

        vector, wssd, discrepancies, coefficients = (
            kronecker_vector_search_mobius_transform(
                n_max=n,
                d_max=3,
                searchsize=3,
                kernel=_bernoulli_two,
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
                        kernel= lambda x: 3 * _bernoulli_two(x),
                        coord_weights=coord_weights,
                        gen_vec_init=1.25,
                    )
                )
        assert 0 < wssd
        
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
    def test_kronecker_search_validation(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            kronecker_vector_search_mobius_transform(**kwargs)
