from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scripts.remove_trailing_whitespace import remove_trailing_whitespace

from qmcpy.util import (
    _univ_repr,
    NotYetImplemented,
    ParameterError,
    MethodImplementationError,
)
from qmcpy.util.data import Data
from qmcpy.util.dig_shift_invar_ops import to_bin, to_float, weighted_walsh_funcs
from qmcpy.util.latnetbuilder_linker import latnetbuilder_linker
from qmcpy.util.shift_invar_ops import bernoulli_poly
from qmcpy.util.torch_numpy_ops import get_npt
from qmcpy.util.transforms import (
    insert_batch_dims,
    parse_assign_param,
    tf_exp,
    tf_exp_inv,
    tf_square,
    tf_square_inv,
    tf_identity,
    tf_explinear,
    tf_explinear_inv,
)


class TestTorchNumpyOps(unittest.TestCase):
    """Tests for torch_numpy_ops.get_npt()."""

    def test_get_npt_numpy_array(self):
        """get_npt should return np for numpy arrays."""
        x = np.array([1.0, 2.0, 3.0])
        npt = get_npt(x)
        self.assertEqual(npt, np)

    def test_get_npt_torch_tensor(self):
        """get_npt should return torch for torch tensors."""
        try:
            import torch
            x = torch.tensor([1.0, 2.0, 3.0])
            npt = get_npt(x)
            self.assertEqual(npt, torch)
        except ImportError:
            self.skipTest("torch not installed")


class TestTransforms(unittest.TestCase):
    """Tests for transform functions."""

    def test_insert_batch_dims(self):
        """insert_batch_dims should add batch dimensions."""
        param = np.array([1.0, 2.0, 3.0])
        result = insert_batch_dims(param, ndims=2, k=0)
        self.assertEqual(result.shape, (1, 1, 3))

    def test_insert_batch_dims_middle(self):
        """insert_batch_dims should work with k > 0."""
        param = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = insert_batch_dims(param, ndims=1, k=1)
        self.assertEqual(result.shape, (2, 1, 2))

    def test_tf_exp(self):
        """tf_exp should compute element-wise exponential."""
        x = np.array([0.0, 1.0, 2.0])
        result = tf_exp(x)
        expected = np.exp(x)
        np.testing.assert_allclose(result, expected)

    def test_tf_exp_inv(self):
        """tf_exp_inv should compute element-wise logarithm."""
        x = np.array([1.0, np.e, np.e**2])
        result = tf_exp_inv(x)
        expected = np.log(x)
        np.testing.assert_allclose(result, expected)

    def test_tf_square(self):
        """tf_square should square elements."""
        x = np.array([1.0, 2.0, 3.0])
        result = tf_square(x)
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_equal(result, expected)

    def test_tf_square_inv(self):
        """tf_square_inv should take square root."""
        x = np.array([1.0, 4.0, 9.0])
        result = tf_square_inv(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_tf_identity(self):
        """tf_identity should return input unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        result = tf_identity(x)
        np.testing.assert_array_equal(result, x)

    def test_tf_explinear(self):
        """tf_explinear should apply explinear transform."""
        x = np.array([0.0, 1.0, 2.0])
        result = tf_explinear(x)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_tf_explinear_inv(self):
        """tf_explinear_inv should apply inverse explinear transform."""
        x = np.array([0.5, 1.0, 2.0])
        result = tf_explinear_inv(x)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_tf_explinear_inv_large(self):
        """tf_explinear_inv should handle large values correctly."""
        x = np.array([35.0, 100.0])
        result = tf_explinear_inv(x)
        # For large x, should return approximately x
        np.testing.assert_allclose(result[1], x[1], rtol=1e-10)

    def test_parse_assign_param_numpy_constraints(self):
        """parse_assign_param should respect numpy constraints."""
        raw = parse_assign_param(
            pname="sigma",
            param=2.0,
            shape_param=[1],
            requires_grad_param=False,
            tfs_param=(tf_identity, tf_identity),
            endsize_ops=[1, 2],
            constraints=["POSITIVE"],
            torchify=False,
            npt=np,
            nptkwargs={},
        )
        np.testing.assert_allclose(raw, np.array([2.0]))

        with self.assertRaises(AssertionError):
            parse_assign_param(
                pname="k",
                param=1.5,
                shape_param=[1],
                requires_grad_param=False,
                tfs_param=(tf_identity, tf_identity),
                endsize_ops=[1],
                constraints=["INTEGER"],
                torchify=False,
                npt=np,
                nptkwargs={},
            )


class TestDigShiftInvarOps(unittest.TestCase):
    """Tests for digital shift invariant utility helpers."""

    def test_digital_shift_roundtrip_and_dtype_errors(self):
        """to_bin and to_float should roundtrip float inputs."""
        xf = np.array([0.0, 0.125, 0.25, 0.875], dtype=float)
        xb = to_bin(xf, t=3)
        self.assertEqual(xb.dtype, np.uint64)
        x_back = to_float(xb, t=3)
        np.testing.assert_allclose(x_back, xf)

        with self.assertRaises(ParameterError):
            to_bin(np.array([1 + 2j], dtype=np.complex128), t=3)

    def test_weighted_walsh_funcs_and_bernoulli_poly_input_checks(self):
        """weighted_walsh_funcs and bernoulli_poly should validate inputs."""
        xb = np.array([[0, 1], [2, 3]], dtype=np.uint64)
        y = weighted_walsh_funcs(alpha=2, xb=xb, t=2)
        self.assertEqual(y.shape, xb.shape)
        self.assertTrue(np.isfinite(y).all())

        with self.assertRaises(AssertionError):
            weighted_walsh_funcs(alpha=5, xb=xb, t=2)

        yb = bernoulli_poly(2, np.array([0.0, 0.5, 1.0]))
        self.assertEqual(yb.shape, (3,))
        with self.assertRaises(AssertionError):
            bernoulli_poly(11, np.array([0.1]))


class TestLatnetbuilderLinker(unittest.TestCase):
    """Tests for latnetbuilder linker utility."""

    def test_latnetbuilder_linker_not_implemented(self):
        """latnetbuilder_linker should currently raise NotYetImplemented."""
        with self.assertRaises(NotYetImplemented):
            latnetbuilder_linker()


class TestUnivRepr(unittest.TestCase):
    """Tests for _univ_repr utility function."""

    @staticmethod
    def _make_mock(**attrs):
        """helper function to create a mock object with specified attributes."""
        obj = type("MockObject", (), {})()
        for key, value in attrs.items():
            setattr(obj, key, value)
        return obj

    def test_univ_repr_scalar_values(self):
        """_univ_repr should format scalar values nicely."""
        obj = self._make_mock(n_samples=128, dimension=5)
        result = _univ_repr(obj, "MockClass", ["n_samples", "dimension"])
        self.assertIn("MockClass", result)
        self.assertIn("n_samples", result)
        self.assertIn("dimension", result)

    def test_univ_repr_power_of_two(self):
        """_univ_repr should recognize powers of 2."""
        obj = self._make_mock(n=512)  # 2^9
        result = _univ_repr(obj, "Test", ["n"])
        self.assertIn("2^(9)", result)

    def test_univ_repr_array_values(self):
        """_univ_repr should handle numpy arrays."""
        obj = self._make_mock(data=np.array([1.0, 2.0, 3.0]))
        result = _univ_repr(obj, "Test", ["data"])
        self.assertIn("Test", result)

    def test_univ_repr_list_values(self):
        """_univ_repr should handle list values."""
        obj = self._make_mock(seeds=[1, 2, 3])
        result = _univ_repr(obj, "Test", ["seeds"])
        self.assertIn("Test", result)


class TestData(unittest.TestCase):
    """Tests for Data class."""

    @staticmethod
    def _make_data(parameters=None, **attrs):
        data = Data(parameters or [])
        data.time_integrate = 0.0
        for key, value in attrs.items():
            setattr(data, key, value)
        return data

    def test_data_init(self):
        """Data should initialize with parameters list."""
        d = Data(["param1", "param2"])
        self.assertEqual(d.parameters, ["param1", "param2"])

    def test_data_repr_empty(self):
        """Data.__repr__ should generate proper string with empty params."""
        d = self._make_data()
        result = str(d)
        self.assertIn("Data", result)

    def test_data_repr_with_values(self):
        """Data.__repr__ should show attributes when parameters are set."""
        d = self._make_data(parameters=["test_param"], test_param=42)
        d.time_integrate = 0.5
        result = str(d)
        self.assertIn("Data", result)
        self.assertIn("test_param", result)

    def test_data_repr_with_optional_attributes(self):
        """Data.__repr__ should show optional related objects if set."""
        for attr, value in [
            ("stopping_crit", "test_crit"),
            ("integrand", "test_integrand"),
            ("true_measure", "test_measure"),
            ("discrete_distrib", "test_distrib"),
        ]:
            with self.subTest(attr=attr):
                d = self._make_data(**{attr: value})
                result = str(d)
                self.assertIn(value, result)


class TestExceptionsWarnings(unittest.TestCase):
    """Tests for exception and warning classes."""

    class _TestClass:
        pass

    def test_parameter_error_instantiation(self):
        """ParameterError should be instantiable with message."""
        err = ParameterError("Test error message")
        self.assertIn("Test error message", str(err))

    def test_parameter_error_inheritance(self):
        """ParameterError should inherit from Exception."""
        err = ParameterError("Test")
        self.assertIsInstance(err, Exception)

    def test_parameter_error_raised(self):
        """ParameterError should be raised and caught properly."""
        with self.assertRaises(ParameterError):
            raise ParameterError("test")

    def test_method_implementation_error_instantiation(self):
        """MethodImplementationError needs subclass and method_name."""
        err = MethodImplementationError(self._TestClass(), "test_method")
        self.assertIn("_TestClass", str(err))
        self.assertIn("test_method", str(err))

    def test_method_implementation_error_inheritance(self):
        """MethodImplementationError should inherit from Exception."""
        err = MethodImplementationError(self._TestClass(), "test")
        self.assertIsInstance(err, Exception)

    def test_method_implementation_error_raised(self):
        """MethodImplementationError should be raised and caught."""
        with self.assertRaises(MethodImplementationError):
            raise MethodImplementationError(self._TestClass(), "test")


class TestRemoveTrailingWhitespace(unittest.TestCase):
    def test_preserves_python_string_contents_and_line_endings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.py"
            original = b'"""expected output  \r\n"""\r\nx = "value"  \r\ny = 2\t\n'
            expected = b'"""expected output  \r\n"""\r\nx = "value"\r\ny = 2\n'
            path.write_bytes(original)

            self.assertTrue(remove_trailing_whitespace(path, check=True))
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(remove_trailing_whitespace(path, check=False))
            self.assertEqual(path.read_bytes(), expected)
            self.assertFalse(remove_trailing_whitespace(path, check=True))

    def test_preserves_multiline_f_string_trailing_spaces(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.py"
            original = b'x = f"""hello  \nworld  \n"""\nvalue = 1  \n'
            expected = b'x = f"""hello  \nworld  \n"""\nvalue = 1\n'
            path.write_bytes(original)

            self.assertTrue(remove_trailing_whitespace(path, check=False))
            self.assertEqual(path.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
