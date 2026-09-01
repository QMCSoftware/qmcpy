import types
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyg_lib")
pytest.importorskip("torch_geometric")

from qmcpy import MPMC, MPMC_net, mpmc_utils
from qmcpy.util import ParameterError


UNWEIGHTED_DISCREPANCIES = [
    mpmc_utils.L2star,
    mpmc_utils.L2ext,
    mpmc_utils.L2per,
    mpmc_utils.L2ctr,
    mpmc_utils.L2sym,
    mpmc_utils.L2mix,
]

WEIGHTED_DISCREPANCIES = [
    mpmc_utils.L2star_weighted,
    mpmc_utils.L2ext_weighted,
    mpmc_utils.L2per_weighted,
    mpmc_utils.L2ctr_weighted,
    mpmc_utils.L2sym_weighted,
    mpmc_utils.L2mix_weighted,
]


class TestDiscreteDistributionMPMC(unittest.TestCase):
    @staticmethod
    def _sample_x():
        return torch.tensor(
            [
                [[0.1, 0.2], [0.7, 0.8], [0.3, 0.9]],
                [[0.2, 0.4], [0.9, 0.1], [0.6, 0.5]],
            ],
            dtype=torch.float32,
        )

    def test_utils_input_checks_and_helpers(self):
        sample_x = self._sample_x()
        gamma = torch.tensor([0.5, 1.0], dtype=torch.float32)
        b, n, d = mpmc_utils._check_inputs(sample_x, gamma)
        self.assertEqual((b, n, d), (2, 3, 2))

        xi, xj = mpmc_utils._pairwise(sample_x)
        self.assertEqual(xi.shape, (2, 3, 1, 2))
        self.assertEqual(xj.shape, (2, 1, 3, 2))

        safe = mpmc_utils._sqrt_safe(torch.tensor([-1e-9, 0.0, 4.0]))
        self.assertTrue(torch.all(torch.isfinite(safe)).item())
        self.assertAlmostEqual(safe[-1].item(), 2.0)

    def test_utils_invalid_shapes_raise(self):
        sample_x = self._sample_x()
        with self.assertRaises(ValueError):
            mpmc_utils._check_inputs(sample_x[0])

        bad_gamma = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with self.assertRaises(ValueError):
            mpmc_utils._check_inputs(sample_x, bad_gamma)

    def test_unweighted_discrepancies_return_finite_batch_values(self):
        sample_x = self._sample_x()
        for fn in UNWEIGHTED_DISCREPANCIES:
            with self.subTest(fn=fn.__name__):
                out = fn(sample_x)
                self.assertEqual(out.shape, (sample_x.shape[0],))
                self.assertTrue(torch.all(torch.isfinite(out)).item())
                self.assertTrue(torch.all(out >= 0).item())

    def test_weighted_discrepancies_return_finite_batch_values(self):
        sample_x = self._sample_x()
        gamma = torch.tensor([0.3, 0.7], dtype=torch.float32)
        for fn in WEIGHTED_DISCREPANCIES:
            with self.subTest(fn=fn.__name__):
                out = fn(sample_x, gamma)
                self.assertEqual(out.shape, (sample_x.shape[0],))
                self.assertTrue(torch.all(torch.isfinite(out)).item())
                self.assertTrue(torch.all(out >= 0).item())

    def test_mpmc_net_forward_and_invalid_loss(self):
        net = MPMC_net(
            dim=2,
            nhid=4,
            nlayers=1,
            nsamples=4,
            nbatch=1,
            radius=1.0,
            loss_fn="L2star",
            weights=None,
        )
        loss, x = net()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(x.shape, (1, 4, 2))
        self.assertTrue(torch.all((x >= 0) & (x <= 1)).item())

        with self.assertRaises(ValueError):
            MPMC_net(
                dim=2,
                nhid=4,
                nlayers=1,
                nsamples=4,
                nbatch=1,
                radius=1.0,
                loss_fn="NOT_A_REAL_LOSS",
                weights=None,
            )

    def test_mpmc_constructor_validation_and_repr(self):
        with self.assertRaises(ValueError):
            MPMC(dimension=2, loss_fn="L2star_weighted", weights=None, use_pretrained=False)

        with self.assertRaises(ValueError):
            MPMC(dimension=2, loss_fn="L2star", weights=[1.0], use_pretrained=False)

        with self.assertWarns(UserWarning):
            m = MPMC(dimension=2, loss_fn="L2star", weights=[0.5, 0.5], use_pretrained=False)
        self.assertEqual(m.loss_fn, "L2star_weighted")
        self.assertIn("MPMC Generator Object", repr(m))

    def test_mpmc_randomize_validation(self):
        with self.assertRaises(ParameterError):
            MPMC(dimension=2, randomize="bad-mode", use_pretrained=False)

    def test_gen_samples_validation(self):
        m = MPMC(dimension=2, randomize="false", use_pretrained=False)
        with self.assertRaises(ParameterError):
            m._gen_samples(1, 4, False, warn=False)
        with self.assertRaises(ParameterError):
            m._gen_samples(0, 4, True, warn=False)

        fake = np.ones((m.nbatch, 4, m.dim), dtype=float) * 0.25
        with patch.object(m, "_try_load_pretrained", lambda n: fake):
            out = m._gen_samples(0, 4, False, warn=False)
        np.testing.assert_allclose(out, fake)

    def test_gen_samples_shift_and_unrandomized(self):
        m = MPMC(dimension=2, randomize="shift", use_pretrained=False, seed=7)
        fake = np.zeros((m.nbatch, 4, m.dim), dtype=float)

        with patch.object(m, "_try_load_pretrained", lambda n: fake):
            xr, x = m._gen_samples(0, 4, False, warn=False, return_unrandomized=True)
        np.testing.assert_allclose(x, fake)
        self.assertEqual(xr.shape, fake.shape)
        self.assertTrue(np.all((xr >= 0) & (xr < 1)))

    def test_try_load_pretrained_branching(self):
        m = MPMC(dimension=2, randomize="false", use_pretrained=True, nbatch=1)

        m.use_pretrained = False
        self.assertIsNone(m._try_load_pretrained(16))
        m.use_pretrained = True

        self.assertIsNone(m._try_load_pretrained(17))

        with patch.object(m, "_load_pretrained_array", lambda n: (_ for _ in ()).throw(OSError("missing"))), \
             patch.object(m, "_ask_train_from_scratch", lambda: False):
            with self.assertRaises(RuntimeError):
                m._try_load_pretrained(16)

        with patch.object(m, "_load_pretrained_array", lambda n: np.zeros((5, 2), dtype=float)):
            with self.assertWarns(UserWarning):
                self.assertIsNone(m._try_load_pretrained(16))

        good = np.zeros((16, 2), dtype=float)
        with patch.object(m, "_load_pretrained_array", lambda n: good):
            loaded = m._try_load_pretrained(16)
        self.assertEqual(loaded.shape, (1, 16, 2))

        m2 = MPMC(dimension=2, randomize="false", use_pretrained=True, nbatch=3)
        with patch.object(m2, "_load_pretrained_array", lambda n: good):
            with self.assertWarns(UserWarning):
                loaded2 = m2._try_load_pretrained(16)
        self.assertEqual(loaded2.shape, (3, 16, 2))

    def test_load_pretrained_array_local_file(self):
        pts = np.arange(12, dtype=float).reshape(6, 2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            m = MPMC(
                dimension=2,
                randomize="false",
                use_pretrained=True,
                pretrained_local_dir=tmp_dir,
            )
            path = f"{tmp_dir}/{m._pretrained_filename(6)}"
            np.save(path, pts)
            loaded = m._load_pretrained_array(6)
            np.testing.assert_allclose(loaded, pts)

    def test_ask_train_from_scratch_non_tty(self):
        m = MPMC(dimension=2, randomize="false", use_pretrained=True, prompt_on_missing=True)
        with patch("sys.stdin", types.SimpleNamespace(isatty=lambda: False)):
            self.assertTrue(m._ask_train_from_scratch())

    def test_spawn_preserves_configuration(self):
        m = MPMC(
            dimension=2,
            randomize="shift",
            seed=11,
            nbatch=2,
            loss_fn="L2star_weighted",
            weights=[0.2, 0.8],
            use_pretrained=False,
        )
        child = m._spawn(child_seed=13, dimension=3)
        self.assertIsInstance(child, MPMC)
        self.assertEqual(child.dim, 3)
        self.assertEqual(child.nbatch, m.nbatch)
        self.assertTrue(child.loss_fn.endswith("_weighted"))

    def test_train_path_without_real_training(self):
        m = MPMC(dimension=2, randomize="false", use_pretrained=False)
        with patch.object(m, "_try_load_pretrained", lambda n: None), \
             patch.object(m, "_train", lambda args: np.zeros((m.nbatch, args.nsamples, m.dim), dtype=float)):
            out = m._gen_samples(0, 5, False, warn=False)
        self.assertEqual(out.shape, (m.nbatch, 5, m.dim))
