import builtins
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import qmcpy
from qmcpy import plot_proj
from qmcpy.util import stop_notebook


class FakeAxes:
    def __init__(self):
        self.removed = False
        self.calls = []

    def remove(self):
        self.removed = True

    def set_xlim(self, *a, **k):
        self.calls.append(("set_xlim", a))

    def set_ylim(self, *a, **k):
        self.calls.append(("set_ylim", a))

    def set_xticks(self, *a, **k):
        self.calls.append(("set_xticks", a))

    def set_yticks(self, *a, **k):
        self.calls.append(("set_yticks", a))

    def set_aspect(self, *a, **k):
        self.calls.append(("set_aspect", a))

    def grid(self, *a, **k):
        self.calls.append(("grid", a))

    def tick_params(self, *a, **k):
        self.calls.append(("tick_params", a))

    def set_xlabel(self, *a, **k):
        self.calls.append(("set_xlabel", a))

    def set_ylabel(self, *a, **k):
        self.calls.append(("set_ylabel", a))

    def scatter(self, *a, **k):
        self.calls.append(("scatter", a))


class FakeFig:
    def __init__(self):
        self.tl = False

    def tight_layout(self, *a, **k):
        self.tl = True


def make_fake_matplotlib(nrows, ncols):
    plt = types.ModuleType("matplotlib.pyplot")
    plt.style = types.SimpleNamespace()
    plt.style.use = lambda *a, **k: None
    plt.rcParams = {
        "font.family": "sans-serif",
        "axes.prop_cycle": types.SimpleNamespace(
            by_key=lambda: {"color": ["k", "b", "r"]}
        ),
    }

    def subplots(nrows=1, ncols=1, figsize=None, squeeze=False):
        fig = FakeFig()
        ax = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                ax[i, j] = FakeAxes()
        return fig, ax

    plt.subplots = subplots
    plt.suptitle = lambda *a, **k: None
    return plt


class DummySampler(qmcpy.AbstractDiscreteDistribution):
    def __init__(self, d=2):
        super().__init__(dimension=d, replications=1, seed=1, d_limit=10, n_limit=100)

    def _gen_samples(self, n_min, n_max, return_binary=False, warn=True):
        n = n_max - n_min
        return np.tile(np.arange(n)[:, None] / max(1, n - 1), (1, 1, self.d)).reshape(
            self.replications, n, self.d
        )

    def __repr__(self):
        return "DummySampler"


class TestPlotProjAndStopNotebook(unittest.TestCase):

    def test_plot_proj_with_fake_matplotlib_and_sampler(self):
        # Inject fake matplotlib.pyplot
        fake_plt = make_fake_matplotlib(1, 1)
        # Create a proper matplotlib package module with colors submodule
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.pyplot = fake_plt
        fake_matplotlib.colors = types.SimpleNamespace()

        with patch.dict(
            sys.modules,
            {"matplotlib.pyplot": fake_plt, "matplotlib": fake_matplotlib},
        ):
            sampler = DummySampler(d=3)
            fig, ax = plot_proj(
                sampler,
                n=4,
                d_horizontal=1,
                d_vertical=2,
                math_ind=True,
                marker_size=1,
                figfac=1,
            )

        self.assertIsInstance(fig, FakeFig)
        self.assertIsInstance(ax, np.ndarray)
        # At least one axes should have scatter calls or be removed
        found = False
        for a in ax.flatten():
            if getattr(a, "removed", False) or any(c[0] == "scatter" for c in a.calls):
                found = True
                break
        self.assertTrue(found)

    def test_plot_proj_with_callable_sampler(self):
        # sampler not instance of AbstractDiscreteDistribution -> uses t_i labels
        fake_plt = make_fake_matplotlib(1, 1)
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.pyplot = fake_plt
        fake_matplotlib.colors = types.SimpleNamespace()

        with patch.dict(
            sys.modules,
            {"matplotlib.pyplot": fake_plt, "matplotlib": fake_matplotlib},
        ):
            def sampler_callable(n):
                return np.zeros((n, 1))

            fig, ax = plot_proj(
                sampler_callable, n=3, d_horizontal=0, d_vertical=0, math_ind=False
            )

        self.assertIsInstance(fig, FakeFig)

    def test_stop_notebook_yes_and_no(self):
        # When input is 'yes' nothing should happen
        with patch.object(builtins, "input", lambda prompt="": "yes"):
            # Should not raise
            stop_notebook("prompt")

        # When input is not 'yes' should exit
        with patch.object(builtins, "input", lambda prompt="": "no"):
            with self.assertRaises(SystemExit):
                stop_notebook("prompt")


if __name__ == "__main__":
    unittest.main()
