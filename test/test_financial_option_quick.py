import numpy as np

from qmcpy import FinancialOption
import qmcpy


class SmallSampler(qmcpy.AbstractDiscreteDistribution):
    def __init__(self, d=3):
        super().__init__(
            dimension=d, replications=1, seed=123, d_limit=100, n_limit=1024
        )

    def _gen_samples(self, n_min, n_max, return_binary=False, warn=True):
        n = n_max - n_min
        # return shape (replications, n, d)
        arr = np.tile(np.linspace(0.1, 1.0, n)[:, None], (1, self.d))
        return arr.reshape(self.replications, n, self.d)


def test_financial_option_payoffs_and_exact():
    sampler = SmallSampler(d=3)
    fo = FinancialOption(
        sampler,
        option="EUROPEAN",
        call_put="CALL",
        volatility=0.5,
        start_price=30,
        strike_price=25,
        interest_rate=0.01,
        t_final=1,
    )
    gbm = np.array([[30.0, 28.0, 35.0]])
    c = fo.payoff_european_call(gbm)
    p = fo.payoff_european_put(gbm)
    assert c.shape == (1,)
    assert p.shape == (1,)

    # Asian arithmetic trapezoidal
    fo_asian = FinancialOption(
        sampler,
        option="ASIAN",
        asian_mean="ARITHMETIC",
        asian_mean_quadrature_rule="TRAPEZOIDAL",
    )
    gbm2 = np.array([[30.0, 32.0, 34.0]])
    a_call = fo_asian.payoff_asian_arithmetic_trap_call(gbm2)
    assert a_call.shape == (1,)

    # geometric right call
    fo_geo = FinancialOption(
        sampler,
        option="ASIAN",
        asian_mean="GEOMETRIC",
        asian_mean_quadrature_rule="RIGHT",
    )
    g_call = fo_geo.payoff_asian_geometric_right_call(np.array([[30.0, 30.0, 30.0]]))
    assert g_call.shape == (1,)

    # barrier options: up and down behaviors
    fo_barrier_up = FinancialOption(
        sampler, option="BARRIER", barrier_in_out="IN", barrier_price=25, start_price=20
    )
    gbm_up = np.array([[20.0, 26.0, 27.0]])
    v = fo_barrier_up.payoff_barrier_in_up_call(gbm_up)
    assert v.shape == (1,)

    fo_barrier_out = FinancialOption(
        sampler,
        option="BARRIER",
        barrier_in_out="OUT",
        barrier_price=40,
        start_price=30,
    )
    gbm_out = np.array([[30.0, 32.0, 33.0]])
    v2 = fo_barrier_out.payoff_barrier_out_up_call(gbm_out)
    assert v2.shape == (1,)

    # lookback
    fo_lb = FinancialOption(sampler, option="LOOKBACK")
    lb = fo_lb.payoff_lookback_call(np.array([[10.0, 9.0, 12.0]]))
    assert lb.shape == (1,)

    # digital
    fo_dig = FinancialOption(sampler, option="DIGITAL", digital_payout=5)
    dig = fo_dig.payoff_digital_call(np.array([[10.0, 11.0, 12.0]]))
    assert dig.shape == (1,)

    # exact value for European should return a float
    val = fo.get_exact_value()
    assert np.isscalar(val)

    # exact value for Asian geometric right
    val2 = fo_geo.get_exact_value()
    assert np.isscalar(val2)
