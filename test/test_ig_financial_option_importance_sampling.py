"""Explicit price-space importance sampling for FinancialOption (issue #623)."""

import unittest

import numpy as np
from scipy.stats import multivariate_normal

from qmcpy import (
    BrownianMotion,
    DigitalNetB2,
    FinancialOption,
    Gaussian,
    GeometricBrownianMotion,
    ImportanceSampling,
    Uniform,
)
from qmcpy.util import DimensionError, ParameterError


class TestFinancialOptionImportanceSampling(unittest.TestCase):
    model = dict(
        volatility=0.5, start_price=30, strike_price=35,
        interest_rate=0.05, t_final=1,
    )

    def make_proposal(self, sampler, **overrides):
        params = dict(
            initial_value=self.model["start_price"],
            drift=0.30,
            diffusion=self.model["volatility"]**2,
            t_final=self.model["t_final"],
        )
        params.update(overrides)
        return GeometricBrownianMotion(sampler, **params)

    def reference_ratio(self, prices, target, proposal):
        """Calculate the ratio independently in log-price coordinates."""
        log_prices = np.log(prices)
        def log_density(measure):
            mean = np.log(measure.initial_value) + (
                measure.drift - measure.diffusion / 2
            ) * measure.time_vec
            covariance = measure.diffusion * np.minimum.outer(
                measure.time_vec, measure.time_vec
            )
            return multivariate_normal.logpdf(log_prices, mean, covariance)
        return np.exp(log_density(target) - log_density(proposal))

    def test_ordinary_sampling_is_unchanged(self):
        sampler = DigitalNetB2(4, seed=7)
        points = sampler(32)
        for option in ("EUROPEAN", "ASIAN"):
            for call_put in ("CALL", "PUT"):
                for mean in ("ARITHMETIC", "GEOMETRIC"):
                    for rule in ("RIGHT", "TRAPEZOIDAL"):
                        with self.subTest(option=option, side=call_put, mean=mean, rule=rule):
                            args = dict(
                                self.model, option=option, call_put=call_put,
                                asian_mean=mean, asian_mean_quadrature_rule=rule,
                            )
                            ordinary = FinancialOption(sampler, **args)
                            explicit_none = FinancialOption(sampler, proposal=None, **args)
                            np.testing.assert_array_equal(
                                ordinary.f(points), explicit_none.f(points)
                            )
                            brownian = BrownianMotion(sampler)._transform(points)
                            prices = self.model["start_price"] * np.exp(
                                (self.model["interest_rate"] - self.model["volatility"]**2 / 2)
                                * ordinary.true_measure.time_vec
                                + self.model["volatility"] * brownian
                            )
                            expected = ordinary.g(prices)
                            np.testing.assert_allclose(
                                ordinary.f(points), expected, rtol=2e-13, atol=2e-13
                            )
                            self.assertIs(ordinary.sampler, sampler)
                            self.assertIs(ordinary.discrete_distrib, sampler)

    def test_identity_proposal(self):
        for replications in (None, 3):
            with self.subTest(replications=replications):
                sampler = DigitalNetB2(4, seed=7, replications=replications)
                proposal = self.make_proposal(sampler, drift=self.model["interest_rate"])
                ordinary = FinancialOption(sampler, **self.model)
                importance = FinancialOption(sampler, proposal=proposal, **self.model)
                points = sampler(32)
                samples, weights = importance.true_measure._importance_sampling_transform_r(points)
                np.testing.assert_array_equal(samples, ordinary.true_measure._transform(points))
                np.testing.assert_allclose(weights, 1, rtol=2e-15)
                np.testing.assert_allclose(importance.f(points), ordinary.f(points), rtol=2e-15)
                self.assertIs(importance.true_measure.target.discrete_distrib, sampler)
                self.assertIs(importance.discrete_distrib, sampler)

    def test_different_proposal_law_and_manual_weights(self):
        for replications in (None, 3):
            for call_put in ("CALL", "PUT"):
                with self.subTest(replications=replications, side=call_put):
                    sampler = DigitalNetB2(4, seed=7, replications=replications)
                    proposal = self.make_proposal(
                        sampler, initial_value=32, diffusion=0.36, decomp_type="Cholesky"
                    )
                    option = FinancialOption(
                        sampler, proposal=proposal, call_put=call_put, **self.model
                    )
                    points = sampler(64)
                    samples, weights = option.true_measure._importance_sampling_transform_r(points)
                    ratio = self.reference_ratio(samples, option.true_measure.target, proposal)
                    self.assertEqual(weights.shape, points.shape[:-1])
                    self.assertTrue(np.isfinite(option.f(points)).all())
                    np.testing.assert_allclose(weights, ratio, rtol=3e-13)
                    np.testing.assert_allclose(
                        option.f(points), option.g(samples) * ratio,
                        rtol=3e-13, atol=1e-13,
                    )
                    target = option.true_measure.target
                    self.assertEqual(target.initial_value, self.model["start_price"])
                    self.assertEqual(target.diffusion, self.model["volatility"]**2)
                    self.assertEqual(target.drift, self.model["interest_rate"])

    def test_historical_brownian_shift_equivalence(self):
        for decomposition in ("PCA", "Cholesky", "BrownianBridge"):
            for theta in (-0.5, 0.5):
                sampler = DigitalNetB2(4, seed=7)
                proposal = self.make_proposal(
                    sampler,
                    drift=self.model["interest_rate"] + self.model["volatility"] * theta,
                    decomp_type=decomposition,
                )
                points = sampler(64)
                drivers = BrownianMotion(
                    sampler, drift=theta, decomp_type=decomposition
                )._transform(points)
                historical_prices = self.model["start_price"] * np.exp(
                    (self.model["interest_rate"] - self.model["volatility"]**2 / 2)
                    * proposal.time_vec + self.model["volatility"] * drivers
                )
                historical_ratio = np.exp(-theta * drivers[..., -1] + theta**2 / 2)
                for call_put in ("CALL", "PUT"):
                    for mean in ("ARITHMETIC", "GEOMETRIC"):
                        with self.subTest(decomposition=decomposition, theta=theta,
                                          side=call_put, mean=mean):
                            option = FinancialOption(
                                sampler, proposal=proposal, decomp_type=decomposition,
                                call_put=call_put, asian_mean=mean, **self.model
                            )
                            prices, weights = option.true_measure._importance_sampling_transform_r(points)
                            np.testing.assert_allclose(prices, historical_prices, rtol=3e-14)
                            np.testing.assert_allclose(weights, historical_ratio, rtol=3e-13)
                            np.testing.assert_allclose(
                                option.f(points), option.g(historical_prices) * historical_ratio,
                                rtol=3e-12, atol=3e-13,
                            )

    def test_invalid_proposals(self):
        sampler = DigitalNetB2(4, seed=7)
        invalid = [
            ("brownian", BrownianMotion(sampler), ParameterError, "standalone"),
            ("gaussian", Gaussian(sampler), ParameterError, "standalone"),
            ("composed", GeometricBrownianMotion(Uniform(sampler)), ParameterError, "standalone"),
            ("dimension", self.make_proposal(DigitalNetB2(2, seed=7)), DimensionError, "dimensions"),
            ("horizon", self.make_proposal(sampler, t_final=2), ParameterError, "horizon"),
            ("grid", self.make_proposal(
                sampler, decomp_type="BrownianBridge", monitoring_times=[0.2, 0.5, 0.8, 1]
            ), ParameterError, "monitoring"),
            ("sampler", self.make_proposal(DigitalNetB2(4, seed=7)), ParameterError, "discrete distribution"),
            ("importance", ImportanceSampling(Uniform(sampler), Uniform(sampler)), ParameterError, "standalone"),
        ]
        for name, proposal, error, message in invalid:
            with self.subTest(case=name):
                with self.assertRaisesRegex(error, message):
                    FinancialOption(sampler, proposal=proposal, **self.model)
        with self.assertRaisesRegex(ParameterError, "discrete distribution"):
            FinancialOption(
                Uniform(sampler), proposal=self.make_proposal(sampler), **self.model
            )

    def test_no_implicit_importance_sampling(self):
        sampler = DigitalNetB2(4, seed=7)
        chained = FinancialOption(BrownianMotion(sampler))
        with self.assertRaises(ParameterError):
            chained(4)
        terminal = ImportanceSampling(Uniform(sampler), Uniform(sampler))
        with self.assertRaisesRegex(ParameterError, "cannot be used as a sampler"):
            FinancialOption(terminal)

    def test_spawn_preserves_laws_and_child_sampler(self):
        sampler = DigitalNetB2(4, seed=7, replications=3)
        proposal = self.make_proposal(
            sampler, initial_value=32, diffusion=0.36, decomp_type="BrownianBridge",
            monitoring_times=[1, 0.5, 0.25, 0.75],
        )
        option = FinancialOption(sampler, proposal=proposal, **self.model)
        children = option.spawn([0, 1])
        for level, child in enumerate(children):
            with self.subTest(level=level):
                self.assertEqual(child.level, level)
                self.assertEqual(child.d, 2 * 2**level)
                self.assertIsNot(child.sampler, sampler)
                self.assertIsNot(child.true_measure, option.true_measure)
                self.assertIsNot(child.proposal, proposal)
                self.assertIs(child.proposal.discrete_distrib, child.sampler)
                self.assertIs(child.true_measure.target.discrete_distrib, child.sampler)
                self.assertIs(child.true_measure.proposal, child.proposal)
                for field in ("initial_value", "drift", "diffusion", "decomp_type", "t"):
                    self.assertEqual(getattr(child.proposal, field), getattr(proposal, field))
                self.assertEqual(child.true_measure.target.drift, self.model["interest_rate"])
                self.assertEqual(child.true_measure.target.initial_value, self.model["start_price"])
                self.assertEqual(child.true_measure.target.diffusion, self.model["volatility"]**2)
                np.testing.assert_array_equal(
                    child.proposal.time_vec, child.true_measure.target.time_vec
                )
                self.assertEqual(child(8).shape, (2, 3, 8))
                self.assertTrue(np.isfinite(child(8)).all())

    def test_multilevel_weighted_coarse_and_fine(self):
        for level in (0, 1):
            with self.subTest(level=level):
                sampler = DigitalNetB2(2 * 2**level, seed=7, replications=3)
                proposal = self.make_proposal(sampler)
                option = FinancialOption(
                    sampler, proposal=proposal, level=level, **self.model
                )
                points = sampler(16)
                prices = proposal._transform(points)
                weights = self.reference_ratio(prices, option.true_measure.target, proposal)
                fine = option.payoff(prices) * option.discount_factor
                coarse = (
                    np.zeros_like(fine) if level == 0
                    else option.payoff(prices[..., 1::2]) * option.discount_factor
                )
                expected = np.stack([coarse, fine]) * weights
                self.assertEqual(option.f(points).shape, (2, 3, 16))
                np.testing.assert_allclose(option.f(points), expected, rtol=3e-13)

    def test_european_drifted_proposal_against_exact_value(self):
        sampler = DigitalNetB2(4, seed=7)
        proposal = self.make_proposal(sampler, drift=0.30)
        for call_put in ("CALL", "PUT"):
            with self.subTest(side=call_put):
                option = FinancialOption(
                    sampler, option="EUROPEAN", call_put=call_put,
                    proposal=proposal, **self.model,
                )
                values = option(2**14)
                self.assertTrue(np.isfinite(values).all())
                self.assertLess(abs(values.mean() - option.get_exact_value()), 0.005)
                samples, weights = option.true_measure.gen_samples(
                    64, return_weights=True
                )
                np.testing.assert_allclose(
                    option.f(sampler(64)), option.g(samples) * weights,
                    rtol=2e-14, atol=1e-14,
                )

    def test_deterministic_numerical_sanity(self):
        for mean in ("ARITHMETIC", "GEOMETRIC"):
            for call_put, theta in (("CALL", 0.5), ("PUT", -0.5)):
                with self.subTest(mean=mean, side=call_put):
                    sampler = DigitalNetB2(4, seed=7)
                    args = dict(
                        self.model, asian_mean=mean, asian_mean_quadrature_rule="RIGHT",
                        call_put=call_put,
                    )
                    ordinary = FinancialOption(sampler, **args)
                    proposal = self.make_proposal(
                        sampler, drift=self.model["interest_rate"] + self.model["volatility"] * theta
                    )
                    importance = FinancialOption(sampler, proposal=proposal, **args)
                    reference = ordinary(2**16).mean()
                    estimate = importance(2**14).mean()
                    self.assertLess(abs(estimate - reference), 0.005)
                    if mean == "GEOMETRIC":
                        exact = ordinary.get_exact_value()
                        self.assertLess(abs(reference - exact), 0.005)
                        self.assertLess(abs(estimate - exact), 0.005)


if __name__ == "__main__":
    unittest.main()
