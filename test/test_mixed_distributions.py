import numpy as np
import pytest

from qmcpy import (
    AbstractTrueMeasure,
    DigitalNetB2,
    Gaussian,
    Kumaraswamy,
    Mixture,
)
from qmcpy.util import DimensionError, MethodImplementationError, ParameterError


class TransformOnlyMeasure(AbstractTrueMeasure):
    def __init__(self, sampler):
        self.parameters = []
        self.domain = np.array([[0.0, 1.0]])
        self.range = np.array([[0.0, 1.0]])
        self._parse_sampler(sampler)
        super(TransformOnlyMeasure, self).__init__()

    def _transform(self, x):
        return x


def gaussian_component(dimension, mean):
    return Gaussian(
        DigitalNetB2(dimension, seed=17),
        mean=mean,
        covariance=np.eye(dimension),
    )


def test_two_component_gaussian_mixture_shape():
    components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
    mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])

    samples = mixture(16)

    assert mixture.d == 1
    assert mixture.discrete_distrib.d == 2
    assert samples.shape == (16, 1)


def test_component_selection_at_cumulative_boundaries_preserves_order():
    components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
    mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])
    just_above_boundary = np.nextafter(0.3, 1.0)
    u = np.array(
        [
            [0.0, 0.5],
            [0.9, 0.5],
            [0.3, 0.5],
            [just_above_boundary, 0.5],
            [0.1, 0.5],
            [1.0, 0.5],
        ]
    )

    samples = mixture._transform(u)

    np.testing.assert_allclose(
        samples[:, 0], [-2.0, 3.0, -2.0, 3.0, -2.0, 3.0]
    )


def test_composed_component_applies_full_recursive_transform():
    inner = Kumaraswamy(DigitalNetB2(1, seed=19), a=2.0, b=3.0)
    composed = Gaussian(inner, mean=-2.0, covariance=1.0)
    direct = gaussian_component(1, 3.0)
    mixture = Mixture(DigitalNetB2(2, seed=7), [composed, direct], [0.5, 0.5])
    u = np.array([[0.25, 0.2], [0.75, 0.8]])

    samples = mixture._transform(u)
    expected_composed = composed._transform(inner._transform(u[:1, 1:]))
    expected_direct = direct._transform(u[1:, 1:])

    np.testing.assert_allclose(samples[:1], expected_composed)
    np.testing.assert_allclose(samples[1:], expected_direct)
    assert not np.allclose(expected_composed, composed._transform(u[:1, 1:]))


def test_multiple_components():
    components = [
        gaussian_component(1, -4.0),
        gaussian_component(1, 0.0),
        gaussian_component(1, 5.0),
    ]
    mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.2, 0.3, 0.5])
    u = np.array([[0.2, 0.5], [0.4, 0.5], [0.5, 0.5], [0.8, 0.5]])

    samples = mixture._transform(u)

    np.testing.assert_allclose(samples[:, 0], [-4.0, 0.0, 0.0, 5.0])


def test_one_component_mixture_is_valid():
    component = gaussian_component(1, 1.5)
    mixture = Mixture(DigitalNetB2(2, seed=7), [component], [1.0])
    u = np.array([[0.0, 0.5], [0.4, 0.5], [1.0, 0.5]])

    samples = mixture._transform(u)

    assert mixture(4).shape == (4, 1)
    np.testing.assert_allclose(samples[:, 0], 1.5)


@pytest.mark.parametrize(
    "probabilities",
    [[0.0, 1.0], [-0.1, 1.1], [np.nan, np.nan], [np.inf, 0.5]],
)
def test_invalid_probabilities(probabilities):
    components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

    with pytest.raises(ParameterError, match="positive and finite"):
        Mixture(DigitalNetB2(2, seed=7), components, probabilities)


def test_probabilities_must_sum_to_one():
    components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

    with pytest.raises(ParameterError, match="sum to 1"):
        Mixture(DigitalNetB2(2, seed=7), components, [0.2, 0.7])


def test_number_of_probabilities_must_match_components():
    components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

    with pytest.raises(ParameterError, match="one probability per component"):
        Mixture(DigitalNetB2(2, seed=7), components, [1.0])


def test_requires_at_least_one_component():
    with pytest.raises(ParameterError, match="nonempty list of components"):
        Mixture(DigitalNetB2(2, seed=7), [], [])


def test_components_must_be_true_measures():
    components = [gaussian_component(1, 0.0), object()]

    with pytest.raises(ParameterError, match="AbstractTrueMeasure"):
        Mixture(DigitalNetB2(2, seed=7), components, [0.5, 0.5])


def test_sampler_must_be_discrete_distribution():
    components = [gaussian_component(1, 0.0)]

    with pytest.raises(ParameterError, match="AbstractDiscreteDistribution"):
        Mixture(object(), components, [1.0])


def test_probabilities_must_be_numeric():
    components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

    with pytest.raises(ParameterError, match="numeric"):
        Mixture(DigitalNetB2(2, seed=7), components, ["left", "right"])


def test_probabilities_must_be_one_dimensional():
    components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

    with pytest.raises(ParameterError, match="one probability per component"):
        Mixture(DigitalNetB2(2, seed=7), components, [[0.5, 0.5]])


def test_component_dimensions_must_match():
    components = [gaussian_component(1, 0.0), gaussian_component(2, [0.0, 1.0])]

    with pytest.raises(DimensionError, match="same output dimension"):
        Mixture(DigitalNetB2(2, seed=7), components, [0.5, 0.5])


def test_sampler_dimension_must_be_component_dimension_plus_one():
    components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

    with pytest.raises(DimensionError, match="component dimension plus one"):
        Mixture(DigitalNetB2(1, seed=7), components, [0.5, 0.5])


def test_transform_input_dimension_is_validated():
    mixture = Mixture(
        DigitalNetB2(2, seed=7),
        [gaussian_component(1, 0.0)],
        [1.0],
    )

    with pytest.raises(DimensionError, match="expected last axis 2"):
        mixture._transform(np.zeros((3, 1)))


def test_weight_input_dimension_is_validated():
    mixture = Mixture(
        DigitalNetB2(2, seed=7),
        [gaussian_component(1, 0.0)],
        [1.0],
    )

    with pytest.raises(DimensionError, match="expected last axis 1"):
        mixture._weight(np.zeros((3, 2)))


def test_weight_is_weighted_sum_of_component_weights():
    components = [gaussian_component(1, -1.0), gaussian_component(1, 2.0)]
    probabilities = np.array([0.3, 0.7])
    mixture = Mixture(DigitalNetB2(2, seed=7), components, probabilities)
    x = np.array([[-2.0], [0.0], [1.5], [4.0]])

    expected = sum(
        probability * component._weight(x)
        for probability, component in zip(probabilities, components)
    )

    np.testing.assert_allclose(mixture._weight(x), expected)


def test_component_weight_failure_propagates():
    transform_only = TransformOnlyMeasure(DigitalNetB2(1, seed=23))
    mixture = Mixture(
        DigitalNetB2(2, seed=7),
        [gaussian_component(1, 0.0), transform_only],
        [0.5, 0.5],
    )

    with pytest.raises(MethodImplementationError, match="TransformOnlyMeasure"):
        mixture._weight(np.array([[0.5]]))


def test_spawn_replaces_outer_sampler_and_preserves_components():
    components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
    mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])

    spawned = mixture.spawn(s=2)
    explicit_same_dimension = mixture.spawn(s=1, dimensions=[1])[0]

    for child in spawned + [explicit_same_dimension]:
        assert isinstance(child, Mixture)
        assert child.d == 1
        assert child.discrete_distrib.d == 2
        assert child.discrete_distrib is not mixture.discrete_distrib
        assert all(
            child_component is parent_component
            for child_component, parent_component in zip(child.components, components)
        )
        assert child(4).shape == (4, 1)

    with pytest.raises(DimensionError, match="preserves the component dimension"):
        mixture.spawn(s=1, dimensions=2)


def test_spawn_validates_count_and_dimensions_length():
    mixture = Mixture(
        DigitalNetB2(2, seed=7),
        [gaussian_component(1, 0.0)],
        [1.0],
    )

    with pytest.raises(ParameterError, match="s>0"):
        mixture.spawn(s=0)
    with pytest.raises(ParameterError, match="length s"):
        mixture.spawn(s=2, dimensions=[1])


def test_replicated_sampler_shape_and_selection():
    components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
    mixture = Mixture(
        DigitalNetB2(2, seed=7, replications=3), components, [0.3, 0.7]
    )

    samples = mixture(8)
    manual_u = np.array(
        [
            [[0.1, 0.5], [0.9, 0.5]],
            [[0.3, 0.5], [np.nextafter(0.3, 1.0), 0.5]],
        ]
    )
    manual_samples = mixture._transform(manual_u)

    assert samples.shape == (3, 8, 1)
    assert manual_samples.shape == (2, 2, 1)
    np.testing.assert_allclose(manual_samples[..., 0], [[-2.0, 3.0], [-2.0, 3.0]])
