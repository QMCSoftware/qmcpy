---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
---

# Discrete Distributions

## UML Overview

![](./umls/discrete_distribution_overview.svg)

## `AbstractDiscreteDistribution`

::: qmcpy.discrete_distribution.abstract_discrete_distribution.AbstractDiscreteDistribution

## `DigitalNetB2`

::: qmcpy.discrete_distribution.digital_net_b2.digital_net_b2.DigitalNetB2

## `Lattice`

::: qmcpy.discrete_distribution.lattice.Lattice

## `Halton`

::: qmcpy.discrete_distribution.digital_net_any_bases.halton.Halton

## `Faure`

::: qmcpy.discrete_distribution.digital_net_any_bases.faure.Faure

## `DigitalNetAnyBases`

::: qmcpy.discrete_distribution.digital_net_any_bases.DigitalNetAnyBases

## `Kronecker`

::: qmcpy.discrete_distribution.kronecker.Kronecker

## `DummySampler`

::: qmcpy.discrete_distribution.dummy_sampler.DummySampler

## `IIDStdUniform`

::: qmcpy.discrete_distribution.iid_std_uniform.IIDStdUniform

## `MPMC: Message Passing Monte Carlo`

MPMC requires PyTorch and PyTorch Geometric. Install with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyg_lib torch-geometric
```

For GPU support or platform-specific wheels, see the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

::: qmcpy.discrete_distribution.mpmc.mpmc.MPMC

## UML Specific

![](./umls/discrete_distribution_specific.svg)
