from __future__ import annotations

from typing import TYPE_CHECKING, Union
import types
import numpy as np

if TYPE_CHECKING:
    import torch


def get_npt(x: Union[np.ndarray, torch.Tensor]) -> types.ModuleType:
    """Return the array backend module matching the input.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Array whose backend is wanted.

    Returns:
        types.ModuleType: ``numpy`` for an ``np.ndarray``, otherwise ``torch``.

    Raises:
        AssertionError: If ``x`` is neither an ``np.ndarray`` nor a ``torch.Tensor``.
    """
    if isinstance(x, np.ndarray):
        return np
    else:
        import torch

        if not (isinstance(x, torch.Tensor)):
            raise AssertionError
        return torch
