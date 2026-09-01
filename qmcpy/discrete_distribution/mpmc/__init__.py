"""
Message Passing Monte Carlo (MPMC) discrete distribution.

This module implements MPMC using PyTorch and PyTorch Geometric for 
generating low-discrepancy point sets through neural message passing.

Installation Requirements
--------------------------
MPMC requires PyTorch and PyTorch Geometric. Install with:

    python -m pip install "qmcpy[mpmc]"
    qmcpy-install-mpmc

For GPU support (NVIDIA CUDA), see https://pytorch.org/get-started/locally/
For torch-geometric wheels, see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

If these dependencies are not installed, attempting to use MPMC will raise an ImportError
with installation instructions. You can check availability by running:

    python -c "import torch; import pyg_lib; import torch_geometric; print('MPMC dependencies ready')"
"""

try:
    import torch
    import pyg_lib
    import torch_geometric
    from .mpmc import MPMC
except ImportError as e:
    _missing_dep = str(e)

    class MPMC(object):
        """Placeholder MPMC class shown when PyTorch dependencies are missing."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"MPMC requires PyTorch, pyg_lib, and PyTorch Geometric, but they are not installed.\n"
                f"Original error: {_missing_dep}\n\n"
                f"To use MPMC, install dependencies with:\n"
                f"  python -m pip install 'qmcpy[mpmc]'\n"
                f"  qmcpy-install-mpmc\n\n"
                f"For GPU support, see: https://pytorch.org/get-started/locally/\n"
                f"For torch-geometric installation details, see: "
                f"https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            )

__all__ = ['MPMC']
