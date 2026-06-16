<!--
Source WordPress URL: https://qmcpy.org/2021/02/25/speeding-up-qmcpy-with-distributable-c-code/
Original metadata: Posted by Aleksei Sorokin and Jagadeeswaran Rathinavel; February 25, 2021; updated March 21, 2021.
Image handling: no content images were present in the original post.
-->

# Speeding up QMCPy with Distributable C Code

--8<-- "snippets/blog-authors/speeding-up-qmcpy-with-distributable-c-code.md"

February 25, 2021

This post explains how QMCPy uses distributable C extensions through `ctypes` and packaging metadata to speed up low-discrepancy generators.

Many Python packages rely on underlying C or C++ code to speed up their
numerical methods. For example, [NumPy](https://numpy.org/) calls C and
C++ extensions in order to speed up matrix manipulation algorithms.

Real Python's article
[*Python Bindings: Calling C or C++ From Python*](https://realpython.com/python-bindings-overview/#python-bindings-overview)
discusses a few reasons why you may want to utilize C or C++ extensions
within your Python package. Perhaps you already have a stable library in
C or C++ that you want to call from Python. Our approach in this blog
will allow the existing extension to be called from Python with only
minor code modifications. You may also be interested in speeding up your
Python code by moving it to a compiled language that can optimize
subroutines. For example, in QMCPy we have found that many low
discrepancy sequence generators are significantly faster when
implemented in C.

So why not implement everything in C or C++? In our experience, Python
delivers a convenience, readability, and community engagement that allow
for rapid development, testing, and distribution to a large audience of
active users.

While the benefits of moving certain modules to C/C++ have been well
documented, implementing these extensions to play nicely with your
existing Python codebase can often be quite tricky. Moreover, writing
extensions for platform-independent distribution with PyPI, so someone
can `pip install yourPackage`, can be even more challenging. In this
blog post we share how we developed the QMCPy package [1] to be
platform agnostic while utilizing C extensions.

## C

When you first explore writing C/C++ extensions you will likely come
across
[Python's recommended solution](https://docs.python.org/3/extending/extending.html).
This approach requires a good bit of boilerplate code and prohibits
plug-and-play of an existing C/C++ library. QMCPy's approach uses the
[`ctypes`](https://docs.python.org/3/library/ctypes.html) library to
call a C function with a few lines of Python defining the arguments and
return values of the compiled function.

Let us now turn to an example from our QMCPy library. Based on Art
Owen's work in [2], we wrote the below Halton generator in C. Note that
the implementation does not contain all the boilerplate code of a native
Python solution, but instead may be used as a standalone C file.

```c
#include "MRG63k3a.h"
static int primes[1000] = {2, 3, 5, 7, ...};
EXPORT void halton_owen(int n, int d, int n0, int d0,
                        int randomize, double *ans, long long seed){
    seed_MRG63k3a(seed); ...}
```

A few important notes about the above code are the use of `#include`,
`EXPORT`, and `long long`. Depending on the compiler, such as `gcc` or
Windows `cl.exe`, `EXPORT` allows us to expose a function, in the above
case `halton_owen`, so that the Python code can invoke it. When you
`EXPORT` a function it makes the C code available to `ctypes`. The
Halton generator utilizes the MRG63k3a random number generator [3],
which is stored in a separate file. We can call this function by
creating a `.h` file that defines the external function we wish to call.
In `MRG63k3a.h` we define the `seed_MRG63k3a` method which is then
included and used in the above Halton generator.

When your Python package is installed, the C compiler that builds the
extensions is platform-specific. We found that the `gcc` compiler uses
8 bytes to store a `long` while Windows `cl.exe` uses only 4. As a
workaround, we suggest using the `long long` datatype, which is 8 bytes
for both `gcc` and `cl.exe`. A nice way to verify you are using `gcc`
and debug these cross-language problems is to intentionally trigger
compiler errors.

With these three C files, the Halton generator, MRG63k3a, and MRG63k3a's
header, we are ready to call our function from Python.

## Python Code

First, we will use `ctypes` to define our function from Python. `ctypes`
requires that we define the arguments and return values of our Halton
function in order for it to be treated like a native Python function.
Below is an example of how to set up and call our Halton function in C
using Python.

```python
import ctypes
from ctypes import CDLL, RTLD_GLOBAL
from os.path import dirname, abspath
from glob import glob
from numpy import *

# load the library
c_lib = CDLL(
    glob(dirname(abspath(__file__))+'/c_lib*')[0],
    mode = RTLD_GLOBAL)

# define the function arguments
halton_cf = c_lib.halton_owen
halton_cf.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int, # n0
    ctypes.c_int, # d0
    ctypes.c_int, # randomize
    ctypeslib.ndpointer( # result array
        ctypes.c_double,
        flags = 'C_CONTIGUOUS'),
    ctypes.c_long]  # seed

# define the return value
halton_cf.restype = None

# example call to the function
#   create an empty array to fill with Halton points
x = zeros((5,3), dtype=double)
#   fill the array with 5, 3-dimensional Halton points
halton_cf(5, 3, 0, 0, True, x, 17)
```

The second piece of Python code you will need is a `setup.py`. The
`setup.py` file defines the C extensions of your package and helps
prepare your package for distribution on PyPI. While it is possible to
compile and call your extension function without a `setup.py` file, we
found this method to be the easiest and most straightforward for package
distribution.

Below is a snippet from our `setup.py` file that defines the extensions,
packages, and other metadata. Note that we use the
[`setuptools`](https://setuptools.readthedocs.io/en/latest/) package to
easily define our distribution properties, although
[`distutils`](https://docs.python.org/3/library/distutils.html) may also
be used.

```python
import setuptools
from setuptools import Extension

# define package API
packages = [
    'qmcpy',
    'qmcpy.discrete_distribution',
    'qmcpy.discrete_distribution.halton']

setuptools.setup(
    name="qmcpy",
    packages=packages,
    include_package_data=True,
    ext_modules=[
        Extension(
            name='qmcpy.discrete_distribution.c_lib.c_lib',
            sources=[
                'qmcpy/discrete_distribution/c_lib/halton_owen.c',
                'qmcpy/discrete_distribution/c_lib/MRG63k3a.c'],
            )],)
```

When distributing your package on PyPI, you may come across an error
regarding missing `.h` files, e.g., the `MRG63k3a.h` file mentioned
earlier. For this file to be included in your distribution, you need to
include a `MANIFEST.in` file that defines all the non-Python and non-C
code to be included. That way the `.h` files and other files will be
included in your package distribution. We provide a sample from our
`MANIFEST.in` below.

```text
include qmcpy/discrete_distribution/c_lib/*.h
include qmcpy/discrete_distribution/sobol/generating_matrices/*.npy
include qmcpy/discrete_distribution/lattice/generating_vectors/*.npy
```

## References

1. Choi, S.-C. T., Hickernell, F., McCourt, M., & Sorokin, A. QMCPy: A
   quasi-Monte Carlo Python Library.
   [https://qmcsoftware.github.io/QMCSoftware/](https://qmcsoftware.github.io/QMCSoftware/).
   2020.
2. Owen, A. B. A randomized Halton algorithm in R. 2017.
   [arXiv:1706.02808 [stat.CO]](https://arxiv.org/abs/1706.02808).
3. L'Ecuyer, P. Good parameters and implementations for combined
   multiple recursive random number generators. *Operations Research*,
   47, 159-164.
   [https://pubsonline.informs.org/doi/abs/10.1287/opre.47.1.159](https://pubsonline.informs.org/doi/abs/10.1287/opre.47.1.159).
   1999.
