"""Low-discrepancy sequence choices used by the simplex demo widgets."""

from functools import partial

from qmcpy import Faure, Halton, Hammersley, KorobovLattice, Kronecker, Lattice, Sobol


ld_sequences = {
    "Sobol": {
        "LMS + digital shift": partial(Sobol, randomize="LMS DS"),
        "Digital shift": partial(Sobol, randomize="DS"),
        "LMS": partial(Sobol, randomize="LMS"),
        "Owen (NUS)": partial(Sobol, randomize="NUS"),
        "Deterministic": partial(Sobol, randomize="FALSE"),
    },
    "Lattice": {
        "Shifted": partial(Lattice, randomize="SHIFT"),
        "Deterministic": partial(Lattice, randomize="FALSE"),
    },
    "Halton": {
        "LMS + permutation": partial(Halton, randomize="LMS DP"),
        "LMS + digital shift": partial(Halton, randomize="LMS DS"),
        "LMS": partial(Halton, randomize="LMS"),
        "Permutation": partial(Halton, randomize="DP"),
        "Digital shift": partial(Halton, randomize="DS"),
        "Owen (NUS)": partial(Halton, randomize="NUS"),
        "QRNG": partial(Halton, randomize="QRNG"),
        "Deterministic": partial(Halton, randomize="FALSE"),
    },
    "Faure": {
        "LMS + permutation": partial(Faure, randomize="LMS DP"),
        "LMS + digital shift": partial(Faure, randomize="LMS DS"),
        "LMS": partial(Faure, randomize="LMS"),
        "Permutation": partial(Faure, randomize="DP"),
        "Digital shift": partial(Faure, randomize="DS"),
        "Owen (NUS)": partial(Faure, randomize="NUS"),
        "Deterministic": partial(Faure, randomize="FALSE"),
    },
    "Hammersley": {
        "Deterministic": Hammersley,
    },
    "Kronecker": {
        "CBC, shifted": partial(
            Kronecker, generating_vector="CBC", randomize="SHIFT"
        ),
        "CBC, deterministic": partial(
            Kronecker, generating_vector="CBC", randomize="FALSE"
        ),
        "Richtmyer, shifted": partial(
            Kronecker, generating_vector="RICHTMYER", randomize="SHIFT"
        ),
        "Richtmyer, deterministic": partial(
            Kronecker, generating_vector="RICHTMYER", randomize="FALSE"
        ),
        "Suzuki, shifted": partial(
            Kronecker, generating_vector="SUZUKI", randomize="SHIFT"
        ),
        "Suzuki, deterministic": partial(
            Kronecker, generating_vector="SUZUKI", randomize="FALSE"
        ),
    },
    "Korobov lattice": {
        "Shifted": partial(KorobovLattice, randomize="SHIFT"),
        "Deterministic": partial(KorobovLattice, randomize="FALSE"),
    },
}
