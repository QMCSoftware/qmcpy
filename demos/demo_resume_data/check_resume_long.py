"""Long-run resume check using rel_tol or rmse_tol
"""

from pathlib import Path
from qmcpy import (
    CubBayesNetG,
    CubMCCLTVec,
    CubMLMC,
    CubMLMCCont,
    CubMLQMC,
    CubMLQMCCont,
    CubQMCBayesLatticeG,
    CubQMCLatticeG,
    CubQMCNetG,
    CubQMCRepStudentT,
    DigitalNetB2,
    FinancialOption,
    IIDStdUniform,
    Keister,
    Lattice,
)
from resume_util import make_named_tol_builder, make_tol_case, run_fresh_case, run_resume_case, write_combined_report

DEFAULT_SEED = 7
DEFAULT_CONT_SEED = 11
DEFAULT_DIMENSION = 2


def _build_cases(seed=DEFAULT_SEED, cont_seed=DEFAULT_CONT_SEED, dimension=DEFAULT_DIMENSION):
    iid_keister = lambda: Keister(IIDStdUniform(dimension=dimension, seed=seed))
    lattice_keister = lambda: Keister(Lattice(dimension=dimension, seed=seed))
    net_keister = lambda: Keister(DigitalNetB2(dimension=dimension, seed=seed))
    net_rep_keister_long = lambda: Keister(DigitalNetB2(dimension=max(8, dimension), replications=16, seed=seed))
    bayes_lattice_keister = lambda: Keister(Lattice(dimension=dimension, seed=seed, order="RADICAL INVERSE"))

    clt_vec_builder = make_named_tol_builder(
        CubMCCLTVec, iid_keister, "rel_tol",
        abs_tol=0, n_init=2**8, n_limit=2**22,
    )
    lattice_builder = make_named_tol_builder(
        CubQMCLatticeG, lattice_keister, "rel_tol",
        abs_tol=0, n_init=2**8, n_limit=2**24,
    )
    net_builder = make_named_tol_builder(
        CubQMCNetG, net_keister, "rel_tol",
        abs_tol=0, n_init=2**8, n_limit=2**24,
    )
    rep_student_builder = make_named_tol_builder(
        CubQMCRepStudentT, net_rep_keister_long, "rel_tol",
        abs_tol=0, n_init=2**5, n_limit=2**24,
    )
    bayes_lattice_builder = make_named_tol_builder(
        CubQMCBayesLatticeG, bayes_lattice_keister, "rel_tol",
        abs_tol=0, n_init=2**6, n_limit=2**24,
    )
    bayes_net_builder = make_named_tol_builder(
        CubBayesNetG, net_keister, "rel_tol",
        abs_tol=0, n_init=2**6, n_limit=2**22,
    )

    def iid_financial_option_4d():
        return FinancialOption(IIDStdUniform(dimension=4, seed=seed), start_price=300, strike_price=100)

    mlmc_builder = make_named_tol_builder(
        CubMLMC, iid_financial_option_4d, "rmse_tol",
        n_limit=2**24,
    )

    def qmc_financial_option_4d():
        return FinancialOption(Lattice(replications=32, seed=seed, dimension=4), start_price=300, strike_price=100)

    mlqmc_builder = make_named_tol_builder(
        CubMLQMC, qmc_financial_option_4d, "rmse_tol",
        n_limit=2**24,
    )

    def iid_financial_option_cont_large():
        return FinancialOption(IIDStdUniform(dimension=4, seed=cont_seed), start_price=300, strike_price=100)

    def qmc_financial_option_cont_large():
        return FinancialOption(Lattice(replications=32, seed=cont_seed, dimension=4), start_price=300, strike_price=100)

    mlmc_cont_builder = make_named_tol_builder(
        CubMLMCCont, iid_financial_option_cont_large, "rmse_tol",
        n_tols=1200, inflate=1.001, n_limit=2**24,
    )

    mlqmc_cont_builder = make_named_tol_builder(
        CubMLQMCCont, qmc_financial_option_cont_large, "rmse_tol",
        n_tols=1200, inflate=1.001, n_limit=2**24,
    )

    return [
        make_tol_case("CubMCCLTVec",         clt_vec_builder,         loose_tol=1e-2,  tight_tol=1e-3),
        make_tol_case("CubQMCLatticeG",      lattice_builder,         loose_tol=1e-2,  tight_tol=2e-6),
        make_tol_case("CubQMCNetG",          net_builder,             loose_tol=1e-3,  tight_tol=1e-6),
        make_tol_case("CubQMCRepStudentT",   rep_student_builder,     loose_tol=1e-3,  tight_tol=5e-6),
        make_tol_case("CubQMCBayesLatticeG", bayes_lattice_builder,   loose_tol=1e-3,  tight_tol=1e-9),
        make_tol_case("CubBayesNetG",        bayes_net_builder,       loose_tol=1e-3,  tight_tol=5e-6),
        make_tol_case("CubMLMC",             mlmc_builder,            loose_tol=1.0,   tight_tol=0.05),
        make_tol_case("CubMLQMC",            mlqmc_builder,           loose_tol=1.0,   tight_tol=0.05),
        make_tol_case("CubMLMCCont",         mlmc_cont_builder,       loose_tol=1.0,   tight_tol=0.5),
        make_tol_case("CubMLQMCCont",        mlqmc_cont_builder,      loose_tol=1.0,   tight_tol=0.5),
    ]


def main(verbose=False, seed=DEFAULT_SEED, cont_seed=DEFAULT_CONT_SEED, dimension=DEFAULT_DIMENSION):
    cases = _build_cases(seed=seed, cont_seed=cont_seed, dimension=dimension)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_rows = [run_resume_case(case, verbose=case.get("verbose", verbose)) for case in cases]
    fresh_rows = [run_fresh_case(case, verbose=case.get("verbose", verbose)) for case in cases]

    combined_path = output_dir / "check_resume_long_summary.txt"
    write_combined_report(
        combined_path,
        "Stopping Criteria Long-Run Check: Resume vs Fresh (ML iters may differ)",
        resume_rows,
        fresh_rows,
    )
    print(f"wrote: {combined_path}")


if __name__ == "__main__":
    main(verbose=False)
