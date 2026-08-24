"""Fast-run resume check using abs_tol
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


def _build_cases(seed=7, cont_seed=11, dimension=2, loose_tol=0.2, tight_tol=0.05, rel_tol=0, n_init=2**8, n_limit=2**24):
    iid_keister = lambda dim=dimension: Keister(IIDStdUniform(dimension=dim, seed=seed))
    lattice_keister = lambda dim=dimension: Keister(Lattice(dimension=dim, seed=seed))
    net_keister = lambda dim=dimension: Keister(DigitalNetB2(dimension=dim, seed=seed))
    rep_student_dimension = max(5, dimension)
    net_rep_keister = lambda dim=rep_student_dimension: Keister(
        DigitalNetB2(dimension=dim, replications=16, seed=seed)
    )
    bayes_lattice_keister = lambda dim=dimension: Keister(Lattice(dimension=dim, seed=seed, order="RADICAL INVERSE"))

    def iid_financial_option():
        return FinancialOption(IIDStdUniform(dimension=dimension, seed=seed), start_price=30, strike_price=30)

    def iid_financial_option_cont():
        return FinancialOption(IIDStdUniform(dimension=dimension, seed=cont_seed), start_price=30, strike_price=30)

    def qmc_financial_option():
        return FinancialOption(Lattice(replications=32, seed=seed), start_price=30, strike_price=30)

    mlmc_n_limit = max(n_limit, 2**20)

    return [
        make_tol_case("CubMCCLTVec",
            make_named_tol_builder(CubMCCLTVec, iid_keister, "abs_tol", rel_tol=rel_tol, n_init=n_init, n_limit=n_limit), 5e-2, 2.5e-2),
        make_tol_case("CubQMCLatticeG",
            make_named_tol_builder(CubQMCLatticeG, lattice_keister, "abs_tol", rel_tol=rel_tol, n_init=n_init, n_limit=2**20), 5e-3, 1e-3),
        make_tol_case("CubQMCNetG",
            make_named_tol_builder(CubQMCNetG, net_keister, "abs_tol", rel_tol=rel_tol, n_init=n_init, n_limit=2**22), 5e-3, 1e-3),
        make_tol_case("CubQMCRepStudentT",
            make_named_tol_builder(CubQMCRepStudentT, net_rep_keister, "abs_tol", rel_tol=rel_tol, n_init=2**5, n_limit=2**22), 5e-2, 1e-2),
        make_tol_case("CubQMCBayesLatticeG",
            make_named_tol_builder(CubQMCBayesLatticeG, bayes_lattice_keister, "abs_tol", rel_tol=rel_tol, n_init=2**5, n_limit=2**24), 5e-2, 5e-4),
        make_tol_case("CubBayesNetG",
            make_named_tol_builder(CubBayesNetG, net_keister, "abs_tol", rel_tol=rel_tol, n_init=2**5, n_limit=2**24), 1e-1, 3e-3),
        make_tol_case("CubMLMC",
            make_named_tol_builder(CubMLMC, iid_financial_option, "abs_tol", n_limit=mlmc_n_limit), 0.2, 0.1),
        make_tol_case("CubMLMCCont",
            make_named_tol_builder(CubMLMCCont, iid_financial_option_cont, "abs_tol", n_limit=mlmc_n_limit, n_tols=6), 0.4, 0.3),
        make_tol_case("CubMLQMC",
            make_named_tol_builder(CubMLQMC, qmc_financial_option, "abs_tol", n_limit=2**18), 0.2, 0.1),
        make_tol_case("CubMLQMCCont",
            make_named_tol_builder(CubMLQMCCont, qmc_financial_option, "abs_tol", n_limit=2**18, n_tols=6), 0.6, 0.3),
    ]


def main(verbose=False, seed=7, cont_seed=11, dimension=2):
    cases = _build_cases(seed=seed, cont_seed=cont_seed, dimension=dimension)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_rows = [run_resume_case(case, verbose=case.get("verbose", verbose)) for case in cases]
    fresh_rows  = [run_fresh_case(case, verbose=case.get("verbose", verbose)) for case in cases]

    combined_path = output_dir / "check_resume_summary.txt"
    write_combined_report(combined_path, "Stopping Criteria Check: Resume vs Fresh", resume_rows, fresh_rows)
    print(f"wrote: {combined_path}")


if __name__ == "__main__":
    main(verbose=False, seed=DEFAULT_SEED, cont_seed=DEFAULT_CONT_SEED, dimension=DEFAULT_DIMENSION)
