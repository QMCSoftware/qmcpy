import math
from typing import Dict, Optional

import pandas as pd


def format_number(x: float) -> str:
    """Custom formatting function to avoid unnecessary trailing zeros"""
    if pd.isna(x) or x == "-":
        return "-"
    try:
        num = float(x)
        formatted = f"{num:.8f}"
        if "." not in formatted and abs(num) < 1 and num != 0:
            formatted += ".0"
        return formatted
    except (ValueError, TypeError):
        return str(x)


def format_estimate(value: float, standard_error: float) -> str:
    """Format an estimate to the precision supported by its standard error."""
    if pd.isna(standard_error) or standard_error <= 0:
        return format_number(value)
    decimals = max(0, 1 - math.floor(math.log10(standard_error)))
    return f"{value:.{decimals}f} ({standard_error:.{decimals}f})"


def format_results_dataframe(
    df: pd.DataFrame,
    numeric_columns: list,
    uncertainty_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Format numeric columns and pair estimates with their standard errors."""
    results_formatted = df.copy()
    uncertainty_columns = uncertainty_columns or {}
    paired_columns = set(uncertainty_columns) | set(uncertainty_columns.values())
    for col in numeric_columns:
        if col in results_formatted.columns and col not in paired_columns:
            results_formatted[col] = results_formatted[col].apply(format_number)
    for estimate, standard_error in uncertainty_columns.items():
        if estimate in df.columns and standard_error in df.columns:
            results_formatted[estimate] = [
                format_estimate(value, error)
                for value, error in zip(df[estimate], df[standard_error])
            ]
    results_formatted = results_formatted.drop(
        columns=[column for column in uncertainty_columns.values() if column in df]
    )
    return results_formatted


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table with booktabs formatting"""
    # Create custom header
    header = (
        "Method & Sampler & Mean (SE) & SD (SE) & MAE (SE) & SD Error (SE) & "
        "Mean Time (s) & Time SD (s) & Speedup \\\\\n & & & & & & & & "
        "(same sampler) \\\\"
    )

    latex_table = df.style.hide(axis="index").to_latex(
        caption=caption,
        label=label,
        position="tbp",
        hrules=True,
        column_format=(
            r"ll@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}"
            r"r@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}"
            r"r@{\hspace{0.4em}}r"
        ),
    )
    # Replace default LaTeX table environment with booktabs format
    latex_table = latex_table.replace(
        "\\begin{table}[H]", "\\begin{table}[btp]\\centering"
    )
    latex_table = latex_table.replace("\\hline", "\\toprule", 1)
    latex_table = latex_table.replace("\\hline", "\\midrule", 1)
    latex_table = latex_table.replace("\\hline", "\\bottomrule")

    # Replace the generated header with custom header
    lines = latex_table.split("\n")
    for i, line in enumerate(lines):
        if "\\toprule" in line and i + 1 < len(lines):
            # Replace the line after \toprule with custom header
            lines[i + 1] = header
            break

    return "\n".join(lines)
