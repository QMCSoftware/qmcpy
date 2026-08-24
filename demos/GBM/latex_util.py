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


def format_results_dataframe(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """Apply custom formatting to numeric columns in results dataframe"""
    results_formatted = df.copy()
    for col in numeric_columns:
        if col in results_formatted.columns:
            results_formatted[col] = results_formatted[col].apply(format_number)
    return results_formatted


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table with booktabs formatting"""
    # Create custom header
    header = (
        "Method & Sampler & Mean & Std Dev & Mean Absolute & Std Dev  & "
        "Mean Time (s)  & Std Dev (s) & Speedup \\\\\n &  &  &   &  "
        "Error &  Error &   & &  \\\\"
    )

    latex_table = df.style.hide(axis="index").to_latex(
        caption=caption,
        label=label,
        position="tbp",
        hrules=True,
        column_format=(
            "ll@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}"
            "r@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}"
            "r@{\hspace{0.4em}}r"
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
