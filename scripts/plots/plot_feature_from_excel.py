from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import RESULTS_DIR
from scripts.plots.group_comparison_plot import barplot_mean_se


def _to_numeric_values(series: pd.Series) -> list[float]:
    return pd.to_numeric(series, errors="coerce").dropna().tolist()


def read_groups_from_excel(excel_path: Path) -> tuple[list[float], list[float], pd.DataFrame]:
    """Read Saline/Ghrelin values from a single-sheet, two-column Excel file."""
    df = pd.read_excel(excel_path)

    if len(df.columns) < 2:
        raise ValueError(
            f"Expected at least two columns in {excel_path.name} (Saline, Ghrelin). "
            f"Found: {df.columns.tolist()}"
        )

    saline_values = _to_numeric_values(df.iloc[:, 0])
    ghrelin_values = _to_numeric_values(df.iloc[:, 1])
    summary_df = pd.DataFrame(
        {
            "group": ["Saline"] * len(saline_values) + ["Ghrelin"] * len(ghrelin_values),
            "value": saline_values + ghrelin_values,
        }
    )
    return saline_values, ghrelin_values, summary_df


def _plot_bar(
    saline_values: list[float],
    ghrelin_values: list[float],
    *,
    feature_name: str,
    test: str,
) -> plt.Axes:
    ax = barplot_mean_se(
        saline_values,
        ghrelin_values,
        labels=["Saline", "Ghrelin"],
        ylabel=f"Mean ± SE {feature_name}",
        test=test,
        show_points=True,
    )
    ax.set_title(f"{feature_name}: Saline vs Ghrelin")
    return ax


def _plot_box(
    saline_values: list[float],
    ghrelin_values: list[float],
    *,
    feature_name: str,
) -> plt.Axes:
    _, ax = plt.subplots()
    data = [saline_values, ghrelin_values]
    positions = [0, 1]

    ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    colors = ["tab:blue", "tab:orange"]
    rng = np.random.default_rng(0)
    for i, values in enumerate(data):
        x_jitter = rng.normal(positions[i], 0.05, size=len(values))
        ax.scatter(x_jitter, values, color=colors[i], alpha=0.7, edgecolor="k", linewidth=0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Saline", "Ghrelin"])
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name}: Saline vs Ghrelin")
    return ax


def plot_feature_from_excel(
    excel_path: Path,
    *,
    feature_name: str = "feature",
    plot_kind: str = "bar",
    test: str = "welch",
    outdir: Path | None = None,
) -> tuple[Path, Path]:
    """Read one Excel file and save a summary xlsx + barplot pdf."""
    if outdir is None:
        outdir = RESULTS_DIR / "feature_barplots"

    group1, group2, summary_df = read_groups_from_excel(excel_path)
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("Both groups must contain at least one numeric value.")

    if plot_kind == "box":
        ax = _plot_box(group1, group2, feature_name=feature_name)
    else:
        ax = _plot_bar(group1, group2, feature_name=feature_name, test=test)

    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    slug = feature_name.strip().lower().replace(" ", "_")
    summary_path = outdir / f"{slug}_summary.xlsx"
    plot_path = outdir / f"{slug}_barplot.pdf"

    summary_df.to_excel(summary_path, index=False)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return summary_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read feature values from xlsx and reproduce Saline vs Ghrelin barplots."
    )
    parser.add_argument("excel_file", type=Path, help="Input .xlsx file.")
    parser.add_argument("--feature-name", default="feature", help="Feature name used in output names/title.")
    parser.add_argument("--plot-kind", choices=["bar", "box"], default="bar")
    parser.add_argument(
        "--test",
        choices=["welch", "mann_whitney"],
        default="welch",
    )
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR / "feature_barplots")

    args = parser.parse_args()
    if not args.excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel_file}")

    summary_path, plot_path = plot_feature_from_excel(
        args.excel_file,
        feature_name=args.feature_name,
        plot_kind=args.plot_kind,
        test=args.test,
        outdir=args.outdir,
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()