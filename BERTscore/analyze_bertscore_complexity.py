#!/usr/bin/env python3
"""Reproduce the BERTScore complexity analyses reported in the manuscript.

Place this script in the same folder as the four model Excel files, then run:
    python analyze_bertscore_complexity.py

Outputs:
    bertscore_complexity_analysis_results.xlsx
    Figure6_BERTScore_complexity.png
    Figure6_BERTScore_complexity.pdf

The script computes:
1. Figure 6: scores averaged across the four models for each question, with
   question-level mean +/- 1 SE and paired exact Wilcoxon tests by complexity.
2. Table 2: pairwise comparisons of question-level mean RAG gains across
   complexity groups using two-sided Mann-Whitney U tests and Holm correction.
3. Table B1: model-specific paired LLM vs. RA-LLM comparisons by complexity,
   using two-sided exact Wilcoxon signed-rank tests and within-model Holm correction.
4. Table B2: model-specific Kruskal-Wallis omnibus tests across complexity
   levels (Holm correction across models) and pairwise Mann-Whitney U tests
   (Holm correction within each model).
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(__file__).resolve().parent
COMPLEXITY_ORDER = ["Low", "Middle", "High"]
PAIR_ORDER = [("Low", "Middle"), ("Low", "High"), ("Middle", "High")]

# The script identifies files by model name, so the suffix in the filenames
# may change (e.g., (3), (4)) without requiring code edits.
MODEL_PATTERNS = {
    "Baichuan2-13B": ["baichuan2-13b"],
    "GLM3-6B": ["glm3-6b"],
    "Qwen2.5-14B": ["qwen2.5-14b"],
    "Qwen2.5-7B": ["qwen2-5-7b", "qwen2.5-7b"],
}

REQUIRED_COLUMNS = ["No", "Complexity", "llm_bertscore_r", "rag_bertscore_r"]


def locate_input_file(model_name, patterns):
    """Find the model workbook in the script's directory."""
    candidates = [p for p in BASE_DIR.glob("*.xlsx") if not p.name.startswith("bertscore_complexity_analysis_results")]
    matches = []
    for path in candidates:
        lower_name = path.name.lower()
        if any(pattern in lower_name for pattern in patterns):
            matches.append(path)
    if len(matches) != 1:
        names = ", ".join(p.name for p in matches) if matches else "none"
        raise FileNotFoundError(
            f"Expected one workbook for {model_name}; found {len(matches)} ({names}). "
            f"Place the four source workbooks in: {BASE_DIR}"
        )
    return matches[0]


def read_model_data(model_name, path):
    """Read and validate one model's question-level BERTScore data."""
    df = pd.read_excel(path)
    df.columns = [str(col).strip() for col in df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["No"] = df["No"].astype(str).str.strip()
    df["Complexity"] = df["Complexity"].astype(str).str.strip().str.title()
    df["llm_bertscore_r"] = pd.to_numeric(df["llm_bertscore_r"], errors="coerce")
    df["rag_bertscore_r"] = pd.to_numeric(df["rag_bertscore_r"], errors="coerce")
    df["Model"] = model_name

    if df["No"].duplicated().any():
        duplicated = df.loc[df["No"].duplicated(), "No"].tolist()
        raise ValueError(f"Duplicate question IDs in {path.name}: {duplicated}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        missing_rows = df[df[REQUIRED_COLUMNS].isna().any(axis=1)]["No"].tolist()
        raise ValueError(f"Missing IDs/complexity/scores in {path.name}: {missing_rows}")

    unknown_complexity = sorted(set(df["Complexity"]) - set(COMPLEXITY_ORDER))
    if unknown_complexity:
        raise ValueError(f"Unexpected complexity labels in {path.name}: {unknown_complexity}")

    counts = df["Complexity"].value_counts().to_dict()
    expected_counts = {"Low": 20, "Middle": 20, "High": 20}
    if counts != expected_counts:
        warnings.warn(
            f"{path.name} has complexity counts {counts}; expected {expected_counts}. "
            "Analyses will continue with the observed data."
        )

    df["gain"] = df["rag_bertscore_r"] - df["llm_bertscore_r"]
    return df


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in their original order."""
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return p_values
    return multipletests(p_values, method="holm")[1]


def format_p(p_value):
    """Compact p-value string for the console; Excel keeps numeric values."""
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def build_pooled_question_data(model_data):
    """Average each model's scores within question before group summaries."""
    combined = pd.concat(model_data.values(), ignore_index=True)

    # Every model must contain the same questions and complexity assignment.
    id_check = combined.groupby("No")["Model"].nunique()
    if not (id_check == len(model_data)).all():
        bad_ids = id_check[id_check != len(model_data)].index.tolist()
        raise ValueError(f"Question IDs are not present in all four models: {bad_ids}")

    complexity_check = combined.groupby("No")["Complexity"].nunique()
    if not (complexity_check == 1).all():
        bad_ids = complexity_check[complexity_check != 1].index.tolist()
        raise ValueError(f"Complexity labels differ across model files for: {bad_ids}")

    pooled = (
        combined.groupby(["No", "Complexity"], as_index=False)
        .agg(
            llm_bertscore=("llm_bertscore_r", "mean"),
            rag_bertscore=("rag_bertscore_r", "mean"),
        )
    )
    pooled["gain"] = pooled["rag_bertscore"] - pooled["llm_bertscore"]
    pooled["Complexity"] = pd.Categorical(
        pooled["Complexity"], categories=COMPLEXITY_ORDER, ordered=True
    )
    pooled = pooled.sort_values(["Complexity", "No"]).reset_index(drop=True)
    return pooled


def analyze_figure6(pooled):
    """Compute pooled means/SEs and paired Wilcoxon tests by complexity."""
    rows = []
    for complexity in COMPLEXITY_ORDER:
        group = pooled[pooled["Complexity"] == complexity]
        n = len(group)
        test = wilcoxon(
            group["rag_bertscore"],
            group["llm_bertscore"],
            alternative="two-sided",
            method="exact",
        )
        rows.append(
            {
                "Complexity": complexity,
                "n_questions": n,
                "LLM_mean": group["llm_bertscore"].mean(),
                "LLM_SD": group["llm_bertscore"].std(ddof=1),
                "LLM_SE": group["llm_bertscore"].std(ddof=1) / np.sqrt(n),
                "RA_LLM_mean": group["rag_bertscore"].mean(),
                "RA_LLM_SD": group["rag_bertscore"].std(ddof=1),
                "RA_LLM_SE": group["rag_bertscore"].std(ddof=1) / np.sqrt(n),
                "Mean_gain": group["gain"].mean(),
                "Wilcoxon_statistic": test.statistic,
                "p_raw": test.pvalue,
            }
        )

    result = pd.DataFrame(rows)
    result["p_holm_across_complexity"] = holm_adjust(result["p_raw"].to_numpy())
    result["figure_star_raw_p"] = result["p_raw"].map(significance_stars)
    return result


def analyze_table2(pooled):
    """Pairwise tests of pooled question-level RAG gains across complexity."""
    rows = []
    for group_a, group_b in PAIR_ORDER:
        x = pooled.loc[pooled["Complexity"] == group_a, "gain"].astype(float)
        y = pooled.loc[pooled["Complexity"] == group_b, "gain"].astype(float)
        test = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        rows.append(
            {
                "Comparison": f"{group_a} vs. {group_b}",
                "Mean_gain_group_1": x.mean(),
                "Mean_gain_group_2": y.mean(),
                "Difference_in_mean_gain": x.mean() - y.mean(),
                "Mann_Whitney_U": test.statistic,
                "p_raw": test.pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].to_numpy())
    return result


def analyze_table_b1(model_data):
    """Model-specific paired tests of LLM vs RA-LLM by complexity."""
    rows = []
    for model_name, df in model_data.items():
        raw_p = []
        model_rows = []
        for complexity in COMPLEXITY_ORDER:
            group = df[df["Complexity"] == complexity]
            test = wilcoxon(
                group["rag_bertscore_r"],
                group["llm_bertscore_r"],
                alternative="two-sided",
                method="exact",
            )
            row = {
                "Model": model_name,
                "Complexity": complexity,
                "n_questions": len(group),
                "LLM_mean": group["llm_bertscore_r"].mean(),
                "LLM_SD": group["llm_bertscore_r"].std(ddof=1),
                "RA_LLM_mean": group["rag_bertscore_r"].mean(),
                "RA_LLM_SD": group["rag_bertscore_r"].std(ddof=1),
                "Delta_score": group["gain"].mean(),
                "Wilcoxon_statistic": test.statistic,
                "p_raw": test.pvalue,
            }
            model_rows.append(row)
            raw_p.append(test.pvalue)

        adjusted = holm_adjust(raw_p)
        for row, p_adj in zip(model_rows, adjusted):
            row["p_holm_within_model"] = p_adj
            rows.append(row)

    return pd.DataFrame(rows)


def analyze_table_b2(model_data):
    """Kruskal-Wallis omnibus tests and within-model pairwise comparisons."""
    omnibus_rows = []
    pairwise_rows = []

    for model_name, df in model_data.items():
        groups = [df.loc[df["Complexity"] == c, "gain"].astype(float) for c in COMPLEXITY_ORDER]
        kw = kruskal(*groups)
        omnibus_rows.append(
            {
                "Model": model_name,
                "Kruskal_Wallis_statistic": kw.statistic,
                "p_raw": kw.pvalue,
            }
        )

        model_pair_rows = []
        pair_p = []
        for group_a, group_b in PAIR_ORDER:
            x = df.loc[df["Complexity"] == group_a, "gain"].astype(float)
            y = df.loc[df["Complexity"] == group_b, "gain"].astype(float)
            test = mannwhitneyu(x, y, alternative="two-sided", method="auto")
            row = {
                "Model": model_name,
                "Comparison": f"{group_a} vs. {group_b}",
                "Mean_gain_group_1": x.mean(),
                "Mean_gain_group_2": y.mean(),
                "Difference_in_mean_gain": x.mean() - y.mean(),
                "Mann_Whitney_U": test.statistic,
                "p_raw": test.pvalue,
            }
            model_pair_rows.append(row)
            pair_p.append(test.pvalue)

        adjusted_pair_p = holm_adjust(pair_p)
        for row, p_adj in zip(model_pair_rows, adjusted_pair_p):
            row["p_holm_within_model"] = p_adj
            pairwise_rows.append(row)

    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["p_holm_across_models"] = holm_adjust(omnibus["p_raw"].to_numpy())
    pairwise_long = pd.DataFrame(pairwise_rows)

    # Wide format mirrors the manuscript's Table B2.
    pairwise_wide = pairwise_long.pivot(
        index="Model", columns="Comparison", values="p_holm_within_model"
    ).reset_index()
    desired_columns = ["Model", "Low vs. Middle", "Low vs. High", "Middle vs. High"]
    pairwise_wide = pairwise_wide.reindex(columns=desired_columns)
    return omnibus, pairwise_long, pairwise_wide


def make_figure6(figure_summary, output_png, output_pdf):
    """Create Figure 6 in a style close to the supplied manuscript figure."""
    x = np.arange(len(COMPLEXITY_ORDER))
    llm_means = figure_summary["LLM_mean"].to_numpy()
    rag_means = figure_summary["RA_LLM_mean"].to_numpy()
    llm_se = figure_summary["LLM_SE"].to_numpy()
    rag_se = figure_summary["RA_LLM_SE"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    blue = "#4C91E8"
    orange = "#F39800"
    error_color = "#808080"

    ax.errorbar(
        x, llm_means, yerr=llm_se, color=blue, ecolor=error_color, linewidth=2.2,
        elinewidth=1.2, capsize=3.5, capthick=1.2, label="LLM"
    )
    ax.errorbar(
        x, rag_means, yerr=rag_se, color=orange, ecolor=error_color, linewidth=2.2,
        elinewidth=1.2, capsize=3.5, capthick=1.2, label="RA-LLM"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(COMPLEXITY_ORDER, fontsize=12)
    ax.set_xlabel("Complexity", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("BERTScore", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylim(0.65, 1.0)
    ax.set_yticks(np.arange(0.65, 1.001, 0.05))
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
        frameon=False, fontsize=12, handlelength=2.2, columnspacing=1.5
    )

    # The manuscript figure reports the unadjusted paired test in each group.
    for xpos, p_value in zip(x, figure_summary["p_raw"]):
        stars = significance_stars(p_value)
        if stars != "ns":
            ax.text(xpos, 0.976, stars, ha="center", va="center", fontsize=13)

    fig.tight_layout()
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    model_data = {}
    for model_name, patterns in MODEL_PATTERNS.items():
        path = locate_input_file(model_name, patterns)
        model_data[model_name] = read_model_data(model_name, path)
        print(f"Loaded {model_name}: {path.name} ({len(model_data[model_name])} questions)")

    pooled = build_pooled_question_data(model_data)
    fig_summary = analyze_figure6(pooled)
    table2 = analyze_table2(pooled)
    table_b1 = analyze_table_b1(model_data)
    omnibus_b2, pairwise_b2_long, pairwise_b2_wide = analyze_table_b2(model_data)

    output_xlsx = BASE_DIR / "bertscore_complexity_analysis_results.xlsx"
    output_png = BASE_DIR / "Figure6_BERTScore_complexity.png"
    output_pdf = BASE_DIR / "Figure6_BERTScore_complexity.pdf"

    methods = pd.DataFrame(
        [
            ["Question-level pooled scores", "For each question, average the LLM scores and RA-LLM scores across the four models."],
            ["Figure 6 error bars", "Mean +/- 1 standard error (SE) across questions within each complexity group."],
            ["Figure 6 significance", "Two-sided exact Wilcoxon signed-rank test on paired question-level pooled scores; stars use unadjusted p-values to match the supplied figure caption."],
            ["Table 2", "Question-level gain = mean RA-LLM BERTScore minus mean LLM BERTScore across the four models; two-sided Mann-Whitney U tests; Holm correction across the three pairwise comparisons."],
            ["Table B1", "Within each model and complexity level, paired two-sided exact Wilcoxon signed-rank tests; Holm correction across the three complexity-level tests within each model."],
            ["Table B2 omnibus", "Kruskal-Wallis tests compare question-level RAG gains across the three complexity levels within each model; Holm correction across the four models."],
            ["Table B2 pairwise", "Two-sided Mann-Whitney U tests compare gains between complexity levels; Holm correction across the three pairwise comparisons within each model."],
            ["Summary values", "SD uses sample standard deviation (ddof=1). Delta score is the mean paired difference (RA-LLM minus LLM)."],
        ],
        columns=["Analysis", "Method"],
    )

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        fig_summary.to_excel(writer, sheet_name="Figure6_summary", index=False)
        table2.to_excel(writer, sheet_name="Table2_pairwise", index=False)
        table_b1.to_excel(writer, sheet_name="TableB1_model_paired", index=False)
        omnibus_b2.to_excel(writer, sheet_name="TableB2_omnibus", index=False)
        pairwise_b2_long.to_excel(writer, sheet_name="TableB2_pairwise_long", index=False)
        pairwise_b2_wide.to_excel(writer, sheet_name="TableB2_pairwise_wide", index=False)
        pooled.to_excel(writer, sheet_name="Pooled_question_data", index=False)
        pd.concat(model_data.values(), ignore_index=True).to_excel(
            writer, sheet_name="Model_question_data", index=False
        )
        methods.to_excel(writer, sheet_name="Methods", index=False)

        # Make column headers visible and apply readable widths.
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                max_length = max(
                    (len(str(cell.value)) for cell in column_cells if cell.value is not None),
                    default=10,
                )
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 48)

    make_figure6(fig_summary, output_png, output_pdf)

    print("\nFigure 6 summary (question-level scores averaged across models):")
    for _, row in fig_summary.iterrows():
        print(
            f"  {row['Complexity']}: LLM={row['LLM_mean']:.4f} +/- {row['LLM_SE']:.4f} SE; "
            f"RA-LLM={row['RA_LLM_mean']:.4f} +/- {row['RA_LLM_SE']:.4f} SE; "
            f"gain={row['Mean_gain']:.4f}; p={format_p(row['p_raw'])}; "
            f"Holm p={format_p(row['p_holm_across_complexity'])}; "
            f"star={row['figure_star_raw_p']}"
        )

    print("\nTable 2: pairwise comparisons of pooled RAG gains:")
    for _, row in table2.iterrows():
        print(
            f"  {row['Comparison']}: mean difference={row['Difference_in_mean_gain']:.4f}; "
            f"p={format_p(row['p_raw'])}; Holm p={format_p(row['p_holm'])}"
        )

    print("\nTable B2 omnibus tests:")
    for _, row in omnibus_b2.iterrows():
        print(
            f"  {row['Model']}: KW p={format_p(row['p_raw'])}; "
            f"Holm p={format_p(row['p_holm_across_models'])}"
        )

    print("\nSaved outputs:")
    print(f"  {output_xlsx}")
    print(f"  {output_png}")
    print(f"  {output_pdf}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
