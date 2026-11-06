#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalculate Supplementary Tables B5-B7 from data-1125(1).xlsx.

Requirements:
    python -m pip install numpy pandas scipy openpyxl python-docx

Run with this script and data-1125(1).xlsx in the same folder:
    python recalculate_B5_B7_from_merged_data.py

Optional paths:
    python recalculate_B5_B7_from_merged_data.py --input-file "D:/data/data-1125(1).xlsx" --output-dir "D:/results"

The script reads the four model blocks from columns D:S and uses only rows with
valid question IDs. It ignores the extra formula/summary columns and rows in the
source workbook. It does not regenerate BLEURT or FactCC scores from text.

B5: each model separately plus a cross-model summary.
B6: results by question complexity.
B7: results by health-management task.

For cross-model summaries, scores from the four fixed models are averaged within
each question before descriptive statistics and paired Wilcoxon tests are run.
This avoids treating the four scores for the same question as independent.
Delta = RA-LLM minus LLM in original score units; no percentage changes are used.
SD is sample SD (ddof=1). P-values are two-sided, unadjusted. Subgroup tests are
within-group tests, not direct tests of between-group differences.
"""

import argparse
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH

MODEL_BLOCKS = {
    "Qwen2.5-14b": (3, "qwen2.5-14b"),
    "Qwen2.5-7b": (7, "qwen2-5-7b"),
    "Baichuan2-13b": (11, "baichuan2-13b"),
    "Chatglm3-6b": (15, "glm3-6b"),
}
TASKS = {
    "M": "Medication guidance",
    "D": "Diet advice",
    "C": "Complication management",
    "P": "Patient consultation",
}
COMPLEXITIES = {"L": "Low", "M": "Middle", "H": "High"}
METRIC_COLS = ["llm_bleurt", "rag_bleurt", "llm_factcc", "rag_factcc"]
METRICS = ["BLEURT", "FactCC"]
QUESTION_PATTERN = r"[MDCP][LMH]0[1-5]"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_merged_workbook(path):
    """Read the two-row model/metric header and extract 60 question rows."""
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object)
    if raw.shape[0] < 62 or raw.shape[1] < 19:
        raise ValueError(f"Unexpected workbook shape: {raw.shape}; expected at least 62 rows and 19 columns.")

    # The first row contains merged model headers; the second row contains metric headers.
    expected_metric_headers = METRIC_COLS
    for model, (start_col, expected_name) in MODEL_BLOCKS.items():
        found_name = str(raw.iat[0, start_col]).strip().lower()
        if found_name != expected_name.lower():
            raise ValueError(
                f"Model header mismatch at column {start_col + 1}: expected {expected_name!r}, found {found_name!r}."
            )
        found_metrics = [str(raw.iat[1, start_col + j]).strip().lower() for j in range(4)]
        if found_metrics != expected_metric_headers:
            raise ValueError(
                f"Metric headers for {model} do not match expected {expected_metric_headers}: {found_metrics}"
            )

    # Question rows begin on Excel row 3. Ignore summary/formula rows without a question ID.
    data = raw.iloc[2:, :].copy()
    id_values = data.iloc[:, 0]
    nonblank_ids = id_values.notna() & id_values.astype(str).str.strip().ne("")
    candidates = data.loc[nonblank_ids].copy()
    candidates.iloc[:, 0] = candidates.iloc[:, 0].astype(str).str.strip().str.upper()
    valid_mask = candidates.iloc[:, 0].str.fullmatch(QUESTION_PATTERN)
    if (~valid_mask).any():
        bad = candidates.loc[~valid_mask].iloc[:, 0].tolist()
        raise ValueError(f"Unexpected nonblank values in the question-ID column: {bad}")

    if len(candidates) != 60:
        raise ValueError(f"Expected 60 question rows, found {len(candidates)}.")
    if candidates.iloc[:, 0].duplicated().any():
        duplicates = candidates.loc[candidates.iloc[:, 0].duplicated(keep=False)].iloc[:, 0].tolist()
        raise ValueError(f"Duplicated question IDs: {duplicates}")

    question_ids = candidates.iloc[:, 0].reset_index(drop=True)
    task_labels = candidates.iloc[:, 1].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).reset_index(drop=True)
    complexity_labels = candidates.iloc[:, 2].astype(str).str.strip().str.title().reset_index(drop=True)
    expected_tasks = question_ids.str[0].map(TASKS)
    expected_complexities = question_ids.str[1].map(COMPLEXITIES)

    task_aliases = {
        "medication guidance": "medication guidance",
        "diet advice": "diet advice",
        "diet advise": "diet advice",
        "complication management": "complication management",
        "patient consultation": "patient consultation",
    }
    task_labels = task_labels.replace(task_aliases)
    if not (task_labels == expected_tasks.str.lower()).all():
        bad = pd.DataFrame({"No": question_ids, "Task_in_file": task_labels, "Task_expected": expected_tasks})
        raise ValueError("Task labels disagree with question IDs:\n" + bad.loc[bad.Task_in_file != bad.Task_expected.str.lower()].to_string(index=False))
    if not (complexity_labels == expected_complexities).all():
        bad = pd.DataFrame({"No": question_ids, "Complexity_in_file": complexity_labels, "Complexity_expected": expected_complexities})
        raise ValueError("Complexity labels disagree with question IDs:\n" + bad.loc[bad.Complexity_in_file != bad.Complexity_expected].to_string(index=False))

    datasets = {}
    for model, (start_col, _) in MODEL_BLOCKS.items():
        frame = pd.DataFrame({
            "No": question_ids,
            "Task": expected_tasks.values,
            "Complexity": expected_complexities.values,
        })
        for j, col in enumerate(METRIC_COLS):
            frame[col] = pd.to_numeric(candidates.iloc[:, start_col + j], errors="raise").to_numpy()
        values = frame[METRIC_COLS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing or non-finite metric scores found for {model}.")
        if not frame[["llm_factcc", "rag_factcc"]].apply(lambda col: col.between(0, 1).all()).all():
            raise ValueError(f"FactCC scores for {model} must be within [0, 1].")
        frame["Model"] = model
        datasets[model] = frame.sort_values("No").reset_index(drop=True)

    return datasets


def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon; use exact p where eligible."""
    diff = np.round(np.asarray(b, dtype=float) - np.asarray(a, dtype=float), 12)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0, "all-zero convention"
    tied_absolute_ranks = len(np.unique(np.abs(nonzero))) != len(nonzero)
    use_exact = len(nonzero) == len(diff) and len(diff) <= 50 and not tied_absolute_ranks
    method = "exact" if use_exact else "approx"
    result = wilcoxon(
        diff,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method=method,
    )
    return float(result.statistic), float(result.pvalue), method


def summarize(df, group, unit):
    rows = []
    for metric in METRICS:
        a = df[f"llm_{metric.lower()}"].to_numpy(dtype=float)
        b = df[f"rag_{metric.lower()}"].to_numpy(dtype=float)
        w, p, p_method = paired_wilcoxon(a, b)
        rows.append({
            "Group": group,
            "Metric": metric,
            "n": len(df),
            "analysis_unit": unit,
            "llm_mean": float(a.mean()),
            "llm_sd": float(a.std(ddof=1)),
            "ra_mean": float(b.mean()),
            "ra_sd": float(b.std(ddof=1)),
            "delta": float((b - a).mean()),
            "delta_sd": float((b - a).std(ddof=1)),
            "wilcoxon_W": w,
            "p_value": p,
            "p_method": p_method,
            "zero_differences": int((np.round(b - a, 12) == 0).sum()),
        })
    return rows


def format_p(p):
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def display_table(rows, first_column):
    stats = pd.DataFrame(rows)
    pm = "\u00b1"
    delta_label = "\u0394 score"
    return pd.DataFrame({
        first_column: stats["Group"],
        "Metric": stats["Metric"],
        "LLM": [f"{m:.3f} {pm} {s:.3f}" for m, s in zip(stats.llm_mean, stats.llm_sd)],
        "RA-LLM": [f"{m:.3f} {pm} {s:.3f}" for m, s in zip(stats.ra_mean, stats.ra_sd)],
        delta_label: [f"{x:+.4f}" for x in stats.delta],
        "Wilcoxon p-value": [format_p(p) for p in stats.p_value],
    })


def markdown_table(frame):
    rows = [list(frame.columns)] + frame.astype(str).values.tolist()
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def add_docx_table(doc, label, title, frame, notes):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.add_run(f"{label}. ").bold = True
    p.add_run(title)
    table = doc.add_table(rows=1, cols=len(frame.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [1.40, 0.80, 1.30, 1.30, 0.90, 1.10]
    for col, width in zip(table.columns, widths):
        col.width = Inches(width)
    data = [list(frame.columns)] + frame.astype(str).values.tolist()
    for row_index, row_values in enumerate(data):
        cells = table.rows[0].cells if row_index == 0 else table.add_row().cells
        for col_index, (cell, value) in enumerate(zip(cells, row_values)):
            cell.width = Inches(widths[col_index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            para = cell.paragraphs[0]
            para.paragraph_format.space_before = Pt(4)
            para.paragraph_format.space_after = Pt(4)
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_index < 2 else WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(str(value))
            run.font.size = Pt(9)
            run.bold = row_index == 0
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right"):
                element = OxmlElement(f"w:{edge}")
                for key, val in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
                    element.set(qn(f"w:{key}"), val)
                borders.append(element)
            tc_pr.append(borders)
            if row_index == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "E8EDF2")
                tc_pr.append(shade)
        tr_pr = table.rows[row_index]._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if row_index == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
    for note in notes:
        para = doc.add_paragraph(note)
        para.paragraph_format.space_before = Pt(5)
        para.paragraph_format.space_after = Pt(4)
        for run in para.runs:
            run.font.size = Pt(9)


def save_outputs(tables, out_dir, raw_stats, question_means, overlap_audit, metadata):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.55)
    section.left_margin = section.right_margin = Inches(0.45)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)

    markdown_blocks, html_blocks = [], []
    for idx, (label, title, frame, notes) in enumerate(tables):
        if idx:
            doc.add_page_break()
        add_docx_table(doc, label, title, frame, notes)
        frame.to_csv(out_dir / f"{label.replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")
        markdown_blocks.append(f"## {label}. {title}\n\n{markdown_table(frame)}\n\n" + "\n\n".join(notes))
        html_blocks.append(
            f"<h2>{html.escape(label + '. ' + title)}</h2>" + frame.to_html(index=False, escape=True) +
            "".join(f"<p>{html.escape(note)}</p>" for note in notes)
        )

    (out_dir / "Tables_B5_B7.md").write_text("\n\n".join(markdown_blocks), encoding="utf-8")
    (out_dir / "Tables_B5_B7.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        'body{font-family:serif;margin:30px}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #bbb;padding:6px;text-align:left}th{background:#eee}'
        'p{max-width:1000px}</style></head><body>' + "".join(html_blocks) + "</body></html>",
        encoding="utf-8",
    )
    doc.save(out_dir / "Tables_B5_B7.docx")

    with pd.ExcelWriter(out_dir / "Tables_B5_B7_results.xlsx", engine="openpyxl") as writer:
        for label, _, frame, _ in tables:
            frame.to_excel(writer, sheet_name=label.replace(" ", "_"), index=False)
        raw_stats.to_excel(writer, sheet_name="Unrounded_statistics", index=False)
        question_means.to_excel(writer, sheet_name="Question_level_means", index=False)
        overlap_audit.to_excel(writer, sheet_name="Cross_model_audit", index=False)
        pd.DataFrame([
            {"item": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
            for k, v in metadata.items()
        ]).to_excel(writer, sheet_name="Metadata", index=False)

    raw_stats.to_csv(out_dir / "B5_B7_unrounded_statistics.csv", index=False, encoding="utf-8-sig")
    question_means.to_csv(out_dir / "question_level_model_averages.csv", index=False, encoding="utf-8-sig")
    overlap_audit.to_csv(out_dir / "cross_model_overlap_audit.csv", index=False, encoding="utf-8-sig")
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path, default=Path(__file__).resolve().parent / "data-1125(1).xlsx")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_file = args.input_file.resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input workbook not found: {input_file}")
    output_dir = (args.output_dir or input_file.parent / "results_B5_B7_from_merged_data").resolve()

    datasets = load_merged_workbook(input_file)
    first = next(iter(datasets.values()))
    for model, df in datasets.items():
        if not df["No"].equals(first["No"]):
            raise ValueError(f"Question IDs do not align for model {model}.")

    # Average the four fixed models within each question before cross-model inference.
    combined = pd.concat(datasets.values(), ignore_index=True)
    question_means = combined.groupby(["No", "Task", "Complexity"], sort=True)[METRIC_COLS].mean().reset_index()
    if len(question_means) != 60 or question_means["No"].duplicated().any():
        raise ValueError("Cross-model aggregation did not yield 60 unique questions.")

    b5_rows = []
    for model, df in datasets.items():
        b5_rows.extend(summarize(df, model, "60 matched questions for this model"))
    b5_rows.extend(summarize(question_means, "Average", "60 unique questions; four-model mean within question"))

    b6_rows = []
    for complexity in ("Low", "Middle", "High"):
        subset = question_means[question_means["Complexity"] == complexity]
        if len(subset) != 20:
            raise ValueError(f"Expected 20 questions for {complexity}; found {len(subset)}")
        b6_rows.extend(summarize(subset, complexity, "20 unique questions; four-model mean within question"))

    b7_rows = []
    for task in TASKS.values():
        subset = question_means[question_means["Task"] == task]
        if len(subset) != 15:
            raise ValueError(f"Expected 15 questions for {task}; found {len(subset)}")
        b7_rows.extend(summarize(subset, task, "15 unique questions; four-model mean within question"))

    raw_stats = pd.concat([
        pd.DataFrame(b5_rows).assign(Table="B5"),
        pd.DataFrame(b6_rows).assign(Table="B6"),
        pd.DataFrame(b7_rows).assign(Table="B7"),
    ], ignore_index=True)

    common_note = (
        "Note. Values are mean \u00b1 sample standard deviation (SD; denominator n - 1). "
        "Delta score = mean(RA-LLM - LLM), calculated before rounding and reported in original metric units. "
        "Positive values indicate higher scores for RA-LLM; negative values indicate lower scores. "
        "P-values are two-sided paired Wilcoxon signed-rank tests and are unadjusted."
    )
    cross_model_note = (
        "For the cross-model summary, the four models were averaged within each question separately for LLM and RA-LLM. "
        "The paired test therefore uses 60 question-level mean differences, not 240 model-question observations."
    )
    subgroup_note = (
        "Subgroup p-values test the paired LLM-versus-RA-LLM difference within each subgroup; they do not directly test "
        "whether RAG gains differ between complexity levels or between tasks. These subgroup results should not be "
        "interpreted as tests of between-group differences."
    )
    wilcoxon_note = (
        "Exact p-values were used when there were at most 50 nonzero paired differences, no zero differences, and no tied "
        "absolute differences; otherwise, a normal approximation without continuity correction was used. Paired differences "
        "were rounded to 12 decimal places for rank assignment only."
    )
    tables = [
        ("Table B5", "Comparison of BLEURT and FactCC scores between LLM and RA-LLM",
         display_table(b5_rows, "Model"), [common_note, cross_model_note, wilcoxon_note]),
        ("Table B6", "Comparison of BLEURT and FactCC scores across question complexity levels",
         display_table(b6_rows, "Type"), [common_note, cross_model_note, subgroup_note, wilcoxon_note]),
        ("Table B7", "Comparison of BLEURT and FactCC scores across management tasks",
         display_table(b7_rows, "Type"), [common_note, cross_model_note, subgroup_note, wilcoxon_note]),
    ]

    # Audit exact/near-identical scores across model blocks; report but do not silently change them.
    audit_rows = []
    model_names = list(datasets)
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            a = datasets[model_a].set_index("No")[METRIC_COLS]
            b = datasets[model_b].set_index("No")[METRIC_COLS]
            row_equal = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=1e-12).all(axis=1)
            item = {
                "model_a": model_a,
                "model_b": model_b,
                "identical_all_four_metric_values": int(row_equal.sum()),
                "identical_question_ids": "; ".join(a.index[row_equal].astype(str).tolist()),
                "total_questions": 60,
            }
            for col in METRIC_COLS:
                item[f"identical_{col}_values"] = int(np.isclose(a[col].to_numpy(), b[col].to_numpy(), rtol=0, atol=1e-12).sum())
            audit_rows.append(item)
    overlap_audit = pd.DataFrame(audit_rows)

    metadata = {
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "input_file": input_file.name,
        "input_sha256": sha256_file(input_file),
        "question_count": 60,
        "models": list(datasets.keys()),
        "table_sample_sizes": {
            "B5_each_model": 60,
            "B5_cross_model_summary": 60,
            "B6_each_complexity": 20,
            "B7_each_task": 15,
        },
        "aggregation": "For cross-model summaries, average the four fixed models within each question before descriptive statistics and paired tests.",
        "sd": "Sample standard deviation, ddof=1.",
        "delta": "Mean(RA-LLM minus LLM) on the original metric scale; not a percentage.",
        "wilcoxon": "Two-sided paired signed-rank test; exact where eligible, otherwise asymptotic without continuity correction; p-values unadjusted.",
        "source_limitation": "BLEURT and FactCC were read from the merged workbook; the script does not regenerate metrics from text.",
        "excluded_questions": [],
        "notes": [
            "Only rows with valid question IDs were included; summary/formula rows and extra columns U:X were ignored.",
            "Subgroup p-values are within-group paired tests, not direct tests of differences between subgroup gains.",
            "Identical score patterns across model blocks are listed in the audit sheet and are not automatically removed.",
        ],
    }

    save_outputs(tables, output_dir, raw_stats, question_means, overlap_audit, metadata)
    print("Successfully recalculated Tables B5-B7 from:", input_file)
    print("Question rows included:", len(first))
    print("Output directory:", output_dir)
    print("\nTable B5\n", display_table(b5_rows, "Model").to_string(index=False))
    print("\nTable B6\n", display_table(b6_rows, "Type").to_string(index=False))
    print("\nTable B7\n", display_table(b7_rows, "Type").to_string(index=False))
    print("\nCross-model overlap audit\n", overlap_audit[["model_a", "model_b", "identical_all_four_metric_values", "total_questions"]].to_string(index=False))


if __name__ == "__main__":
    main()
