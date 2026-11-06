#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce Table C4: question-level pointwise ratings from two LLM judges.

Run:
    pip install numpy pandas openpyxl
    python analyze_llm_judges_table_c4.py

Or:
    python analyze_llm_judges_table_c4.py --input "LLM_evulation-qwen(1).xlsx" \
        --output "Table_C4_recalculated.xlsx"

The original file has one row per question, eight score columns in four
pairs (_1 = Gemini-3.0-Pro and _2 = DeepSeek-V3, inferred from Table C4).
Question ID prefixes identify the tasks: M, D, C, and P.

Overall (Pooled) SD is SAMPLE SD (ddof=1) over *all 120 individual ratings*,
not the SD across 60 question-level averages. ICC is ICC(2,2), two-way random
absolute agreement, average of two raters (Shrout-Fleiss / McGraw-Wong).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension

TASKS = [
    ("M", "Medication guidance"),
    ("D", "Diet advice"),
    ("C", "Complication management"),
    ("P", "Patient consultation"),
]
DIMS = ["Completeness", "Clarity", "Correctness", "Usefulness"]
JUDGES = [(1, "Gemini-3.0-Pro"), (2, "DeepSeek-V3")]
INPUT_FILENAME = "LLM_evulation-qwen(1).xlsx"
DEFAULT_OUTPUT = "Table_C4_recalculated.xlsx"

# Manuscript Table C4: values are strings exactly as printed in the screenshot.
# Only for verification; never used to compute results.
REPORTED = {
    ("M", 1): [(4.33, .72), (3.93, .70), (4.20, .68), (3.80, .94)],
    ("M", 2): [(4.47, .64), (3.73, .59), (3.93, .59), (3.60, .83)],
    ("D", 1): [(4.13, .92), (3.93, .59), (3.80, .68), (3.93, .96)],
    ("D", 2): [(4.20, .77), (3.60, .74), (3.87, .64), (3.73, .80)],
    ("C", 1): [(4.40, .51), (3.73, .70), (4.40, .83), (3.60, .91)],
    ("C", 2): [(4.47, .52), (3.80, .77), (4.07, .46), (3.73, .88)],
    ("P", 1): [(4.33, .82), (3.93, .70), (4.27, .70), (4.27, .70)],
    ("P", 2): [(4.33, .62), (3.80, .77), (4.00, .65), (4.13, .74)],
    ("Pooled", 0): [(4.33, .69), (3.81, .69), (4.07, .67), (3.85, .86)],
}
REPORTED_ICC = dict(zip(DIMS, (.677, .599, .692, .668)))


def read_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    # Input may contain trailing blank, preformatted columns/rows.
    if "Dimension" not in df.columns:
        raise ValueError("Expected a column named 'Dimension' with question IDs.")
    df = df.loc[df["Dimension"].notna()].copy()
    cols = [f"{dim}_{j}" for dim in DIMS for j, _ in JUDGES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")
    df["Question_ID"] = df["Dimension"].astype(str).str.strip().str.upper()
    if df["Question_ID"].duplicated().any():
        raise ValueError("Duplicate question IDs: " + ", ".join(df.loc[df["Question_ID"].duplicated(), "Question_ID"]))
    bad = [x for x in df["Question_ID"] if not re.fullmatch(r"[MDCP][LMH][0-9]{2,}", x)]
    if bad:
        raise ValueError(f"Unexpected IDs: {bad}. Expected e.g. ML01, DH05, CH01.")
    df["Task_Code"] = df["Question_ID"].str[0]
    df["Complexity"] = df["Question_ID"].str[1].map({"L": "Low", "M": "Middle", "H": "High"})
    for col in cols:
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any():
            raise ValueError(f"Missing or nonnumeric scores in {col} at IDs {df.loc[v.isna(), 'Question_ID'].tolist()}")
        if not ((v >= 1) & (v <= 5) & (v == np.floor(v))).all():
            raise ValueError(f"Scores in {col} must be integers in 1..5.")
        df[col] = v.astype(int)
    taskcounts = df["Task_Code"].value_counts()
    print("Questions by task:", ", ".join(f"{name}={taskcounts.get(code, 0)}" for code, name in TASKS))
    if any(taskcounts.get(code, 0) == 0 for code, _ in TASKS):
        raise ValueError("All four tasks must have question-level data.")
    if len(df) != 60 or any(taskcounts.get(code, 0) != 15 for code, _ in TASKS):
        print("WARNING: dataset differs from published 60 questions / 15 per task.")
    df["_task_order"] = df["Task_Code"].map({code: i for i, (code, _) in enumerate(TASKS)})
    return df.sort_values(["_task_order", "Question_ID"]).reset_index(drop=True)


def icc_two_way_random_absolute(scores: np.ndarray) -> dict:
    """Balanced two-way ANOVA, ICC(2,1) and ICC(2,k).

    Rows are targets (questions), columns are raters (LLM judges).
    ICC(2,k) = (MSR-MSE) / (MSR + (MSC-MSE)/n)
    ICC(2,1) = (MSR-MSE) / (MSR+(k-1)*MSE+k*(MSC-MSE)/n)
    """
    a = np.asarray(scores, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("ICC requires a complete numeric n-by-k matrix.")
    n, k = a.shape
    if n < 2 or k < 2:
        raise ValueError("ICC requires at least two questions and two judges.")
    grand = a.mean()
    rowmean = a.mean(axis=1, keepdims=True)
    colmean = a.mean(axis=0, keepdims=True)
    ss_row = k * np.square(rowmean - grand).sum()
    ss_col = n * np.square(colmean - grand).sum()
    ss_err = np.square(a - rowmean - colmean + grand).sum()
    msr = ss_row / (n - 1)
    msc = ss_col / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    icc21 = (msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse))
    icc2k = (msr - mse) / (msr + (msc - mse) / n)
    return dict(n=n, k=k, grand=grand, MSR=msr, MSC=msc, MSE=mse,
                ICC_2_1=icc21, ICC_2_k=icc2k)


def compute(df: pd.DataFrame):
    rows = []
    checks = []
    names = {key: name for key, name in TASKS}
    for taskcode, taskname in TASKS:
        g = df[df["Task_Code"] == taskcode]
        for judge_id, judge in JUDGES:
            record = dict(Task=taskname, Model=judge, Task_Code=taskcode, Judge_ID=judge_id, n=len(g))
            for dim in DIMS:
                v = g[f"{dim}_{judge_id}"].to_numpy(dtype=float)
                mean = float(np.mean(v))
                sd = float(np.std(v, ddof=1))
                record[f"{dim}_Mean"] = mean
                record[f"{dim}_SD"] = sd
                expmean, expsd = REPORTED[(taskcode, judge_id)][DIMS.index(dim)]
                checks.extend([
                    dict(Row=f"{taskname} / {judge}", Metric=f"{dim}: mean", Calculated=mean, Reported=expmean, Digits=2, Match=round(mean, 2) == expmean),
                    dict(Row=f"{taskname} / {judge}", Metric=f"{dim}: sample SD", Calculated=sd, Reported=expsd, Digits=2, Match=round(sd, 2) == expsd),
                ])
            rows.append(record)
    pooled = dict(Task="Overall", Model="Pooled", Task_Code="Pooled", Judge_ID=0, n=2*len(df))
    iccs = {}
    for dim in DIMS:
        a = df[[f"{dim}_1", f"{dim}_2"]].to_numpy(dtype=float)
        values = a.ravel()  # 60 questions x 2 separate judge scores = 120 ratings
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        pooled[f"{dim}_Mean"] = mean
        pooled[f"{dim}_SD"] = sd
        exmean, exsd = REPORTED[("Pooled", 0)][DIMS.index(dim)]
        checks.extend([
            dict(Row="Overall / Pooled", Metric=f"{dim}: mean", Calculated=mean, Reported=exmean, Digits=2, Match=round(mean, 2) == exmean),
            dict(Row="Overall / Pooled", Metric=f"{dim}: sample SD", Calculated=sd, Reported=exsd, Digits=2, Match=round(sd, 2) == exsd),
        ])
        iccs[dim] = icc_two_way_random_absolute(a)
        checks.append(dict(Row="Overall / ICC(2,2)", Metric=dim, Calculated=iccs[dim]["ICC_2_k"], Reported=REPORTED_ICC[dim], Digits=3,
                           Match=round(iccs[dim]["ICC_2_k"], 3) == REPORTED_ICC[dim]))
    rows.append(pooled)
    return pd.DataFrame(rows), iccs, pd.DataFrame(checks)


NAVY = "17324D"
ACCENT = "176E91"
PALE = "ECF5FA"
PALE_GRAY = "F2F5F8"
GRAY = "64748B"
GREEN = "137D57"
ORANGE = "A96516"


def prepare_sheet(ws, widths, freeze="A2"):
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = freeze
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[1].height = 29
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(name="Aptos", size=11, color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.sheet_properties.outlinePr.summaryBelow = True


def table_header(ws, names):
    ws.append(names)


def write_output(outpath: Path, df: pd.DataFrame, summ: pd.DataFrame, iccs: dict, checks: pd.DataFrame):
    wb = Workbook()
    ws = wb.active
    ws.title = "Table_C4"
    table_header(ws, ["Task", "Model", *DIMS])
    for _, r in summ.iterrows():
        ws.append([r["Task"], r["Model"]] + [f"{r[f'{d}_Mean']:.2f} ± {r[f'{d}_SD']:.2f}" for d in DIMS])
    ws.append(["Overall", "ICC(2,2)"] + [round(iccs[d]["ICC_2_k"], 3) for d in DIMS])
    prepare_sheet(ws, [26, 20, 19, 19, 19, 19], "C2")
    ws.auto_filter.ref = "A1:F10"
    for row in ws.iter_rows(min_row=2, max_row=10):
        for c in row:
            c.alignment = Alignment(horizontal="center" if c.column >= 3 else "left", vertical="center")
            c.font = Font(name="Aptos", size=10.5, color=NAVY, bold=(c.row >= 10))
            if c.row in (9,10):
                c.fill = PatternFill("solid", fgColor=PALE)
            elif c.row % 2 == 0:
                c.fill = PatternFill("solid", fgColor="F7FAFC")
        ws.row_dimensions[row[0].row].height = 23
    # ICC row 11 (rows: 1 header + 8 group rows + 1 pooled + 1 ICC)
    for c in ws[11]:
        c.fill = PatternFill("solid", fgColor=PALE)
        c.font = Font(name="Aptos", size=10.5, color=NAVY, bold=True)
        c.alignment = Alignment(horizontal="center" if c.column > 2 else "left")
    ws.row_dimensions[11].height = 24
    for row_idx, label, description in [
        (13, "Notes", "Mean ± sample SD (ddof=1); scale 1–5; 15 questions per task per judge."),
        (14, "Pooled", "Mean and sample SD over all 120 individual scores (60 questions × 2 judges)."),
        (15, "ICC", "ICC(2,2): two-way random, absolute agreement, average of two raters."),
    ]:
        ws.cell(row_idx, 1, label)
        ws.cell(row_idx, 2, description)
    for idx in (13,14,15):
        ws.cell(idx,1).font=Font(name="Aptos", color=ACCENT, bold=True)
        ws.merge_cells(start_row=idx, start_column=2, end_row=idx, end_column=6)
        ws.cell(idx,2).alignment=Alignment(wrap_text=True)
        ws.row_dimensions[idx].height=27

    # Raw matrix, sorted by task, retaining pairing by question ID.
    raw = wb.create_sheet("Question_Ratings")
    table_header(raw, ["Question ID", "Task", "Complexity", *[f"{d} ({j})" for d in DIMS for j in ("Gemini", "DeepSeek")]])
    # Explicit in selected order [dim1,dim2] per dimension.
    for _, r in df.iterrows():
        raw.append([r["Question_ID"], dict(TASKS)[r["Task_Code"]], r["Complexity"]]
                   + [int(r[f"{d}_{jid}"]) for d in DIMS for jid, _ in JUDGES])
    prepare_sheet(raw, [18, 26, 15, 19, 19, 18, 18, 20, 20, 19, 19], "D2")
    raw.auto_filter.ref = f"A1:K{raw.max_row}"
    for row in raw.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Aptos", size=10, color=GREEN if c.column >= 4 else GRAY)
            if c.row % 2 == 0:
                c.fill = PatternFill("solid", fgColor="F7FAFC")
            c.alignment = Alignment(horizontal="center")
    # ICC Components, with Excel formulas for recomputing ICC from ANOVA MS terms.
    isc = wb.create_sheet("ICC_Components")
    table_header(isc, ["Dimension", "n", "k", "MSR (targets)", "MSC (judges)", "MSE (residual)", "ICC(2,1)", "ICC(2,k) formula", "Python ICC(2,k)"])
    for dim in DIMS:
        d = iccs[dim]
        rowi=isc.max_row+1
        isc.append([dim, d["n"], d["k"], d["MSR"], d["MSC"], d["MSE"],
                    f"=(D{rowi}-F{rowi})/(D{rowi}+(C{rowi}-1)*F{rowi}+(C{rowi}/B{rowi})*(E{rowi}-F{rowi}))",
                    f"=(D{rowi}-F{rowi})/(D{rowi}+(E{rowi}-F{rowi})/B{rowi})", d["ICC_2_k"]])
    prepare_sheet(isc, [22, 10, 10, 22, 22, 22, 19, 22, 23])
    for row in isc.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(horizontal="center")
            c.font = Font(name="Aptos", size=10, color=NAVY)
            if c.column >= 4:
                c.number_format="0.000000"
    isc.cell(7,1,"Formula")
    isc.cell(7,2,"ICC(2,k) = (MSR-MSE) / [MSR + (MSC-MSE)/n].")
    isc.merge_cells(start_row=7, start_column=2, end_row=7, end_column=9)
    isc.cell(8,1,"Definition")
    isc.cell(8,2,"Two-way random effects; absolute agreement; average measures (k = 2).")
    isc.merge_cells(start_row=8, start_column=2, end_row=8, end_column=9)
    isc.cell(9,1,"MSR / MSC / MSE")
    isc.cell(9,2,"ANOVA mean squares for questions / judges / residual errors.")
    isc.merge_cells(start_row=9, start_column=2, end_row=9, end_column=9)

    vf=wb.create_sheet("Verification")
    table_header(vf, ["Table C4 row", "Metric", "Calculated (unrounded)", "Reported (rounded)", "Digits", "Match?"])
    for _, r in checks.iterrows():
        vf.append([r["Row"], r["Metric"], float(r["Calculated"]), float(r["Reported"]), int(r["Digits"]), "PASS" if r["Match"] else "CHECK"])
    prepare_sheet(vf,[38,27,27,24,12,15],"C2")
    vf.auto_filter.ref = f"A1:F{vf.max_row}"
    for row in vf.iter_rows(min_row=2):
        for c in row:
            c.font=Font(name="Aptos", size=10, color=NAVY)
            c.alignment=Alignment(horizontal="center" if c.column > 2 else "left")
            if c.row%2==0:
                c.fill=PatternFill("solid",fgColor="F7FAFC")
        row[2].number_format="0.000000"
        row[3].number_format="0.000"
        last=row[5]
        last.font=Font(name="Aptos", size=10, bold=True, color=GREEN if last.value=="PASS" else ORANGE)

    guide=wb.create_sheet("Readme")
    table_header(guide,["Item","Description"])
    items=[
        ("Input", f"{INPUT_FILENAME}; question IDs in 'Dimension' and eight columns DIMENSION_1/_2."),
        ("Question IDs", "M: medication guidance; D: diet advice; C: complication management; P: patient consultation."),
        ("LLM mapping", "_1: Gemini-3.0-Pro; _2: DeepSeek-V3. This mapping is inferred from manuscript Table C4."),
        ("Task rows", "n=15 per task per judge, sample SD with ddof=1."),
        ("Overall pooled", "n=120 ratings, not n=60 averaged answers. Two scores per question remain separate."),
        ("ICC(2,2)", "Two-way random-effects absolute-agreement ICC for the mean of two judges."),
        ("Question_Ratings", "Sorted original question-level inputs; no score transformation."),
        ("ICC_Components", "Two-way ANOVA mean squares and formulas for ICC(2,1) and ICC(2,2)."),
        ("Verification", "Comparison with reported values rounded as in screenshot; reported values never influence calculations."),
        ("Reproducibility", "Run analyze_llm_judges_table_c4.py with numpy, pandas, openpyxl installed."),
    ]
    for a,b in items: guide.append([a,b])
    prepare_sheet(guide,[24,112])
    for row in guide.iter_rows(min_row=2):
        row[1].alignment=Alignment(wrap_text=True, vertical="center")
        row[0].font=Font(name="Aptos",bold=True,color=ACCENT)
        guide.row_dimensions[row[0].row].height=33

    try:
        from openpyxl.workbook.properties import CalcProperties
        wb.calculation = CalcProperties(calcMode="auto", fullCalcOnLoad=True, forceFullCalc=True)
    except ImportError:
        pass
    outpath.parent.mkdir(parents=True, exist_ok=True)
    wb.save(outpath)
    return outpath


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path(INPUT_FILENAME), help="Path to the original Excel workbook")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Path to the output Excel workbook")
    args=parser.parse_args()
    df=read_and_validate(args.input)
    summ,iccs,checks=compute(df)
    out=write_output(args.output, df, summ, iccs, checks)
    print("\n=== Table C4 (mean ± sample SD) ===")
    print(summ[["Task","Model", *[f"{d}_Mean" for d in DIMS]]].to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("\n=== ICC(2,2): two-way random / absolute / average of two judges ===")
    for dim in DIMS:
        print(f"{dim:14s}: {iccs[dim]['ICC_2_k']:.6f}  (table: {REPORTED_ICC[dim]:.3f})")
    npass=int(checks["Match"].sum()); ntotal=len(checks)
    print(f"\nVerified {npass} / {ntotal} published entries at their reported precision.")
    if npass != ntotal:
        print("MISMATCHES:\n",checks.loc[~checks["Match"]].to_string(index=False))
        return 2
    print("Output:",out.resolve())
    return 0

if __name__ == '__main__':
    sys.exit(main())
