#!/usr/bin/env python3
"""Reproduce manuscript Figure 7, Table 3, Table B3 and Table B4.

Put this script alongside the four original model Excel workbooks, then run:
    pip install pandas numpy scipy statsmodels matplotlib openpyxl
    python analyze_bertscore_tasks.py

Input columns (first sheet of each Excel file):
    No, Task, llm_bertscore_r, rag_bertscore_r

Outputs in the script's folder:
    bertscore_task_analysis_results.xlsx
    Figure7_BERTScore_tasks.png
    Figure7_BERTScore_tasks.pdf

IMPORTANT:
- Question is the statistical unit for the pooled analyses: first average the
  scores from four models for the same question, then calculate group statistics.
- Figure 7 is annotated using Holm-adjusted p values across FOUR paired
  within-task Wilcoxon tests (this reproduces *** / * / ** / **).
- Table 3 and B4 use two-sided independent-group Mann-Whitney U tests on gains;
  six comparisons are adjusted by Holm in each relevant family.
- Table B3 uses paired exact two-sided Wilcoxon tests with Holm adjustment
  across four tasks separately in each model.
- Sample SD uses ddof=1 and Figure 7 error bars use 1 SE = sample SD/sqrt(n).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
MODEL_FILES = {
    'Baichuan2-13B': ('baichuan2-13b',),
    'GLM3-6B': ('glm3-6b',),
    'Qwen2.5-14B': ('qwen2.5-14b',),
    'Qwen2.5-7B': ('qwen2-5-7b', 'qwen2.5-7b'),
}
TASK_CODES = ['M', 'D', 'C', 'P']
TASK_NAMES = {
    'M': 'Medication guidance',
    'D': 'Diet advice',
    'C': 'Complication management',
    'P': 'Patient consultation',
}
TASK_ALIASES = {
    'medication guidance': 'M',
    'diet advise': 'D',  # original Excel spelling
    'diet advice': 'D',
    'complication management': 'C',
    'patient consultation': 'P',
}
PAIR_CODES = list(combinations(TASK_CODES, 2))
SCORE_COLUMNS = ['No', 'Task', 'llm_bertscore_r', 'rag_bertscore_r']
OUTPUT_XLSX = 'bertscore_task_analysis_results.xlsx'
OUTPUT_PNG = 'Figure7_BERTScore_tasks.png'
OUTPUT_PDF = 'Figure7_BERTScore_tasks.pdf'


def find_one_workbook(model: str, aliases: tuple[str, ...]) -> Path:
    matches = [
        x for x in BASE.glob('*.xlsx')
        if not x.name.startswith('~$')
        and any(a in x.name.lower() for a in aliases)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f'For {model}, expected exactly one input Excel file in {BASE}; '
            f'found {len(matches)}: {[p.name for p in matches]}. '
            'Remove duplicate model workbooks or edit MODEL_FILES.'
        )
    return matches[0]


def read_one_model(model: str, file: Path) -> pd.DataFrame:
    src = pd.read_excel(file)
    src.columns = [str(s).strip() for s in src.columns]
    missing = [c for c in SCORE_COLUMNS if c not in src.columns]
    if missing:
        raise ValueError(f'{file.name} lacks necessary columns: {missing}')

    cols = SCORE_COLUMNS + (['Complexity'] if 'Complexity' in src.columns else [])
    d = src[cols].copy()
    d['No'] = d['No'].astype(str).str.strip()
    cleaned = d['Task'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    d['Task'] = cleaned.map(TASK_ALIASES)
    if d['Task'].isna().any():
        bad = sorted(set(cleaned[d['Task'].isna()]))
        raise ValueError(f'Unknown health management task in {file.name}: {bad}')
    for c in ['llm_bertscore_r', 'rag_bertscore_r']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
        if d[c].isna().any():
            raise ValueError(f'Invalid/missing values in {file.name}: {c}')
    if d['No'].eq('').any() or d['No'].eq('nan').any():
        raise ValueError(f'Blank question ID in {file.name}')
    if d['No'].duplicated().any():
        raise ValueError(f'Duplicate question IDs in {file.name}')
    if not ((d[['llm_bertscore_r', 'rag_bertscore_r']] >= 0).all().all() and
            (d[['llm_bertscore_r', 'rag_bertscore_r']] <= 1).all().all()):
        warnings.warn(f'BERTScores in {file.name} contain values outside [0,1].')
    counts = d['Task'].value_counts().to_dict()
    if counts != {'M': 15, 'D': 15, 'C': 15, 'P': 15}:
        warnings.warn(f'{file.name}: task counts {counts}, expected 15 in each task')
    d['Model'] = model
    d['gain'] = d['rag_bertscore_r'] - d['llm_bertscore_r']
    return d


def aggregate_across_models(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_models = pd.concat(data.values(), ignore_index=True)
    n_models = len(data)
    id_counts = all_models.groupby('No')['Model'].nunique()
    if not (id_counts == n_models).all():
        raise ValueError('A question ID is not represented in every model: ' +
                         str(id_counts[id_counts != n_models].to_dict()))
    if (all_models.groupby('No')['Task'].nunique() != 1).any():
        raise ValueError('Task labels for the same question differ across model files')
    pooled = (
        all_models.groupby(['No', 'Task'], as_index=False)
        .agg(LLM=('llm_bertscore_r', 'mean'), RA_LLM=('rag_bertscore_r', 'mean'))
    )
    pooled['gain'] = pooled['RA_LLM'] - pooled['LLM']
    pooled['Task'] = pd.Categorical(pooled['Task'], TASK_CODES, ordered=True)
    return pooled.sort_values(['Task', 'No']).reset_index(drop=True)


def holm(raw_p):
    return multipletests(np.asarray(raw_p, dtype=float), method='holm')[1]


def wilcoxon_exact_pair(df, rag_col, llm_col):
    # SciPy's 'exact' method corresponds to the manuscript analysis.
    test = wilcoxon(df[rag_col], df[llm_col], alternative='two-sided', method='exact')
    return float(test.statistic), float(test.pvalue)


def stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.4f}'


def figure7_statistics(pooled):
    results = []
    for task in TASK_CODES:
        x = pooled.loc[pooled['Task'] == task]
        n = len(x)
        w, p = wilcoxon_exact_pair(x, 'RA_LLM', 'LLM')
        results.append({
            'Task': task, 'Task_name': TASK_NAMES[task], 'n_questions': n,
            'LLM_mean': x['LLM'].mean(), 'LLM_SD': x['LLM'].std(ddof=1),
            'LLM_SE': x['LLM'].std(ddof=1) / np.sqrt(n),
            'RA_LLM_mean': x['RA_LLM'].mean(), 'RA_LLM_SD': x['RA_LLM'].std(ddof=1),
            'RA_LLM_SE': x['RA_LLM'].std(ddof=1) / np.sqrt(n),
            'Delta_score': x['gain'].mean(), 'Wilcoxon_W': w,
            'p_raw': p,
        })
    out = pd.DataFrame(results)
    out['p_holm_across_4_tasks'] = holm(out['p_raw'])
    # Holm adjustment is needed to reproduce original manuscript annotations.
    out['Figure7_star'] = out['p_holm_across_4_tasks'].map(stars)
    return out


def two_sample_gain_tests(df, task_a, task_b, gain_col='gain'):
    x = df.loc[df['Task'] == task_a, gain_col].astype(float)
    y = df.loc[df['Task'] == task_b, gain_col].astype(float)
    res = mannwhitneyu(x, y, alternative='two-sided', method='auto')
    return {
        'Comparison': f'{task_a} vs. {task_b}',
        'Task_1': TASK_NAMES[task_a], 'Task_2': TASK_NAMES[task_b],
        'n_1': len(x), 'n_2': len(y),
        'Mean_gain_1': x.mean(), 'Mean_gain_2': y.mean(),
        'Difference_in_mean_gain': x.mean() - y.mean(),
        'Mann_Whitney_U': float(res.statistic), 'p_raw': float(res.pvalue),
    }


def table3_statistics(pooled):
    d = pd.DataFrame([two_sample_gain_tests(pooled, a, b) for a,b in PAIR_CODES])
    d['p_holm_6_comparisons'] = holm(d['p_raw'])
    return d


def table_b3_statistics(model_data):
    results = []
    for model, src in model_data.items():
        blocks = []
        for task in TASK_CODES:
            d = src.loc[src['Task'] == task]
            w,p = wilcoxon_exact_pair(d, 'rag_bertscore_r', 'llm_bertscore_r')
            blocks.append({
                'Model': model, 'Task': task, 'Task_name': TASK_NAMES[task],
                'n_questions': len(d),
                'LLM_mean': d['llm_bertscore_r'].mean(),
                'LLM_SD': d['llm_bertscore_r'].std(ddof=1),
                'RA_LLM_mean': d['rag_bertscore_r'].mean(),
                'RA_LLM_SD': d['rag_bertscore_r'].std(ddof=1),
                'Delta_score': d['gain'].mean(), 'Wilcoxon_W': w, 'p_raw': p,
            })
        p_adj = holm([r['p_raw'] for r in blocks])
        for row, pa in zip(blocks, p_adj):
            row['p_holm_within_model'] = float(pa)
            results.append(row)
    return pd.DataFrame(results)


def table_b4_statistics(model_data):
    rows = []
    for model, d in model_data.items():
        tmp = [two_sample_gain_tests(d, a, b) for a,b in PAIR_CODES]
        adjusted = holm([r['p_raw'] for r in tmp])
        for row, p_adj in zip(tmp, adjusted):
            row['Model'] = model
            row['p_holm_within_model_6'] = float(p_adj)
            rows.append(row)
    long = pd.DataFrame(rows)
    wide = long.pivot(index='Model', columns='Comparison', values='p_holm_within_model_6')
    wide = wide.reindex(
        index=list(MODEL_FILES),
        columns=[f'{a} vs. {b}' for a,b in PAIR_CODES]
    ).reset_index()
    return long, wide


def plot_figure7(summary, path_png, path_pdf):
    positions = np.arange(len(TASK_CODES))
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    for label, vals, errs, color in [
        ('LLM', summary['LLM_mean'].to_numpy(), summary['LLM_SE'].to_numpy(), '#4c91e8'),
        ('RA-LLM', summary['RA_LLM_mean'].to_numpy(), summary['RA_LLM_SE'].to_numpy(), '#f39800'),
    ]:
        ax.errorbar(positions, vals, yerr=errs, color=color, ecolor='#7c7c7c',
                    linewidth=2.0, elinewidth=1.1, capthick=1.1, capsize=3.2,
                    label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Medication\nguidance', 'Diet advice',
                        'Complication\nmanagement', 'Patient\nconsultation'],
                       fontsize=12)
    ax.set_xlabel('Task', fontsize=14, fontweight='bold', labelpad=13)
    ax.set_ylabel('BERTScore', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylim(0.65, 1.0)
    ax.set_yticks(np.arange(0.65, 1.001, 0.05))
    ax.set_yticklabels([f'{v:.2f}'.rstrip('0').rstrip('.') for v in np.arange(0.65, 1.001, 0.05)],
                       fontsize=11)
    ax.set_xlim(-0.5, 3.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.15)
    ax.spines['bottom'].set_linewidth(1.15)
    # Manuscript-style legend: line segments only, without the error-bar caps.
    legend_handles = [
        Line2D([0], [0], color='#4c91e8', linewidth=2, label='LLM'),
        Line2D([0], [0], color='#f39800', linewidth=2, label='RA-LLM'),
    ]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5,1.17),
              ncol=2, frameon=False, fontsize=12, handlelength=2.0, columnspacing=1.4)
    for i, label in enumerate(summary['Figure7_star']):
        if label != 'ns':
            ax.text(i, 0.978, label, ha='center', va='center', fontsize=13)
    fig.tight_layout()
    fig.savefig(path_png, dpi=600, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig)


def style_workbook(wb):
    ink = '183153'; blue = '204465'; light = 'EAF2F8'
    green = '087E60'
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 30
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor=blue)
            cell.font = Font(name='Aptos', size=10, bold=True, color='FFFFFF')
            cell.alignment = Alignment(vertical='center', horizontal='left', wrap_text=True)
            cell.border = Border(bottom=Side(style='medium', color='799BB9'))
        for row in ws.iter_rows(min_row=2):
            row_num = row[0].row
            ws.row_dimensions[row_num].height = 21
            for cell in row:
                if row_num % 2 == 0:
                    cell.fill = PatternFill('solid', fgColor='F5F9FC')
                cell.font = Font(name='Aptos', size=10, color=green if ws.title.endswith('_data') else ink)
                cell.alignment = Alignment(vertical='center')
                if isinstance(cell.value, (float, np.floating)):
                    cell.number_format = '0.000000'
        for col in ws.columns:
            letter = get_column_letter(col[0].column)
            max_len = max(len(str(c.value)) for c in col if c.value is not None)
            ws.column_dimensions[letter].width = min(max(max_len+3, 12), 46)


def check_against_manuscript(fig7, tab3, b3, b4):
    """Report checks without inventing an exact match if files have changed."""
    check_rows=[]
    target_stars={'M':'***','D':'*','C':'**','P':'**'}
    for _,r in fig7.iterrows():
        check_rows.append({'Source': 'Figure 7', 'Item':r['Task'],
                           'Expected':target_stars[r['Task']], 'Computed':r['Figure7_star'],
                           'Match': target_stars[r['Task']]==r['Figure7_star']})
    expected_table3=[0.0296,-0.0134,-0.0141,-0.0430,-0.0437,-0.0007]
    expected_p=[0.4075,1.0000,1.0000,0.3384,0.4075,1.0000]
    for j,(_,r) in enumerate(tab3.iterrows()):
        for column, expected in [('Difference_in_mean_gain',expected_table3[j]),
                                 ('p_holm_6_comparisons',expected_p[j])]:
            computed=round(float(r[column]),4)
            check_rows.append({'Source':'Table 3', 'Item':r['Comparison']+' / '+column,
                               'Expected':f'{expected:.4f}', 'Computed':f'{computed:.4f}',
                               'Match':abs(computed-expected)<1e-7})
    # Check two detailed model-specific points documented in the manuscript.
    for m,t,expected_gain,expected_p in [
        ('Baichuan2-13B','M',0.0679,0.0007),
        ('GLM3-6B','P',0.0822,0.0511),
        ('Qwen2.5-14B','P',0.1072,0.0046),
        ('Qwen2.5-7B','D',0.0382,0.2293),
    ]:
        r=b3[(b3.Model==m)&(b3.Task==t)].iloc[0]
        for metric,expected in [('Delta_score',expected_gain),('p_holm_within_model',expected_p)]:
            actual=round(float(r[metric]),4)
            check_rows.append({'Source':'Table B3', 'Item':m+' / '+t+' / '+metric,
                               'Expected':f'{expected:.4f}', 'Computed':f'{actual:.4f}',
                               'Match':abs(actual-expected)<1e-7})
    for m,pair,expected in [('Baichuan2-13B','M vs. D',0.4855),
                             ('Baichuan2-13B','D vs. C',0.2789),
                             ('Qwen2.5-14B','M vs. D',0.9819),
                             ('Qwen2.5-14B','D vs. C',0.6899)]:
        r=b4[(b4.Model==m)&(b4.Comparison==pair)].iloc[0]
        actual=round(float(r['p_holm_within_model_6']),4)
        check_rows.append({'Source':'Table B4', 'Item':m+' / '+pair,
                           'Expected':f'{expected:.4f}', 'Computed':f'{actual:.4f}',
                           'Match':abs(actual-expected)<1e-7})
    return pd.DataFrame(check_rows)


def main():
    data={}
    for model, aliases in MODEL_FILES.items():
        file=find_one_workbook(model,aliases)
        data[model]=read_one_model(model,file)
        print(f'Read {model}: {file.name} / {len(data[model])} questions')
    pooled=aggregate_across_models(data)
    fig7=figure7_statistics(pooled)
    tab3=table3_statistics(pooled)
    b3=table_b3_statistics(data)
    b4_long,b4_wide=table_b4_statistics(data)
    reconciliation=check_against_manuscript(fig7,tab3,b3,b4_long)
    methods=pd.DataFrame([
        ('Data unit','The pooled analyses aggregate each question over four models (one observation per question).'),
        ('Task name normalization','Medication Guidance, Diet advise, Complication management, Patient consultation mapped to M/D/C/P.'),
        ('Figure 7: summary','Mean score +/- standard error across the 15 question-level averaged scores for each task.'),
        ('Figure 7: test','Exact paired two-sided Wilcoxon signed-rank tests; stars use Holm p adjusted across 4 tasks.'),
        ('Figure 7: stars','*** p<0.001; ** p<0.01; * p<0.05 based on Holm-adjusted p.'),
        ('Table 3','Each question\'s RA-LLM minus LLM gain averaged across 4 models; two-sided Mann-Whitney U between task groups; Holm across 6 comparisons.'),
        ('Table B3','Paired two-sided exact Wilcoxon within each model/task; Holm across 4 tasks WITHIN EACH model.'),
        ('Table B4','Gain = RA-LLM minus LLM for each question/model; two-sided Mann-Whitney U pairwise between tasks, Holm across 6 pairs WITHIN EACH model.'),
        ('Mean and variability','Means use unrounded original numeric scores; SD is sample standard deviation (ddof=1); SE=SD/sqrt(n).'),
        ('p columns','Both raw and Holm-adjusted numeric p-values are retained. Manuscript tables 3/B3/B4 use adjusted p.'),
        ('Design note','Group comparisons across different tasks are independent-group comparisons, not paired comparisons.'),
    ],columns=['Analysis','Description'])
    out_xlsx=BASE/OUTPUT_XLSX
    with pd.ExcelWriter(out_xlsx,engine='openpyxl') as wr:
        fig7.to_excel(wr, sheet_name='Figure7_summary',index=False)
        tab3.to_excel(wr,sheet_name='Table3_pairwise',index=False)
        b3.to_excel(wr,sheet_name='TableB3_model_paired',index=False)
        b4_wide.to_excel(wr,sheet_name='TableB4_published',index=False)
        b4_long.to_excel(wr,sheet_name='TableB4_details',index=False)
        pooled.to_excel(wr,sheet_name='Pooled_question_data',index=False)
        pd.concat(data.values(),ignore_index=True).to_excel(
            wr,sheet_name='Model_question_data',index=False)
        reconciliation.to_excel(wr,sheet_name='Manuscript_check',index=False)
        methods.to_excel(wr,sheet_name='Methods',index=False)
        style_workbook(wr.book)
    plot_figure7(fig7, BASE/OUTPUT_PNG, BASE/OUTPUT_PDF)
    print('\nFIGURE 7 (p-values in figure STARS are Holm-adjusted):')
    for _,r in fig7.iterrows():
        print(f"  {r['Task']}: LLM {r['LLM_mean']:.4f} +/- {r['LLM_SE']:.4f} SE; "
              f"RA-LLM {r['RA_LLM_mean']:.4f} +/- {r['RA_LLM_SE']:.4f} SE; "
              f"p_raw={fmt_p(r['p_raw'])} p_holm={fmt_p(r['p_holm_across_4_tasks'])} "
              f"star={r['Figure7_star']}")
    print('\nTABLE 3:')
    for _,r in tab3.iterrows():
        print(f"  {r['Comparison']}: difference {r['Difference_in_mean_gain']:+.4f}; "
              f"p_holm={r['p_holm_6_comparisons']:.4f}")
    print('\nMANUSCRIPT CHECK:',len(reconciliation),'checks /',int(reconciliation.Match.sum()),'matched')
    if not reconciliation.Match.all():
        print(reconciliation[~reconciliation.Match].to_string(index=False))
    print('\nOutput files:')
    for name in (OUTPUT_XLSX,OUTPUT_PNG,OUTPUT_PDF):
        print(' ',BASE/name)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise
