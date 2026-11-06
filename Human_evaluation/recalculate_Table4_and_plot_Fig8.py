"""Calculate Table 4 and draw Fig. 8 (mean ± SE).

Dependencies: pip install openpyxl matplotlib
Run: python recalculate_Table4_and_plot_Fig8.py
Input: Human_evulation-point(3).xlsx, Sheet1, beside this script.
Override input: --input "path/to/ratings.xlsx"
Table 4: pooled mean, sample SD, ICC(2,5), absolute agreement.
Fig. 8: SE = sample SD of the 20 question means / sqrt(20).
Each question mean averages five evaluators; SE conditions on these evaluators.
"""
import argparse
from collections import Counter
from math import sqrt, isclose, ceil, floor
from statistics import mean, stdev
from pathlib import Path
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.properties import CalcProperties
TASKS = [('M', 'Medication guidance'), ('D', 'Diet advice'), ('C', 'Complication management'), ('P', 'Patient consultation')]
DIMENSIONS = [('Completeness', ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']), ('Clarity', ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']), ('Correctness', ['ZQ1', 'ZQ2', 'ZQ3', 'ZQ4', 'ZQ5']), ('Usefulness', ['KY1', 'KY2', 'KY3', 'KY4', 'KY5'])]
QUESTION_RE = re.compile('^([MDCP])([LMH])(\\d{2})$')
HEADER = ['编号'] + [field for _, fields in DIMENSIONS for field in fields]
NAVY = '1C3045'
TEAL = '1A7184'
LIGHT = 'E9F2F3'
LIGHT_GREY = 'F2F5F8'
GREEN = '08744F'
GREY = '617182'
ORANGE = '9A5A12'
COMPLEXITY_LEVELS = ('L', 'M', 'H')
FIGURE_COLORS = {'Completeness': '#2C91E0', 'Clarity': '#3ABF99', 'Correctness': '#F0A73A', 'Usefulness': '#595959'}

def extract_ratings(input_path, sheet_name='Sheet1'):
    workbook = load_workbook(input_path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f'Worksheet {sheet_name!r} not found.')
        sheet = workbook[sheet_name]
        rows = sheet.iter_rows(values_only=True)
        header = [str(value).strip() if value is not None else ''
                  for value in next(rows, ())]
        columns = {value: i for i, value in enumerate(header) if value}
        id_headers = [name for name in ('Dimension', '编号') if name in columns]
        if len(id_headers) != 1:
            raise ValueError("Expected one question ID column named 'Dimension' or '编号'.")
        id_header = id_headers[0]
        required = [id_header] + [field for _, fields in DIMENSIONS for field in fields]
        missing = [field for field in required if field not in columns]
        if missing:
            raise ValueError(f'Missing required columns: {missing}')
        if any((header.count(field) != 1 for field in required)):
            raise ValueError('A required column header appears more than once.')
        records = []
        ids_seen = set()
        for excel_row, row in enumerate(rows, start=2):
            identifier = row[columns[id_header]]
            if identifier is None or str(identifier).strip() == '':
                continue
            identifier = str(identifier).strip().upper()
            match = QUESTION_RE.fullmatch(identifier)
            if not match:
                raise ValueError(f'Row {excel_row}: invalid question ID {identifier!r}.')
            if identifier in ids_seen:
                raise ValueError(f'Duplicate question ID: {identifier}.')
            ids_seen.add(identifier)
            scores = {}
            for dimension, fields in DIMENSIONS:
                ratings = []
                for field in fields:
                    value = row[columns[field]]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (1, 2, 3, 4, 5):
                        raise ValueError(f'{identifier}, {field}: expected an integer score 1-5; received {value!r}. Missing ratings are not discarded.')
                    ratings.append(int(value))
                scores[dimension] = ratings
            records.append({'id': identifier, 'task': match[1], 'complexity': match[2], 'scores': scores})
        counts = Counter(((r['task'], r['complexity']) for r in records))
        if len(records) != 60 or any((counts[task, level] != 5 for task in [task for task, _ in TASKS] for level in COMPLEXITY_LEVELS)):
            raise ValueError(f'Expected 60 questions, with five questions in each task x complexity cell. Observed counts: {dict(counts)}')
        task_order = {task: i for i, (task, _) in enumerate(TASKS)}
        complexity_order = {level: i for i, level in enumerate(COMPLEXITY_LEVELS)}
        records.sort(key=lambda r: (task_order[r['task']], complexity_order[r['complexity']], int(r['id'][2:])))
        return [(r['id'], [score for dimension, _ in DIMENSIONS for score in r['scores'][dimension]]) for r in records]
    finally:
        workbook.close()

def calculate(ratings):
    n = len(ratings)
    k = 5
    all_scores = [value for question_scores in ratings for value in question_scores]
    total = sum(all_scores)
    sum_of_squares = sum((value ** 2 for value in all_scores))
    n_ratings = n * k
    average = total / n_ratings
    ss_total = sum_of_squares - total ** 2 / n_ratings
    sd_sample = sqrt(max(0, ss_total / (n_ratings - 1)))
    question_means = [sum(row) / k for row in ratings]
    rater_means = [sum((row[j] for row in ratings)) / n for j in range(k)]
    sum_row_means_squared = sum((value ** 2 for value in question_means))
    ss_questions = k * sum_row_means_squared - total ** 2 / n_ratings
    ss_raters = n * sum((value ** 2 for value in rater_means)) - total ** 2 / n_ratings
    ss_error = ss_total - ss_questions - ss_raters
    ms_questions = ss_questions / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_questions + (ms_raters - ms_error) / n
    if denominator == 0:
        raise ValueError('ICC cannot be calculated: zero denominator')
    icc_2k = (ms_questions - ms_error) / denominator
    return {'n': n, 'k': k, 'count': n_ratings, 'sum': total, 'mean': average, 'sd': sd_sample, 'sum_sq': sum_of_squares, 'sum_question_mean_sq': sum_row_means_squared, 'rater_means': rater_means, 'ss_total': ss_total, 'ss_questions': ss_questions, 'ss_raters': ss_raters, 'ss_error': ss_error, 'ms_questions': ms_questions, 'ms_raters': ms_raters, 'ms_error': ms_error, 'icc': icc_2k}

def cache_xlsx_formulas(path, cached):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    ET.register_namespace('', ns)
    fd, temp_name = tempfile.mkstemp(suffix='.xlsx', dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(path, 'r') as reader, zipfile.ZipFile(temp_name, 'w') as writer:
            for entry in reader.infolist():
                payload = reader.read(entry.filename)
                match = re.fullmatch('xl/worksheets/sheet(\\d+)\\.xml', entry.filename)
                if match:
                    index = int(match.group(1))
                    sheet_values = cached.get(index, {})
                    if sheet_values:
                        root = ET.fromstring(payload)
                        for cell in root.findall('.//{%s}c' % ns):
                            address = cell.get('r')
                            if address not in sheet_values:
                                continue
                            value = sheet_values[address]
                            value_node = cell.find('{%s}v' % ns)
                            if value_node is None:
                                value_node = ET.SubElement(cell, '{%s}v' % ns)
                            if isinstance(value, str):
                                cell.set('t', 'str')
                                value_node.text = value
                            else:
                                cell.attrib.pop('t', None)
                                value_node.text = str(value)
                        payload = ET.tostring(root, encoding='utf-8', xml_declaration=True)
                writer.writestr(entry, payload)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

def style_header(sheet, row, start=1, end=6):
    for cells in sheet.iter_rows(min_row=row, max_row=row, min_col=start, max_col=end):
        for cell in cells:
            cell.fill = PatternFill('solid', fgColor=NAVY)
            cell.font = Font(name='Aptos', size=10, bold=True, color='FFFFFF')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    sheet.row_dimensions[row].height = 34

def export_table2(records, output_path, source_path, complexity_summary):
    book = Workbook()
    main = book.active
    main.title = 'Table4_论文表格'
    calc = book.create_sheet('计算明细')
    raw = book.create_sheet('原始评分')
    notes = book.create_sheet('统计说明')
    cache = {1: {}, 2: {}, 3: {}}
    raw.append(HEADER + [f'{dimension}_question_mean' for dimension, _ in DIMENSIONS])
    for rownum, (question, scores) in enumerate(records, start=2):
        raw.append([question] + scores)
        for j in range(4):
            first_col = get_column_letter(2 + 5 * j)
            last_col = get_column_letter(6 + 5 * j)
            average_col = get_column_letter(22 + j)
            raw[f'{average_col}{rownum}'] = f'=AVERAGE({first_col}{rownum}:{last_col}{rownum})'
            cache[3][f'{average_col}{rownum}'] = sum(scores[5 * j:5 * (j + 1)]) / 5
    style_header(raw, 1, 1, 25)
    raw.freeze_panes = 'B2'
    raw.auto_filter.ref = 'A1:U61'
    raw.column_dimensions['A'].width = 13
    for col in range(2, 26):
        letter = get_column_letter(col)
        raw.column_dimensions[letter].width = 12 if col <= 21 else 25
        for row in range(2, 62):
            raw.cell(row, col).font = Font(name='Aptos', size=10, color=GREEN if col <= 21 else '152B40')
    raw.sheet_view.showGridLines = False
    detail_columns = ['Task', 'Dimension', 'N_questions', 'N_raters', 'N_scores', 'Sum', 'Mean', 'SD_sample', 'SumSq', 'SumQuestionMeanSq', 'Rater1_mean', 'Rater2_mean', 'Rater3_mean', 'Rater4_mean', 'Rater5_mean', 'SS_total', 'SS_questions', 'SS_raters', 'SS_error', 'MS_questions', 'MS_raters', 'MS_error', 'ICC_2k', 'Mean ± SD']
    calc.append(detail_columns)
    style_header(calc, 1, 1, len(detail_columns))
    calc.freeze_panes = 'C2'
    calc.sheet_view.showGridLines = False
    calc.column_dimensions['A'].width = 27
    calc.column_dimensions['B'].width = 18
    for col in range(3, len(detail_columns) + 1):
        calc.column_dimensions[get_column_letter(col)].width = 17 if col != 24 else 20
    calc.row_dimensions[1].height = 42
    group_ranges = {}
    for index, (_, label) in enumerate(TASKS):
        first = 2 + index * 15
        group_ranges[label] = (first, first + 14)
    group_ranges['Overall'] = (2, 61)
    detail_rows = {}
    for task_name in [name for _, name in TASKS] + ['Overall']:
        first_row, last_row = group_ranges[task_name]
        group_records = [pair for pair in records if task_name == 'Overall' or pair[0][0] == next((code for code, label in TASKS if label == task_name))]
        for dim_index, (dimension, _) in enumerate(DIMENSIONS):
            rownum = calc.max_row + 1
            detail_rows[task_name, dimension] = rownum
            values = [scores[5 * dim_index:5 * (dim_index + 1)] for _, scores in group_records]
            result = calculate(values)
            begin = get_column_letter(2 + 5 * dim_index)
            end = get_column_letter(6 + 5 * dim_index)
            question_mean_col = get_column_letter(22 + dim_index)
            rg = f"'原始评分'!${begin}${first_row}:${end}${last_row}"
            avg_rg = f"'原始评分'!${question_mean_col}${first_row}:${question_mean_col}${last_row}"
            formulas = {'C': (f"=COUNTA('原始评分'!$A${first_row}:$A${last_row})", result['n']), 'E': (f'=COUNT({rg})', result['count']), 'F': (f'=SUM({rg})', result['sum']), 'G': (f'=AVERAGE({rg})', result['mean']), 'H': (f'=STDEV.S({rg})', result['sd']), 'I': (f'=SUMSQ({rg})', result['sum_sq']), 'J': (f'=SUMSQ({avg_rg})', result['sum_question_mean_sq']), 'P': (f'=I{rownum}-F{rownum}^2/E{rownum}', result['ss_total']), 'Q': (f'=D{rownum}*J{rownum}-F{rownum}^2/E{rownum}', result['ss_questions']), 'R': (f'=C{rownum}*SUMSQ(K{rownum}:O{rownum})-F{rownum}^2/E{rownum}', result['ss_raters']), 'S': (f'=P{rownum}-Q{rownum}-R{rownum}', result['ss_error']), 'T': (f'=Q{rownum}/(C{rownum}-1)', result['ms_questions']), 'U': (f'=R{rownum}/(D{rownum}-1)', result['ms_raters']), 'V': (f'=S{rownum}/((C{rownum}-1)*(D{rownum}-1))', result['ms_error']), 'W': (f'=(T{rownum}-V{rownum})/(T{rownum}+(U{rownum}-V{rownum})/C{rownum})', result['icc']), 'X': (f'=TEXT(G{rownum},"0.00")&" ± "&TEXT(H{rownum},"0.00")', f"{result['mean']:.2f} ± {result['sd']:.2f}")}
            calc[f'A{rownum}'] = task_name
            calc[f'B{rownum}'] = dimension
            calc[f'D{rownum}'] = 5
            for j in range(5):
                letter = get_column_letter(2 + 5 * dim_index + j)
                cell = f'{get_column_letter(11 + j)}{rownum}'
                formulas[get_column_letter(11 + j)] = (f"=AVERAGE('原始评分'!${letter}${first_row}:${letter}${last_row})", result['rater_means'][j])
            for col, (formula, value) in formulas.items():
                address = f'{col}{rownum}'
                calc[address] = formula
                cache[2][address] = value
                calc[address].number_format = '0.0000' if col not in ('C', 'D', 'E', 'F', 'I', 'J', 'X') else 'General'
            if task_name == 'Overall':
                for cell in calc[rownum][:24]:
                    cell.fill = PatternFill('solid', fgColor=LIGHT)
                    cell.font = Font(name='Aptos', bold=True, color=NAVY)
            elif dim_index == 0:
                for cell in calc[rownum][:24]:
                    cell.fill = PatternFill('solid', fgColor=LIGHT_GREY)
    calc.auto_filter.ref = f'A1:X{calc.max_row}'
    main.merge_cells('A1:F1')
    main['A1'] = 'Table 4. Result of human evaluation — recalculated'
    main['A1'].font = Font(name='Aptos Display', size=16, bold=True, color='FFFFFF')
    main['A1'].alignment = Alignment(vertical='center')
    main['A1'].fill = PatternFill('solid', fgColor=NAVY)
    main.row_dimensions[1].height = 36
    main.append([])
    for col, label in enumerate(['Task', 'Indicator'] + [dim for dim, _ in DIMENSIONS], start=1):
        main.cell(3, col, label)
    style_header(main, 3, 1, 6)
    main.column_dimensions['A'].width = 29
    main.column_dimensions['B'].width = 14
    for col in 'CDEF':
        main.column_dimensions[col].width = 22
    main.freeze_panes = 'C4'
    main.sheet_view.showGridLines = False
    for index, task_name in enumerate([name for _, name in TASKS] + ['Overall']):
        pooled_row = 4 + index * 2
        icc_row = pooled_row + 1
        main.merge_cells(start_row=pooled_row, start_column=1, end_row=icc_row, end_column=1)
        main.cell(pooled_row, 1, task_name)
        main.cell(pooled_row, 2, 'Pooled')
        main.cell(icc_row, 2, 'ICC')
        main.row_dimensions[pooled_row].height = 30
        main.row_dimensions[icc_row].height = 26
        for dim_index, (dimension, _) in enumerate(DIMENSIONS):
            target_col = dim_index + 3
            detail_row = detail_rows[task_name, dimension]
            pooled = main.cell(pooled_row, target_col, f"='计算明细'!X{detail_row}")
            icc = main.cell(icc_row, target_col, f"=ROUND('计算明细'!W{detail_row},3)")
            result = calculate([scores[5 * dim_index:5 * (dim_index + 1)] for question, scores in records if task_name == 'Overall' or question[0] == next((code for code, label in TASKS if label == task_name))])
            cache[1][pooled.coordinate] = f"{result['mean']:.2f} ± {result['sd']:.2f}"
            cache[1][icc.coordinate] = round(result['icc'], 3)
            icc.number_format = '0.000'
        for row in (pooled_row, icc_row):
            for col in range(1, 7):
                cell = main.cell(row, col)
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                if index == 4:
                    cell.fill = PatternFill('solid', fgColor=LIGHT)
                    cell.font = Font(name='Aptos', size=10, color=NAVY, bold=True)
                elif row == pooled_row and index % 2 == 0:
                    cell.fill = PatternFill('solid', fgColor='F7FAFC')
                if row == icc_row:
                    cell.border = Border(bottom=Side(style='hair', color='D4DFE4'))
    main.merge_cells('A15:F15')
    main['A15'] = 'Note. Mean ± sample SD (1–5 scale). Pooled: 75 ratings per task; 300 overall. ICC(2,5): two-way random-effects, absolute agreement, average measures.'
    main['A15'].alignment = Alignment(wrap_text=True, vertical='center')
    main['A15'].font = Font(name='Aptos', color=GREY, italic=True, size=10)
    main.row_dimensions[15].height = 45
    explanations = [('Source file', f'{source_path.name}; original expert scores are copied unchanged into 原始评分.'), ('Question mapping', 'M=Medication guidance; D=Diet advice; C=Complication management; P=Patient consultation.'), ('Design', '60 questions × 5 experts × 4 dimensions; each task has 15 questions.'), ('Completeness', 'Q1–Q5 (original columns B–F).'), ('Clarity', 'Z1–Z5 (original columns G–K).'), ('Correctness', 'ZQ1–ZQ5 (original columns L–P).'), ('Usefulness', 'KY1–KY5 (original columns Q–U).'), ('Pooled mean', 'Arithmetic mean of all 5 × N individual ratings, without mixing dimensions. Pooled ratings describe the score distribution; they are not 5 × N independent questions.'), ('SD', 'Sample SD = sqrt(Σ(x − mean)^2 / (5 × N − 1)), not standard error (SE).'), ('ICC formulation', 'ICC(2,k) two-way random effects; absolute agreement; average of k=5 raters.'), ('ICC formula', '(MS_questions − MS_error) / [MS_questions + (MS_raters − MS_error) / N_questions].'), ('Missing data', 'A missing/invalid score or an unexpected number of questions raises an error; no silent deletion.'), ('Calculated sheets', 'Table4_论文表格 references 计算明细, which references 原始评分.'), ('Changes to data', 'Editing score values in 原始评分 lets Excel recalculate. For new IDs, strata or question counts, rerun the Python script.'), ('Data sharing', '原始评分 includes anonymous question IDs and numerical ratings, not patient records.')]
    notes.append(['Method', 'Description'])
    style_header(notes, 1, 1, 2)
    notes.column_dimensions['A'].width = 24
    notes.column_dimensions['B'].width = 110
    notes.freeze_panes = 'A2'
    notes.sheet_view.showGridLines = False
    for i, (label, value) in enumerate(explanations, start=2):
        notes.cell(i, 1, label).font = Font(name='Aptos', bold=True, color=NAVY)
        notes.cell(i, 2, value).font = Font(name='Aptos', color=GREY)
        notes.cell(i, 2).alignment = Alignment(wrap_text=True, vertical='center')
        notes.row_dimensions[i].height = 42
        if i % 2 == 0:
            for col in (1, 2):
                notes.cell(i, col).fill = PatternFill('solid', fgColor=LIGHT_GREY)
    add_figure_data_sheet(book, cache, records, complexity_summary)
    for label, description in [('Fig. 8 grouping', 'Second ID character: L=low, M=medium, H=high complexity; each group has 20 questions and 100 ratings per dimension.'), ('Fig. 8 statistical unit', 'Average the five ratings for each question first. SE = sample SD of the 20 question means / sqrt(20). This conditions on the five evaluators and does not model sampling of new evaluators.'), ('Fig. 8 error bars', 'Mean +/- one SE, calculated from question means.'), ('Figure updates', 'Fig8_绘图数据 contains formulas linked to 原始评分. Rerun this Python script after editing input ratings to regenerate the image.')]:
        row = notes.max_row + 1
        notes.cell(row, 1, label).font = Font(name='Aptos', bold=True, color=NAVY)
        notes.cell(row, 2, description).font = Font(name='Aptos', color=GREY)
        notes.cell(row, 2).alignment = Alignment(wrap_text=True, vertical='center')
        notes.row_dimensions[row].height = 42
    book.calculation = CalcProperties(calcMode='auto', fullCalcOnLoad=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    book.save(output_path)
    cache_xlsx_formulas(output_path, cache)
    checked = load_workbook(output_path, data_only=True, read_only=True)
    summary = checked['Table4_论文表格']
    for task_index, (code, label) in enumerate(TASKS + [(None, 'Overall')]):
        selected = [scores for question, scores in records if code is None or question[0] == code]
        for dim_index, (dimension, _) in enumerate(DIMENSIONS):
            values = [row[5 * dim_index:5 * (dim_index + 1)] for row in selected]
            flat = [x for row in values for x in row]
            expected_text = f'{mean(flat):.2f} ± {stdev(flat):.2f}'
            actual = summary.cell(4 + 2 * task_index, 3 + dim_index).value
            if actual != expected_text:
                raise RuntimeError(f'Cached result mismatch for {label}/{dimension}: {actual} vs {expected_text}')
            detail_row = 2 + task_index * 4 + dim_index
            if not isclose(checked['计算明细'].cell(detail_row, 8).value, stdev(flat), rel_tol=1e-12):
                raise RuntimeError(f'SD verification failed for {label}/{dimension}')
            if not isclose(checked['计算明细'].cell(detail_row, 23).value, calculate(values)['icc'], rel_tol=1e-12):
                raise RuntimeError(f'ICC export verification failed for {label}/{dimension}')
    figure_data = checked['Fig8_绘图数据']
    for level_index, level in enumerate(COMPLEXITY_LEVELS):
        for dimension_index, (dimension, _) in enumerate(DIMENSIONS):
            row = 4 + level_index * 4 + dimension_index
            expected = complexity_summary[level][dimension]
            for column, key in ((6, 'mean'), (7, 'sd'), (9, 'se')):
                if not isclose(figure_data.cell(row, column).value, expected[key], abs_tol=1e-12):
                    raise RuntimeError(f'Fig. 8 Excel mismatch: {level}/{dimension}/{key}')
    checked.close()
    print(f'Finished: {output_path}')
    print('Verified all 20 mean/SD results. SD uses denominator n_ratings - 1 (Excel STDEV.S).')

def choose_serif_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in ('Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'):
        if family in installed:
            return family
    return 'DejaVu Serif'

def draw_figure(summary, output_path, dpi=600):
    font_family = choose_serif_font()
    style = {'font.family': font_family, 'font.size': 12, 'axes.labelsize': 15, 'axes.labelweight': 'bold', 'axes.unicode_minus': False, 'axes.edgecolor': '#D9D9D9', 'axes.linewidth': 0.7, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.16, top=0.855)
        bar_width = 0.14
        bar_step = 0.17
        for index, (dimension, _) in enumerate(DIMENSIONS):
            colour = FIGURE_COLORS[dimension]
            positions = [group + (index - 1.5) * bar_step for group in range(3)]
            averages = [summary[level][dimension]['mean'] for level in COMPLEXITY_LEVELS]
            deviations = [summary[level][dimension]['se'] for level in COMPLEXITY_LEVELS]
            ax.bar(positions, averages, width=bar_width, color=colour, edgecolor='none', label=dimension, yerr=deviations, error_kw={'ecolor': '#858585', 'elinewidth': 0.7, 'capsize': 2.0, 'capthick': 0.7}, zorder=2)
        lower, upper = common_axis_limits(summary)
        ax.set_ylim(lower, upper)
        ax.set_xlim(-0.5, 2.5)
        ticks = [value / 5 for value in range(ceil(lower * 5), floor(upper * 5) + 1)]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xticks(range(3), COMPLEXITY_LEVELS)
        ax.set_xlabel('Complexity', labelpad=7)
        ax.set_ylabel('Score', labelpad=10)
        ax.tick_params(axis='both', which='major', direction='out', length=3, width=0.6, color='#D9D9D9', labelcolor='black', pad=7)
        ax.tick_params(axis='y', which='minor', length=2, width=0.5, color='#E2E2E2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, handlelength=0.55, handleheight=0.55, handletextpad=0.25, columnspacing=1.1, borderaxespad=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
    return (font_family, (lower, upper))

def common_axis_limits(summary):
    lower_endpoint = min((value['mean'] - value['se'] for group in summary.values() for value in group.values()))
    upper_endpoint = max((value['mean'] + value['se'] for group in summary.values() for value in group.values()))
    return (floor((lower_endpoint - 0.08) * 10) / 10, ceil((upper_endpoint + 0.08) * 10) / 10)

def summarise_by_complexity(records):
    summary = {}
    for level in COMPLEXITY_LEVELS:
        summary[level] = {}
        for dimension_index, (dimension, _) in enumerate(DIMENSIONS):
            ratings = [scores[5 * dimension_index:5 * (dimension_index + 1)] for question, scores in records if question[1] == level]
            result = calculate(ratings)
            question_means = [mean(values) for values in ratings]
            n_questions = len(question_means)
            if not isclose(result['mean'], mean(question_means), abs_tol=1e-12):
                raise ValueError('Pooled mean and mean of question means do not match.')
            sd_question = stdev(question_means)
            se = sd_question / sqrt(n_questions)
            result.update({'sd_question': sd_question, 'se': se})
            summary[level][dimension] = result
    return summary

def add_figure_data_sheet(book, cache, records, summary):
    sheet = book.create_sheet('Fig8_绘图数据')
    sheet_index = book.sheetnames.index(sheet.title) + 1
    cache[sheet_index] = {}
    sheet.merge_cells('A1:J1')
    sheet['A1'] = 'Fig. 8. Human evaluation by question complexity'
    sheet['A1'].font = Font(name='Aptos Display', size=16, bold=True, color='FFFFFF')
    sheet['A1'].fill = PatternFill('solid', fgColor=NAVY)
    sheet['A1'].alignment = Alignment(vertical='center')
    sheet.row_dimensions[1].height = 36
    headers = ['Complexity', 'Dimension', 'N_questions', 'N_raters', 'N_scores', 'Mean', 'Pooled sample SD', 'SD of question means', 'SE', 'Mean ± SE']
    for column, value in enumerate(headers, start=1):
        sheet.cell(3, column, value)
    style_header(sheet, 3, 1, 10)
    sheet.freeze_panes = 'C4'
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions['A'].width = 15
    sheet.column_dimensions['B'].width = 21
    for column in 'CDEFHI':
        sheet.column_dimensions[column].width = 15
    sheet.column_dimensions['G'].width = 18
    sheet.column_dimensions['J'].width = 21
    sheet.row_dimensions[3].height = 42
    for level_index, level in enumerate(COMPLEXITY_LEVELS):
        row_blocks = []
        for task_code, _ in TASKS:
            rows = [index + 2 for index, (question, _) in enumerate(records) if question[0] == task_code and question[1] == level]
            if rows != list(range(rows[0], rows[-1] + 1)):
                raise ValueError('Noncontiguous raw-data block; cannot build figure formulas.')
            row_blocks.append((rows[0], rows[-1]))
        for dimension_index, (dimension, _) in enumerate(DIMENSIONS):
            row = 4 + level_index * 4 + dimension_index
            result = summary[level][dimension]
            first_column = get_column_letter(2 + 5 * dimension_index)
            last_column = get_column_letter(6 + 5 * dimension_index)
            rating_ranges = ','.join((f"'原始评分'!${first_column}${first}:${last_column}${last}" for first, last in row_blocks))
            question_ranges = ','.join((f"'原始评分'!$A${first}:$A${last}" for first, last in row_blocks))
            question_mean_column = get_column_letter(22 + dimension_index)
            question_mean_ranges = ','.join((f"'原始评分'!${question_mean_column}${first}:${question_mean_column}${last}" for first, last in row_blocks))
            sheet.cell(row, 1, level)
            sheet.cell(row, 2, dimension)
            sheet.cell(row, 4, result['k'])
            formulas = {'C': (f'=COUNTA({question_ranges})', result['n']), 'E': (f'=COUNT({rating_ranges})', result['count']), 'F': (f'=AVERAGE({question_mean_ranges})', result['mean']), 'G': (f'=STDEV.S({rating_ranges})', result['sd']), 'H': (f'=STDEV.S({question_mean_ranges})', result['sd_question']), 'I': (f'=H{row}/SQRT(C{row})', result['se']), 'J': (f'=TEXT(F{row},"0.00")&" ± "&TEXT(I{row},"0.00")', f"{result['mean']:.2f} ± {result['se']:.2f}")}
            for column, (formula, value) in formulas.items():
                address = f'{column}{row}'
                sheet[address] = formula
                cache[sheet_index][address] = value
                if column in 'FGHI':
                    sheet[address].number_format = '0.0000'
            for cell in sheet[row][:10]:
                cell.font = Font(name='Aptos', size=10, color=NAVY)
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                cell.border = Border(bottom=Side(style='hair', color='D4DFE4'))
                if level_index % 2 == 0:
                    cell.fill = PatternFill('solid', fgColor=LIGHT_GREY)
            sheet.row_dimensions[row].height = 28
    sheet.merge_cells('A17:J17')
    sheet['A17'] = 'Note. L = low, M = medium, H = high complexity. Each group contains 20 questions and five raters. SE is calculated from the 20 question means, after averaging the five ratings per question. SE estimates condition on these five raters.'
    sheet['A17'].font = Font(name='Aptos', size=10, italic=True, color=GREY)
    sheet['A17'].alignment = Alignment(wrap_text=True, vertical='center')
    sheet.row_dimensions[17].height = 48

def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=directory / 'Human_evulation-point(3).xlsx')
    parser.add_argument('--sheet', default='Sheet1')
    parser.add_argument('--output-dir', type=Path, default=directory / 'Table4_Fig8_results')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Input workbook not found: {args.input}')
    if args.dpi < 72:
        parser.error('--dpi must be at least 72.')
    excel_path = args.output_dir / 'Table4_and_Fig8_results.xlsx'
    figure_path = args.output_dir / 'Fig8_SE.png'
    if args.input.resolve() == excel_path.resolve():
        parser.error('The output must not overwrite the input workbook.')
    print(f'Input: {args.input.resolve()} [{args.sheet}]')
    records = extract_ratings(args.input, args.sheet)
    summary = summarise_by_complexity(records)
    export_table2(records, excel_path, args.input, summary)
    draw_figure(summary, figure_path, args.dpi)
    print(f'Finished: {figure_path}')
if __name__ == '__main__':
    main()
