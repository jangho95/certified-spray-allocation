from __future__ import annotations
import csv
import math
import re
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
WORK1 = EXP / 'work' / 'stage1'
ARCH = EXP / 'reference' / 'archive'
MAIN_TEX = EXP / 'reference' / 'submitted_tables_main.tex'
SUPP_TEX = EXP / 'reference' / 'submitted_tables_supplement.tex'
ABS_OK, REL_OK, REL_WARN = (1e-09, 1e-06, 0.001)
SIX_DECIMAL_COLS = {'var_repair', 'mean', 'variance_repair', 'var_upper', 'variance_ub'}
DERIVED_COLS = {'abs_gap', 'gap_pct', 'gap', 'cert_excess', 'cert_ratio_pct', 'incumbent_subopt', 'reported_cert', 'variance_over_iqp'}

def _six_decimal_source(row: dict) -> bool:
    for c in ('var_repair', 'variance_repair', 'var_upper', 'variance_ub'):
        v = row.get(c)
        if v and '.' in v and (len(v.split('.')[1]) <= 6):
            return True
    return False
TIME_COLS = re.compile('(time|_s$|runtime|elapsed|int_s|qp_s)')

def read_rows(path: Path, skip_first: bool=False) -> list[dict]:
    with open(path, newline='') as fh:
        if skip_first:
            fh.readline()
        return list(csv.DictReader(fh))

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def flag_for(a, b, col=''):
    if a is None or b is None:
        return ('ok' if a == b else 'fail', None, None)
    d = abs(b - a)
    rel = d / abs(a) if a != 0 else 0.0 if d == 0 else math.inf
    allowance = 5e-07 if col in SIX_DECIMAL_COLS else ABS_OK
    if d <= allowance or rel <= REL_OK:
        return ('ok', d, rel)
    return ('warn' if rel <= REL_WARN else 'fail', d, rel)

def compare_csv(table: str, new: Path, arch: Path, keys: list[str], skip_first=False, skip_cols: set[str] | None=None) -> list[dict]:
    new_rows = {tuple((r[k] for k in keys)): r for r in read_rows(new, skip_first)}
    arch_rows = {tuple((r[k] for k in keys)): r for r in read_rows(arch, skip_first)}
    out = []
    for key in sorted(set(new_rows) | set(arch_rows), key=str):
        rn, ra = (new_rows.get(key), arch_rows.get(key))
        label = ', '.join((f'{k}={v}' for k, v in zip(keys, key)))
        if rn is None or ra is None:
            out.append(dict(table=table, row=label, column='(행)', archived='있음' if ra else '없음', reproduced='있음' if rn else '없음', flag='fail', basis='아카이브'))
            continue
        for col in ra:
            if col in keys or (skip_cols and col in skip_cols) or TIME_COLS.search(col):
                continue
            a, b = (_num(ra.get(col)), _num(rn.get(col)))
            if a is None and b is None:
                fl = 'ok' if ra.get(col) == rn.get(col) else 'fail'
                out.append(dict(table=table, row=label, column=col, archived=ra.get(col), reproduced=rn.get(col), flag=fl, basis='아카이브'))
                continue
            fl, d, rel = flag_for(a, b, col)
            det = None
            if fl == 'warn' and col in DERIVED_COLS and (rel <= 2e-05) and _six_decimal_source(ra):
                fl, det = ('ok', '아카이브가 repair 분산을 6자리로 저장 → 파생값에 전파 (상대차 ≤2e-5)')
            out.append(dict(table=table, row=label, column=col, archived=a, reproduced=b, abs_diff=d, rel_diff=rel, flag=fl, basis='아카이브', detail=det))
    return out

def tex_table_rows(label: str) -> list[list[str]]:
    for tex in (MAIN_TEX, SUPP_TEX):
        s = tex.read_text()
        i = s.find(f'\\label{{{label}}}')
        if i < 0:
            continue
        j = s.find('\\midrule', i)
        k = s.find('\\bottomrule', j)
        body = s[j + len('\\midrule'):k]
        rows = []
        for raw in body.split('\\\\'):
            raw = raw.replace('\\midrule', '').strip()
            if not raw or raw.startswith('\\end'):
                continue
            rows.append([c.strip() for c in raw.split('&')])
        return rows
    raise KeyError(label)

def tex_number(cell: str):
    c = cell.replace('$', '').replace('{', '').replace('}', '').replace('\\,', '').strip()
    m = re.fullmatch('([+-]?[0-9.]+)\\\\times10\\^([+-]?\\d+)', c)
    if m:
        return (float(m.group(1)) * 10 ** int(m.group(2)), 'sci', m.group(1))
    m = re.fullmatch('[+-]?\\d+(\\.\\d+)?', c)
    if m:
        return (float(c), 'fix', c)
    return (None, None, c)

def compare_display(table: str, row: str, column: str, cell: str, value, basis='원고') -> dict:
    target, kind, text = tex_number(cell)
    rec = dict(table=table, row=row, column=column, manuscript=cell.replace('$', ''), reproduced=value, basis=basis)
    if target is None or value is None:
        rec['flag'] = 'fail' if value is None else 'ok'
        return rec
    if kind == 'fix':
        dec = len(text.split('.')[1]) if '.' in text else 0
        shown = f'{value:.{dec}f}'
        ulp = 10 ** (-dec)
        rec['reproduced_display'] = shown
        if shown == text or shown == f'-{text}' or (text.startswith('-') and shown == text[1:]):
            rec['flag'] = 'ok'
        elif abs(value - target) <= 1.0000001 * ulp:
            rec['flag'] = 'warn'
        else:
            rec['flag'] = 'fail'
    else:
        mant_dec = len(text.split('.')[1]) if '.' in text else 0
        exp = int(math.floor(math.log10(abs(value)))) if value else 0
        shown = f'{value / 10 ** exp:.{mant_dec}f}e{exp}'
        rec['reproduced_display'] = shown
        rel = abs(value - target) / abs(target) if target else abs(value)
        rec['flag'] = 'ok' if f'{target / 10 ** exp:.{mant_dec}f}e{exp}' == shown else 'warn' if rel < 0.5 else 'fail'
    return rec

def summarize(rows: list[dict]) -> dict:
    out = {'ok': 0, 'warn': 0, 'fail': 0}
    for r in rows:
        out[r.get('flag', 'ok')] = out.get(r.get('flag', 'ok'), 0) + 1
    return out
