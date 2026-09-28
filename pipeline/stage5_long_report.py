import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import progress
from stage5_long_analyze import interval, percent
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / 'results/stage5_long25'
FIGURES = EXP / 'results/revision_figures'
LABELS = [('zero', 'Zero'), ('random_1_30_seed7', 'Random'), ('center30', 'Center-30'), ('center100', 'Center-100')]

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = json.loads((OUT / 'analysis.json').read_text())
    verified = json.loads((OUT / 'verification.json').read_text())
    assert verified['status'] == 'passed'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.8), sharex=True, layout='constrained')
    for ax, (field, label) in zip(axes.flat, LABELS):
        paths = [OUT / r['history'] for r in data['rows'] if r['field'] == field and r['iters'] == 5000]
        series, grids = ([], [])
        for path in paths:
            hist = list(csv.DictReader(path.open()))
            grids.append([int(h['iter']) for h in hist])
            series.append([float(h['gbest_variance']) for h in hist])
        assert len(series) == 25 and all((x == grids[0] for x in grids))
        q1, med, q3 = np.quantile(np.array(series), [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(grids[0], q1, q3, color='#0072B2', alpha=0.2, linewidth=0, label='Sample IQR')
        ax.plot(grids[0], med, color='#0072B2', linewidth=1.5, label='Sample median')
        ax.axvline(500, color='0.4', linestyle='--', linewidth=0.9)
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100, 500, 5000], ['1', '10', '100', '500', '5000'])
        ax.set_title(label, loc='left', fontsize=11)
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('Variance of selected allocation', fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('Iteration')
    for extension in ['pdf', 'png']:
        path = FIGURES / f'pso_long25_convergence.{extension}'
        fig.savefig(path, dpi=300)
    plt.close(fig)
    rows = []
    for field, label in LABELS:
        s = next((s for s in data['summary'] if s['field'] == field and s['iters'] == 5000))
        rs = [r for r in data['rows'] if r['field'] == field and r['iters'] == 5000]
        q1 = sorted((Fraction(r['excess_low_exact']) for r in rs))[6]
        q3 = sorted((Fraction(r['excess_high_exact']) for r in rs))[18]
        rows.append(f"{label} & ${s['median']}$ & ${interval(q1, q3)}$ & ${s['range']}$ & ${s['median_ci']}$" + '\\\\')
    table = "\\begin{table}[!tbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\caption{Twenty-five independent PSO runs of $5000$ iterations for each fixed\ninitial field. All values are percentages of excess over the integer optimum\nat each run's final budget. The sample median, quartile envelope and range\ninclude the uncertainty in that optimum through exact lower and upper bounds.\nThe quartile envelope extends from the lower bound on sample Q1 to the upper\nbound on sample Q3. The last column instead quantifies sampling uncertainty\nin the population median: under independent runs, the eighth ordered lower\nendpoint and eighteenth ordered upper endpoint give marginal coverage of at\nleast $1-2\\sum_{j=0}^{7}\\binom{25}{j}2^{-25}=0.9567147\\ldots$.\nNo simultaneous coverage across fields is claimed. All endpoints are rounded\noutward to one decimal place.}\n\\label{tab:pso-long25-si}\n\\begin{tabular}{lrrrr}\n\\toprule\nInstance & Sample median & Q1--Q3 envelope & Range & Median confidence interval\\\\\n\\midrule\n" + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (FIGURES / 'tableS12.tex').write_text(table)
    print('Wrote convergence figures, Table S12 fragment and experiment report.')
if __name__ == '__main__':
    main()
