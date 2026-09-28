from package_paths import load_json
import csv
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
EXP = Path(__file__).resolve().parents[1]
WORK, OUT, DASH = (EXP / 'work' / 'stage5', EXP / 'results' / 'stage5', EXP / 'reports' / 'data')
SURF, TEXT, TEXT2, GRID = ('#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0')
S1, S2 = ('#2a78d6', '#eb6834')
FIELDS = [('zero', 'zero'), ('random_1_30_seed7', 'random (seed 7)'), ('center30', 'center-30'), ('center100', 'center-100')]

def history(field, iters):
    with open(WORK / f'pso_{field}_s1_it{iters}_history.csv') as fh:
        return [(int(r['iter']), float(r['gbest_variance']), int(r['gbest_budget'])) for r in csv.DictReader(fh)]

def main():
    runs = load_json((OUT / 'pso_runs.json').read_text())['rows']
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, (field, title) in zip(axes.ravel(), FIELDS):
        ax.set_facecolor(SURF)
        for sp_ in ('top', 'right'):
            ax.spines[sp_].set_visible(False)
        for sp_ in ('left', 'bottom'):
            ax.spines[sp_].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT2, labelsize=8.5)
        h5 = history(field, 5000)
        h0 = history(field, 500)
        ax.plot([h[0] for h in h5], [h[1] for h in h5], color=S1, linewidth=2, label='PSO 5000 iterations')
        ax.plot([h[0] for h in h0], [h[1] for h in h0], color=S2, linewidth=2, label='PSO 500 iterations')
        r = next((x for x in runs if x['field'] == field and x['iters'] == 5000))
        ax.axhline(r['U'], color=TEXT2, linewidth=1.5, linestyle='--')
        ax.annotate(f"V* ∈ [{r['L']:.2f}, {r['U']:.2f}] at B={r['B']}", xy=(1.2, r['U']), xytext=(0, 4), textcoords='offset points', fontsize=8, color=TEXT2)
        ax.set_xscale('log')
        ax.set_ylim(0, max((h[1] for h in h5[1:])) * 1.05)
        ax.set_title(title, color=TEXT, fontsize=10, loc='left')
    for ax in axes[1]:
        ax.set_xlabel('iteration', color=TEXT2)
    for ax in axes[:, 0]:
        ax.set_ylabel('gbest variance', color=TEXT2)
    leg = axes[0, 0].legend(frameon=False, fontsize=8.5, loc='upper right')
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.suptitle('Stage 5 · PSO seed 1 against the certified optimum at the final budget', color=TEXT, fontsize=11, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(OUT / 'stage5_pso_convergence.png', facecolor=SURF)
    DASH.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / 'stage5_pso_convergence.png', DASH / 'stage5_pso_convergence.png')
    if not (DASH / 's5_summary.json').exists():
        from stage5_pso_same_budget import publish
        publish(runs, load_json((OUT / 'pso_budgets.json').read_text()), load_json((OUT / 'pso_summary.json').read_text()))
    payload = load_json((DASH / 's5_summary.json').read_text())
    payload['images'] = [dict(src='data/stage5_pso_convergence.png', alt='PSO 수렴과 최종 예산의 인증 구간')]
    progress.result('s5_summary', 'S5 · 요약', payload)
if __name__ == '__main__':
    main()
