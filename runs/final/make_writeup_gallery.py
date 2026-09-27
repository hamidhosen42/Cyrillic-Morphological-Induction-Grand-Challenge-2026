"""Extra media-gallery images for the Kaggle writeup (pipeline diagram, error budget, metric weight)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e6e5e0'
BLUE, ORANGE = '#2a78d6', '#eb6834'
BLUE_T, ORANGE_T = '#cde2fb', '#fbdccf'
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15})


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.055, 0.935, title, fontsize=26, fontweight='bold', color=INK)
    fig.text(0.055, 0.885, subtitle, fontsize=15.5, color=INK2)
    return fig


def save(fig, name):
    path = os.path.join(OUT, 'writeup_media', name)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print('wrote', path)


# 1. Pipeline diagram ---------------------------------------------------------------------------
fig = canvas('Generate candidates, decide the stress, then pick the spelling',
             'Team Hack2Publish · 14th of 66 (private 0.67529) · pipeline of the final clean submission')
ax = fig.add_axes([0.03, 0.08, 0.94, 0.76]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')
stages = [
    ('1  Generate', ['5 char transformers', '(v1 ×3, v3 ×2)', 'input: lemma + tags +', 'known forms of the lemma', 'gold in beam 97–98%'], BLUE_T),
    ('2  Stress prior', ['k-medoids paradigm model', 'cur5 (K=600 ×6)', '3way (K=300 ×3)', 'observed cells →', 'P(class | forms)'], ORANGE_T),
    ('3  Decide class', ['Σ log NN class mass', '+ λ · log prior', 'joint over dialects', 'λ: 2 / 5 / 5 / 2', 'PoE for completion'], BLUE_T),
    ('4  Pick spelling', ['within chosen class:', 'log mass + 0.6 ·', 'log P(length delta)', 'pool 3 local class-', 'conditioned models'], ORANGE_T),
    ('5  Post-fix', ['consonant skeleton', 'ADJ long = short', '+ ending', 'geminates', '0 rows broken'], BLUE_T),
]
w, gap, y0, h = 16.8, 3.0, 8, 44
for i, (head, lines, fill) in enumerate(stages):
    x0 = 1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fill, ec='none'))
    ax.text(x0 + w / 2, y0 + h - 5, head, ha='center', va='center', fontsize=18, fontweight='bold', color=INK)
    for j, line in enumerate(lines):
        ax.text(x0 + w / 2, y0 + h - 13 - j * 5.6, line, ha='center', va='center', fontsize=12.8, color=INK2)
    if i < len(stages) - 1:
        ax.annotate('', xy=(x0 + w + gap - 0.4, y0 + h / 2), xytext=(x0 + w + 0.4, y0 + h / 2),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=2, mutation_scale=22))
ax.text(1, 2, 'Noise-cleaned context: rows whose form is unrelated to their lemma are dropped from the paradigm evidence.',
        fontsize=13, color=MUTED)
save(fig, '2_pipeline.png')

# 2. Error budget: exact match now vs. if the stress class were known ------------------------------
fig = canvas('The remaining error is the stress of new lemmas',
             'Holdout exact match of the final pipeline, and exact match on rows where the stress class is right')
ax = fig.add_axes([0.07, 0.15, 0.9, 0.66]); ax.set_facecolor(SURFACE)
segs = ['Transfer', 'Completion', 'Wug', 'Unseen']
now = [0.960, 0.955, 0.496, 0.403]
known = [0.988, 0.990, 0.977, 0.964]
cls = [0.97, 0.97, 0.50, 0.42]
bw = 0.36
for k, s in enumerate(segs):
    ax.bar(k - bw / 2 - 0.01, now[k], bw, color=BLUE, zorder=3, label='Exact match (final pipeline)' if k == 0 else None)
    ax.bar(k + bw / 2 + 0.01, known[k], bw, color=ORANGE, zorder=3, label='Exact match when the stress class is right' if k == 0 else None)
    ax.text(k - bw / 2 - 0.01, now[k] + 0.015, f'{now[k]:.3f}', ha='center', fontsize=14, color=INK)
    ax.text(k + bw / 2 + 0.01, known[k] + 0.015, f'{known[k]:.3f}', ha='center', fontsize=14, color=INK)
ax.set_xticks(range(len(segs)))
ax.set_xticklabels([f'{s}\nstress-class accuracy {c:.2f}' for s, c in zip(segs, cls)], fontsize=14, color=INK2)
ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis='y', colors=MUTED, labelsize=13); ax.tick_params(axis='x', length=0)
ax.grid(axis='y', color=GRID, lw=1); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=14.5, labelcolor=INK)
fig.text(0.055, 0.03, 'From one wug seed, stress-class accuracy stops near 0.47–0.49 however it is modelled; unseen adjective stress is close to uniform.',
         fontsize=12.5, color=MUTED)
save(fig, '3_error_budget.png')

# 3. Metric weight: share of test rows vs share of the score --------------------------------------
fig = canvas('Wug and unseen rows carry about 65% of the score',
             'Segment weights 1 / 2 / 3 / 4, decoded from probe submissions, applied to the test row counts')
ax = fig.add_axes([0.16, 0.12, 0.8, 0.68]); ax.set_facecolor(SURFACE)
rows = {'Transfer': 18000, 'Completion': 15000, 'Wug': 18260, 'Unseen': 8740}
wts = {'Transfer': 1, 'Completion': 2, 'Wug': 3, 'Unseen': 4}
tot_r = sum(rows.values()); tot_w = sum(rows[s] * wts[s] for s in rows)
names = list(rows)[::-1]
share_r = [rows[s] / tot_r for s in names]
share_w = [rows[s] * wts[s] / tot_w for s in names]
bh = 0.36
for k, s in enumerate(names):
    ax.barh(k + bh / 2 + 0.01, share_r[k], bh, color=BLUE, zorder=3, label='Share of test rows' if k == 0 else None)
    ax.barh(k - bh / 2 - 0.01, share_w[k], bh, color=ORANGE, zorder=3, label='Share of the score' if k == 0 else None)
    ax.text(share_r[k] + 0.006, k + bh / 2 + 0.01, f'{share_r[k]:.0%}', va='center', fontsize=14, color=INK)
    ax.text(share_w[k] + 0.006, k - bh / 2 - 0.01, f'{share_w[k]:.0%}', va='center', fontsize=14, color=INK)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([f'{s}\n{rows[s]:,} rows · weight {wts[s]}' for s in names], fontsize=14, color=INK2)
ax.set_xlim(0, 0.46); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.tick_params(axis='x', colors=MUTED, labelsize=13); ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color=GRID, lw=1); ax.set_axisbelow(True)
for s in ('top', 'right', 'bottom'): ax.spines[s].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.legend(loc='lower right', frameon=False, fontsize=14.5, labelcolor=INK)
fig.text(0.055, 0.03, 'Score = 0.9 × WeightedEM + 0.1 × (1 − WeightedCER).', fontsize=12.5, color=MUTED)
save(fig, '4_metric_weight.png')
