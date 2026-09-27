"""Cover image for the Kaggle writeup: public vs private LB score across the clean pipeline versions."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e6e5e0'
PUBLIC, PRIVATE = '#2a78d6', '#eb6834'   # categorical slots 1 and 2

steps = ['v1 ensemble +\nparadigm re-rank', 'joint stress\ndecision', '+ v3\nmodels', 'prior weight\nnew lemmas',
         'candidate\npooling', '3way stress\nrep.', 'clean context\n+ 3 local', 'post-fixes',
         'PoE completion\n(Final A)']
public = [0.65492, 0.66360, 0.66653, 0.66783, 0.66871, 0.67826, 0.67904, 0.67963, 0.68003]
private = [0.65085, 0.66135, 0.66205, 0.66315, 0.66511, 0.67308, 0.67415, 0.67455, 0.67529]
x = list(range(len(steps)))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15})
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

for ys, c, name in ((public, PUBLIC, 'Public LB'), (private, PRIVATE, 'Private LB')):
    ax.plot(x, ys, color=c, lw=2.5, marker='o', ms=10, mec=SURFACE, mew=2, label=name, zorder=3)
    ax.annotate(f'{name}  {ys[-1]:.5f}', (x[-1], ys[-1]), xytext=(14, 0), textcoords='offset points',
                va='center', fontsize=16, color=INK, fontweight='bold')

ax.annotate('3way stress representation\n+0.0080 private', (5, private[5]), xytext=(4.1, 0.6605),
            fontsize=14, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=1.2))

ax.set_xticks(x); ax.set_xticklabels(steps, fontsize=12.5, color=INK2)
ax.set_ylim(0.646, 0.686); ax.set_xlim(-0.4, len(steps) + 1.3)
ax.set_ylabel('Leaderboard score', color=INK2, fontsize=15)
ax.tick_params(axis='y', colors=MUTED, labelsize=13); ax.tick_params(axis='x', length=0)
ax.grid(axis='y', color=GRID, lw=1); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.legend(loc='upper left', frameon=False, fontsize=15, labelcolor=INK)

fig.text(0.055, 0.945, 'Every holdout-validated step raised the private score', fontsize=26, fontweight='bold', color=INK)
fig.text(0.055, 0.895, 'Team Hack2Publish · 14th of 66 on the private leaderboard · Cyrillic Morphological Induction Grand Challenge 2026',
         fontsize=15.5, color=INK2)
fig.text(0.055, 0.03, 'Clean pipeline versions in submission order. Copy-from-other-dialect baseline (0.219) not shown.', fontsize=12.5, color=MUTED)
fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.17)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'writeup_media', '1_score_progression.png')
fig.savefig(out, facecolor=SURFACE); print('wrote', out)
