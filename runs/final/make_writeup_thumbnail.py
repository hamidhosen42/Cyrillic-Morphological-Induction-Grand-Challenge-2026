"""Kaggle writeup card/thumbnail, exactly 560 x 280 px.
Kaggle uses the full image as the 2:1 cover and a centred 280 x 280 square as the thumbnail,
so every key element sits inside the central square (x = 140..420 px)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e6e5e0'
BLUE, ORANGE = '#2a78d6', '#eb6834'
public = [0.65492, 0.66360, 0.66653, 0.66783, 0.66871, 0.67826, 0.67904, 0.67963, 0.68003]
private = [0.65085, 0.66135, 0.66205, 0.66315, 0.66511, 0.67308, 0.67415, 0.67455, 0.67529]

plt.rcParams.update({'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.patch.set_facecolor(SURFACE)
C = 0.5  # horizontal centre; the thumbnail square spans 0.25..0.75 of the width

fig.text(C, 0.84, '14th of 66', fontsize=27, fontweight='bold', color=INK, ha='center', va='center')
fig.text(C, 0.665, 'Private LB 0.67529', fontsize=14, color=INK, ha='center', va='center')
fig.text(C, 0.555, 'Public LB 0.68003', fontsize=11, color=INK2, ha='center', va='center')

ax = fig.add_axes([0.29, 0.13, 0.42, 0.33]); ax.set_facecolor(SURFACE)
x = range(len(public))
ax.plot(x, public, color=BLUE, lw=2, marker='o', ms=3.8, mec=SURFACE, mew=0.8)
ax.plot(x, private, color=ORANGE, lw=2, marker='o', ms=3.8, mec=SURFACE, mew=0.8)
ax.set_ylim(0.648, 0.684); ax.set_xticks([]); ax.set_yticks([])
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID)
fig.text(C, 0.055, 'Team Hack2Publish', fontsize=9.5, color=MUTED, ha='center', va='center')

# side notes: visible on the 2:1 cover, cropped away in the square thumbnail
fig.text(0.125, 0.5, 'Char\ntransformers\n+ stress-\nparadigm\ninference', fontsize=10, color=INK2,
         ha='center', va='center', linespacing=1.35)
from matplotlib.lines import Line2D
for yy, col, lab in ((0.60, BLUE, 'public'), (0.48, ORANGE, 'private')):
    fig.add_artist(Line2D([0.795, 0.835], [yy, yy], color=col, lw=2.5, transform=fig.transFigure))
    fig.text(0.845, yy, lab, fontsize=10, color=INK2, ha='left', va='center')
fig.text(0.875, 0.34, 'score by\npipeline\nversion', fontsize=9, color=MUTED, ha='center', va='center', linespacing=1.3)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'writeup_media', '0_card_thumbnail_560x280.png')
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
