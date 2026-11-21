"""Architecture overview figure for the report (figs/fig_architecture.png)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figs" / "fig_architecture.png"
INK, MUTED, BG = "#13233a", "#4a5568", "#fcfcfb"
BLUE, RED, GREEN, GREY = "#2a78d6", "#c0392b", "#1f7a4d", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "figure.facecolor": BG})
fig, ax = plt.subplots(figsize=(15, 5.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 5.2); ax.set_axis_off()


def box(x, y, w, h, title, lines=(), fc="#ffffff", ec=INK, lw=1.2, title_c=INK, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=title_c)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.5 - 0.26 * i, t, ha="center", va="top", fontsize=fs, color=MUTED)


def arrow(x1, y1, x2, y2, c=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.2, color=c))


box(0.2, 1.5, 2.0, 2.1, "Input tile", ["Sentinel-1 GRD, 10 m", "pre-event: VV, VH", "post-event: VV, VH", "4 channels (early fusion)", "dB + normalisation"], fc="#f3f6fb")
arrow(2.2, 2.55, 2.65, 2.55)
box(2.65, 1.8, 1.45, 1.5, "Patch embed", ["16 x 16 patches", "VV/VH weights", "loaded twice"])
arrow(4.1, 2.55, 4.5, 2.55)
box(4.5, 0.6, 4.0, 3.3, "TerraMind v1 base encoder (frozen)", ["12 transformer blocks, 85 M parameters", "pretrained on millions of S1 and S2 images"], fc="#eef2f8", fs=7.8)
for i in range(4):
    bx = 4.75 + i * 0.9
    ax.add_patch(FancyBboxPatch((bx, 1.0), 0.78, 1.8, boxstyle="round,pad=0.01,rounding_size=0.05", fc="#ffffff", ec=MUTED, lw=0.9))
    ax.text(bx + 0.39, 2.7, "block" if i < 3 else "...", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.text(bx + 0.39, 2.44, "attention", ha="center", va="top", fontsize=6.8, color=MUTED)
    ax.text(bx + 0.39, 2.24, "q  k  v  proj", ha="center", va="top", fontsize=6.8, color=MUTED)
    ax.text(bx + 0.39, 1.92, "MLP", ha="center", va="top", fontsize=6.8, color=MUTED)
    ax.text(bx + 0.39, 1.72, "fc1  fc2  fc3", ha="center", va="top", fontsize=6.8, color=MUTED)
    ax.add_patch(FancyBboxPatch((bx + 0.1, 1.08), 0.58, 0.3, boxstyle="round,pad=0.01,rounding_size=0.04", fc="#f6e3e0", ec=RED, lw=0.9))
    ax.text(bx + 0.39, 1.23, "+ B\u00b7A", ha="center", va="center", fontsize=6.8, color=RED, fontweight="bold")
ax.text(6.5, 0.78, "LoRA: W + B\u00b7A with W frozen; rank 16 on q, v (0.6 M) or rank 32 on all linear layers (5.7 M)", ha="center", va="center", fontsize=7.6, color=RED)
arrow(8.5, 2.55, 8.95, 2.55)
box(8.95, 1.5, 2.0, 2.1, "UNet decoder", ["15.5 M parameters", "multi-scale features", "always trained", "3-class head"])
arrow(10.95, 2.55, 11.4, 2.55)
box(11.4, 1.5, 3.3, 2.1, "Flood map", ["per 10 m pixel: background,", "permanent water or flood", "sliding window 256 / 128,", "averaged softmax"], fc="#f6e3e0", ec=RED)
arrow(13.05, 1.5, 13.05, 1.2, c=GREEN)
box(11.4, 0.15, 3.3, 1.05, "WorldPop overlay (H4)", ["flooded fraction x people", "per 100 m cell"], fc="#e9f3ee", ec=GREEN, title_c=GREEN)
ax.text(0.2, 4.95, "Three adaptation regimes, same data, decoder and schedule:", fontsize=9, color=INK, fontweight="bold")
ax.text(0.2, 4.62, "frozen encoder (decoder only)   |   LoRA adapters on the encoder (this work)   |   full fine-tuning of all 101 M weights", fontsize=8.6, color=MUTED)
ax.text(0.2, 4.3, "Loss: cross-entropy over the three classes (H2 variant: class-weighted CE + Dice on the flood class).", fontsize=8, color=MUTED)
ax.text(0.2, 4.02, "AdamW, cosine schedule, 10 epochs, early stopping on validation flood IoU, best checkpoint tested.", fontsize=8, color=MUTED)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
