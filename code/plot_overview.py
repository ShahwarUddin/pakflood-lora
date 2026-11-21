"""EMSR838 overview figure from figs/mosaic_emsr838.npz (made on the server by make_figures.py) + results/pop_*.csv.
  python3 pakflood/plot_overview.py  -> figs/fig_emsr838_overview.png
Left: where the activation's AOIs are (official flood outline over WorldPop density). Right: zoom on the two most
telling AOIs, official outline vs hold-out vs few-shot vs full fine-tune, with the affected population per AOI.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
z = np.load(HERE / "figs" / "mosaic_emsr838.npz", allow_pickle=True)
W, S, E, N = z["bounds"]
res = float(z["res"])
mapped, people = z["mapped"], z["people"]
H, Wd = mapped.shape
layers = {"label": z["label"], "sa_holdout": z["sa_holdout"], "sa_fewshot": z["sa_fewshot"], "full_finetune": z["full_finetune"]}
NAMES = {"label": "CEMS official outline", "sa_holdout": "LoRA hold-out\n(never saw South Asia)", "sa_fewshot": "LoRA few-shot\n(+26 SA tiles)", "full_finetune": "Full fine-tune\n(trained on PK)"}
pops = {}
for k, f in [("label", "pop_sa_holdout"), ("sa_holdout", "pop_sa_holdout"), ("sa_fewshot", "pop_sa_fewshot"), ("full_finetune", "pop_full_finetune")]:
    d = pd.read_csv(HERE / "results" / f"{f}.csv").groupby("aoi")[["pop_pred", "pop_label"]].sum()
    pops[k] = d.pop_label if k == "label" else d.pop_pred
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

dens = np.log10(1 + people)
vmax = np.percentile(dens[dens > 0], 99.5)
# AOI bounding boxes from connected components of the mapped mask around each AOI centroid
comp, n = ndi.label(mapped > 0.05)
boxes = {}
for aoi, (x, y) in zip(z["aois"], z["aoi_xy"]):
    c, r = int((x - W) / res), int((N - y) / res)
    cid = comp[r, c]
    if cid == 0:  # centroid not on mapped pixel: nearest mapped pixel
        rr, cc = np.nonzero(mapped > 0.05)
        i = np.argmin((rr - r) ** 2 + (cc - c) ** 2)
        cid = comp[rr[i], cc[i]]
    rows, cols = np.nonzero(comp == cid)
    boxes[str(aoi)] = (W + cols.min() * res, N - (rows.max() + 1) * res, W + (cols.max() + 1) * res, N - rows.min() * res)


def draw(ax, key, ext=None, alpha_map=0.25):
    w, s, e, n = ext or (W, S, E, N)
    c0, c1 = int((w - W) / res), int(np.ceil((e - W) / res))
    r0, r1 = int((N - n) / res), int(np.ceil((N - s) / res))
    sl = (slice(r0, r1), slice(c0, c1))
    ex = (W + c0 * res, W + c1 * res, N - r1 * res, N - r0 * res)
    ax.imshow(dens[sl], cmap="Greys", vmin=0, vmax=vmax, extent=ex, interpolation="nearest")
    rgba = np.zeros((r1 - r0, c1 - c0, 4), np.float32)
    rgba[..., :3] = (0.12, 0.47, 0.71)
    rgba[..., 3] = alpha_map * (mapped[sl] > 0.05)
    ax.imshow(rgba, extent=ex, interpolation="nearest")
    rgba = np.zeros((r1 - r0, c1 - c0, 4), np.float32)
    rgba[..., :3] = (0.84, 0.15, 0.16)
    rgba[..., 3] = np.clip(layers[key][sl] * 1.4, 0, 1)
    ax.imshow(rgba, extent=ex, interpolation="nearest")
    ax.set_xlim(ex[0], ex[1])
    ax.set_ylim(ex[2], ex[3])


fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(2, 5, width_ratios=[1.35, 1, 1, 1, 1], wspace=0.12, hspace=0.3, left=0.04, right=0.99, top=0.86, bottom=0.1)
ax = fig.add_subplot(gs[:, 0])
draw(ax, "label", alpha_map=0.35)
offs = {"EMSR838-10": (-0.25, 0.05), "EMSR838-11": (0.2, 0.05), "EMSR838-8": (-0.3, 0.0), "EMSR838-1": (0.35, 0.05)}
for aoi, (x, y) in zip(z["aois"], z["aoi_xy"]):
    dx, dy = offs.get(str(aoi), (0.25, 0.0))
    ax.annotate(str(aoi).replace("EMSR838-", "AOI "), (x, y), (x + dx, y + dy), fontsize=7.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
for key in ["EMSR838-1", "EMSR838-7"]:
    w, s, e, n = boxes[key]
    ax.add_patch(Rectangle((w - 0.03, s - 0.03), e - w + 0.06, n - s + 0.06, fill=False, edgecolor=INK, lw=1.0, ls="--"))
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("EMSR838 activation: 9 mapped areas in Punjab\nofficial flood outline (red) over WorldPop density", fontsize=9)

keys = ["label", "sa_holdout", "sa_fewshot", "full_finetune"]
GROUPS = [("EMSR838-1", ["EMSR838-1", "EMSR838-8"], "AOI 1 + 8 (Chenab)"), ("EMSR838-7", ["EMSR838-3", "EMSR838-7"], "AOI 3 + 7 (Sutlej)")]
for row, (aoi, members, rowname) in enumerate(GROUPS):
    w, s, e, n = boxes[aoi]
    pad = 0.03
    for col, key in enumerate(keys):
        ax = fig.add_subplot(gs[row, col + 1])
        draw(ax, key, (w - pad, s - pad, e + pad, n + pad))
        ax.set_xticks([]); ax.set_yticks([])
        p = sum(pops[key][m] for m in members)
        off = sum(pops["label"][m] for m in members)
        t = f"{p:,.0f} people" if key == "label" else f"{p:,.0f} people ({100 * (p / off - 1):+.0f}%)"
        ax.set_title((NAMES[key] + "\n" if row == 0 else "") + t, fontsize=8.5)
        if col == 0:
            ax.set_ylabel(rowname, fontsize=9.5, fontweight="bold")
fig.legend(handles=[Patch(color="#d62728", label="flooded (fraction of ~200 m cell)"), Patch(color="#1f77b4", alpha=0.3, label="mapped area (AOI)"),
                    Patch(color="grey", label="WorldPop 2020 population density (log)")], loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.6, 0.0))
fig.suptitle("Pakistan floods, August 2025 (EMSR838, Sentinel-1 at 10 m): flood extent and affected population per model", fontsize=11, fontweight="bold")
fig.savefig(HERE / "figs" / "fig_emsr838_overview.png", dpi=170)
print("wrote figs/fig_emsr838_overview.png; boxes:", {k: tuple(round(v, 2) for v in b) for k, b in boxes.items()})
