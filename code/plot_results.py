"""Local result figures from results/results.json (no server needed).
  python3 pakflood/plot_results.py            -> writes pakflood/figs/fig_*.png
Colors: validated categorical palette, one fixed hue per model (never cycled).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results" / "results.json").read_text())
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)

COL = {"baseline_frozen": "#2a78d6", "lora_r16": "#eb6834", "lora_r32_all": "#6250d6", "full_finetune": "#1baf7a",
       "lora_r16_cedice": "#e87ba4", "sa_holdout": "#008300", "sa_fewshot": "#e34948", "official": "#52514e"}
LAB = {k: v["label"] for k, v in R["models"].items()}
LAB.update({"sa_holdout": "LoRA r=16 hold-out\n(never saw South Asia)", "sa_fewshot": "LoRA r=16 few-shot\n(+26 SA tiles)", "official": "CEMS official"})
SHORT = {"baseline_frozen": "Frozen", "lora_r16": "LoRA r16", "lora_r32_all": "LoRA r32\n(all linear)", "full_finetune": "Full FT",
         "lora_r16_cedice": "LoRA r16\nCE+Dice", "sa_holdout": "Hold-out\n(no S. Asia)", "sa_fewshot": "Few-shot\n(+26 tiles)", "official": "CEMS official"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def tidy(ax, ylabel=None):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def bars(ax, keys, vals, los=None, his=None, fmt="{:.3f}", width=0.62):
    x = np.arange(len(keys))
    for i, k in enumerate(keys):
        ax.bar(x[i], vals[i], width, color=COL[k], linewidth=0)
        if los is not None and los[i] is not None:
            ax.errorbar(x[i], vals[i], yerr=[[vals[i] - los[i]], [his[i] - vals[i]]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        ax.text(x[i], (his[i] if his is not None and his[i] is not None else vals[i]) + 0.012, fmt.format(vals[i]), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=8)
    return x


# ---------- Fig 1: validation curves ----------
def fig_curves():
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    order = ["baseline_frozen", "lora_r16", "lora_r32_all", "full_finetune", "lora_r16_cedice"]
    for k in order:
        y = R["val_curves"][k]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=COL[k], linewidth=2, marker="o", markersize=5, markerfacecolor="#fcfcfb", markeredgewidth=1.6, label=f"{LAB[k]}  (best {max(y):.3f})")
        b = int(np.argmax(y))
        ax.plot(x[b], y[b], marker="o", markersize=7, color=COL[k])
    ax.set_xlim(0.7, 10.3)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("epoch")
    tidy(ax, "validation flood IoU")
    ax.set_title("Validation flood IoU per epoch (filled marker = best epoch, used for testing)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_val_curves.png", dpi=200)
    plt.close(fig)


# ---------- Fig 2: H1 accuracy + cost ----------
def fig_h1():
    h = R["h1_official_test"]["full_tile"]
    keys = [k for k in ["baseline_frozen", "lora_r16", "lora_r32_all", "full_finetune", "lora_r16_cedice"] if h.get(k)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    vals = [h[k]["iou"] for k in keys]
    bars(ax, keys, vals, [h[k]["ci_tile"][0] for k in keys], [h[k]["ci_tile"][1] for k in keys])
    ax.set_ylim(0, 0.7)
    tidy(ax, "flood IoU (official test split)")
    ax.set_title(f"H1: accuracy on the official test split ({R['h1_official_test']['n_tiles']} tiles, {R['h1_official_test']['n_events']} events)")
    ax.text(0.01, 0.97, "error bars: 95% paired bootstrap CI over tiles", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax = axes[1]
    m = R["models"]
    vals = [m[k]["adapter_M"] if m[k]["adapter_M"] > 0 else 0.05 for k in keys]
    x = bars(ax, keys, vals, fmt="{:.1f} M")
    for i, k in enumerate(keys):
        if m[k]["adapter_M"] == 0:
            ax.texts[-len(keys) + i].set_text("0 (decoder only)")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 400)
    tidy(ax, "encoder parameters trained / stored (M, log)")
    ax.set_title("What has to be stored per task")
    ax = axes[2]
    vals = [m[k]["peak_mem_GiB"] for k in keys]
    x = bars(ax, keys, vals, fmt="{:.1f} GiB")
    for i, k in enumerate(keys):
        ax.text(x[i], 0.3, f"{m[k]['it_per_s']:.1f} it/s", ha="center", fontsize=7.5, color="#fcfcfb", fontweight="bold")
    ax.set_ylim(0, 10.5)
    tidy(ax, "peak GPU memory, batch 32 bf16 (GiB)")
    ax.set_title("Training cost (RTX PRO 6000)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_h1_accuracy_cost.png", dpi=200)
    plt.close(fig)


# ---------- Fig 3: H3 transfer ----------
def fig_h3():
    h3, pk = R["h3"], R["pk_official_test"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [0.7, 2.0, 1.4]})
    ax = axes[0]
    keys = ["sa_holdout", "sa_fewshot"]
    s = h3["sa_test_238"]
    bars(ax, keys, [s[k]["full_tile"] for k in keys], [s[k]["ci_tile"][0] for k in keys], [s[k]["ci_tile"][1] for k in keys])
    d = s["diff_fewshot_minus_holdout"]["tile"]
    ax.annotate(f"few-shot - hold-out: +{d[0]:.3f}\n95% CI [{d[1]:+.3f}, {d[2]:+.3f}]", xy=(0.5, 0.86), ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 0.97)
    tidy(ax, "flood IoU")
    ax.set_title("South-Asia test\n(238 tiles, 6 events)")
    ax = axes[1]
    keys = ["sa_holdout", "sa_fewshot", "baseline_frozen", "lora_r16", "lora_r32_all", "full_finetune", "lora_r16_cedice"]
    v = pk["full_tile"]
    bars(ax, keys, [v[k][0] for k in keys], [v[k][1] for k in keys], [v[k][2] for k in keys])
    ax.axvspan(-0.5, 1.5, color="#e6e5e1", alpha=0.5, lw=0, zorder=0)
    ax.text(0.5, 0.91, "never trained on Pakistan\n(or only 26 tiles)", ha="center", fontsize=7.5, color=INK2)
    ax.text(4.0, 0.91, "trained on 208 Pakistan tiles (EMSR629/631/838 train split)", ha="center", fontsize=7.5, color=INK2)
    ax.set_ylim(0, 1.0)
    tidy(ax)
    ax.set_title(f"Pakistan official test tiles (EMSR838, n={pk['n_tiles']})")
    ax = axes[2]
    rows = [(k, v) for k, v in h3["per_aoi"].items() if v[2] >= 3]
    rows.sort(key=lambda kv: kv[1][0])
    y = np.arange(len(rows))
    for i, (k, (a, b, n)) in enumerate(rows):
        ax.plot([a, b], [i, i], color=INK2, linewidth=1.2, zorder=1)
        ax.plot(a, i, "o", color=COL["sa_holdout"], markersize=6, zorder=2)
        ax.plot(b, i, "o", color=COL["sa_fewshot"], markersize=6, zorder=2)
        ax.text(max(a, b) + 0.02, i, f"n={n}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in rows], fontsize=7.5)
    ax.set_xlim(-0.02, 0.95)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("flood IoU per AOI (>= 3 test tiles)")
    ax.set_title("Per AOI: hold-out -> few-shot")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=COL["sa_holdout"], lw=0, label="hold-out"),
                       plt.Line2D([], [], marker="o", color=COL["sa_fewshot"], lw=0, label="few-shot")], loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("H3: transfer to South Asia (train on all non-South-Asian events, then adapt with 26 tiles)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_h3_transfer.png", dpi=200)
    plt.close(fig)


# ---------- Fig 4: H4 population ----------
def fig_h4():
    h4 = R["h4_emsr838"]
    off = h4["official_pop"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.4, 1.2]})
    ax = axes[0]
    keys = ["sa_holdout", "sa_fewshot", "baseline_frozen", "lora_r16", "lora_r32_all", "full_finetune", "lora_r16_cedice"]
    vals = [h4["models"][k]["pop"] / 1e3 for k in keys]
    x = bars(ax, keys, vals, fmt="{:,.0f} k")
    ax.axhline(off / 1e3, color=COL["official"], linewidth=1.5, linestyle="--")
    ax.text(0.01, 0.97, f"dashed line = CEMS official outline, {off / 1e3:,.0f} k people\ninside bars: total error, and population-weighted mean error per AOI", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    for i, k in enumerate(keys):
        m = h4["models"][k]
        ax.text(x[i], 30, f"{100 * (m['pop'] / off - 1):+.1f}%", ha="center", fontsize=8, color="#fcfcfb", fontweight="bold")
        ax.text(x[i], 110, f"AOI\n{m['aoi_mape_w']:.0f}%", ha="center", fontsize=7.5, color="#fcfcfb")
    ax.set_ylim(0, 1250)
    tidy(ax, "affected population (thousands)")
    ax.set_title(f"H4: people inside the flood extent, EMSR838 ({h4['n_tiles']} tiles)")
    ax = axes[1]
    aois = list(h4["per_aoi_holdout"].keys())
    aois.sort(key=lambda a: -h4["per_aoi_holdout"][a][1])
    xx = np.arange(len(aois))
    w = 0.27
    for j, (k, src) in enumerate([("official", None), ("sa_holdout", "per_aoi_holdout"), ("sa_fewshot", "per_aoi_fewshot")]):
        vals = [h4["per_aoi_holdout"][a][1] if src is None else h4[src][a][0] for a in aois]
        ax.bar(xx + (j - 1) * w, np.maximum(vals, 1), w - 0.03, color=COL[k], linewidth=0, label=LAB[k].split("\n")[0])
    ax.set_yscale("log")
    ax.set_ylim(50, 2e6)
    ax.set_xticks(xx)
    ax.set_xticklabels([a.replace("EMSR838-", "AOI ") for a in aois], fontsize=8)
    tidy(ax, "affected people per AOI (log)")
    ax.set_title("Per AOI: official vs models that never (or barely) saw Pakistan")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_h4_population.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_curves(); fig_h1(); fig_h3(); fig_h4()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.png")))
