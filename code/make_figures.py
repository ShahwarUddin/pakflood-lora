"""Figures that need the rasters (server, CPU only, a few minutes). Queue ON THE SERVER:
  bash ~/pakflood/code/pakflood/run_job.sh cmd figs .venv/bin/python pakflood/make_figures.py --out /home/shahwar/pakflood/figs

Writes into --out:
  fig_tiles_emsr838.png     the 4 official PK test tiles with most flood: post VV | pre VV | CEMS label | hold-out | few-shot | full FT
  fig_emsr838_overview.png  EMSR838 (Pakistan 2022) mosaic at ~200 m, EPSG:4326: flood fraction per model over WorldPop density
  mosaic_emsr838.npz        mosaic arrays (label / per-model flood fraction, mapped mask, people per cell, bounds) for re-plotting
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

CLS_CMAP = ListedColormap(["#e6e6e6", "#1f77b4", "#d62728"])  # background, permanent water, flood
CLS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
MODELS = [("sa_holdout", "LoRA hold-out (never saw South Asia)"),
          ("sa_fewshot", "LoRA few-shot (+26 SA tiles)"),
          ("full_finetune", "Full fine-tune (trained on PK)")]


def read1(p):
    with rasterio.open(p) as r:
        return r.read(1), r.transform, r.crs, r.bounds


def show_cls(ax, a, title=""):
    ax.imshow(np.ma.masked_equal(a, 255), cmap=CLS_CMAP, norm=CLS_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def flood_iou(pred, label):
    v = label != 255
    tp = ((pred == 2) & (label == 2) & v).sum()
    fp = ((pred == 2) & (label != 2) & v).sum()
    fn = ((pred != 2) & (label == 2) & v).sum()
    return tp / max(tp + fp + fn, 1)


def fig_tiles(a, out):
    root, preds = Path(a.root), Path(a.preds)
    df = pd.read_csv(root / "data_tiles_s256_st128.csv", usecols=["tile_id", "event_id", "modality", "image_time", "label_id"])
    df = df[(df.modality == "s1grd") & (df.event_id == "EMSR838")].drop_duplicates(["label_id", "image_time", "tile_id"])
    pt = pd.read_csv(preds / "sa_holdout_emsr838" / "per_tile.csv")
    pt = pt[pt.split == "test"].copy()
    pt["lab_flood"] = pt.cm_20 + pt.cm_21 + pt.cm_22
    tiles = a.tiles or pt.sort_values("lab_flood", ascending=False).label_id.head(4).tolist()
    ncol = 3 + len(MODELS)
    fig, axes = plt.subplots(len(tiles), ncol, figsize=(2.5 * ncol, 2.6 * len(tiles)))
    axes = np.atleast_2d(axes)
    for i, lid in enumerate(tiles):
        g = df[df.label_id == lid]
        pre, post = g[g.image_time == "pre"].tile_id.iloc[0], g[g.image_time == "post"].tile_id.iloc[0]
        aoi = lid.replace("_label", "").rsplit("-", 1)[0]  # tiles live in data/geoid-flood/<AOI>/{s1grd,label}
        label = read1(root / aoi / "label" / f"{lid}.tif")[0]
        for j, (tid, name) in enumerate([(post, "post-event S1 VV (dB)"), (pre, "pre-event S1 VV (dB)")]):
            with rasterio.open(root / aoi / "s1grd" / f"{tid}.tif") as r:
                vv = r.read(1).astype(np.float32)
            vv = 10 * np.log10(np.clip(vv, 1e-4, None))
            axes[i, j].imshow(vv, cmap="gray", vmin=-25, vmax=5)
            axes[i, j].set_title(name if i == 0 else "", fontsize=8)
            axes[i, j].set_axis_off()
        show_cls(axes[i, 2], label, "CEMS official label" if i == 0 else "")
        for k, (m, name) in enumerate(MODELS):
            pred = read1(preds / f"{m}_emsr838" / f"{lid}_pred.tif")[0]
            show_cls(axes[i, 3 + k], pred, (name.replace(" (", "\n(") + "\n" if i == 0 else "") + f"flood IoU {flood_iou(pred, label):.2f}")
        axes[i, 0].text(-0.03, 0.5, lid.replace("_label", ""), transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.legend(handles=[Patch(color=CLS_CMAP(0), label="background"), Patch(color=CLS_CMAP(1), label="permanent water"),
                        Patch(color=CLS_CMAP(2), label="flood")], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    fig.savefig(out / "fig_tiles_emsr838.png", dpi=150)
    plt.close(fig)
    print("wrote", out / "fig_tiles_emsr838.png", "tiles:", tiles, flush=True)


def fig_overview(a, out):
    root, preds, res = Path(a.root), Path(a.preds), a.res
    lids = sorted(Path(p).name.replace("_pred.tif", "") for p in glob.glob(f"{preds}/sa_fewshot_emsr838/*_pred.tif"))
    bb = {}
    def aoi_of(lid):
        return lid.replace("_label", "").rsplit("-", 1)[0]
    for lid in lids:
        with rasterio.open(root / aoi_of(lid) / "label" / f"{lid}.tif") as r:
            bb[lid] = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
    arr = np.array(list(bb.values()))
    W, S, E, N = arr[:, 0].min() - res, arr[:, 1].min() - res, arr[:, 2].max() + res, arr[:, 3].max() + res
    width, height = int(np.ceil((E - W) / res)), int(np.ceil((N - S) / res))
    print(f"mosaic grid {width}x{height} at {res} deg, {len(lids)} tiles", flush=True)
    layers = {"label": np.zeros((height, width), np.float32), **{m: np.zeros((height, width), np.float32) for m, _ in MODELS}}
    mapped = np.zeros((height, width), np.float32)
    aoi_pts = {}
    for n, lid in enumerate(lids, 1):
        aoi = aoi_of(lid)
        label, tr, crs, _ = read1(root / aoi / "label" / f"{lid}.tif")
        w, s, e, nn = bb[lid]
        aoi_pts.setdefault(aoi, []).append(((w + e) / 2, (s + nn) / 2))
        c0, r0 = int((w - W) / res), int((N - nn) / res)
        c1, r1 = int(np.ceil((e - W) / res)), int(np.ceil((N - s) / res))
        dst_tr = from_origin(W + c0 * res, N - r0 * res, res, res)
        srcs = {"label": label}
        for m, _ in MODELS:
            srcs[m] = read1(preds / f"{m}_emsr838" / f"{lid}_pred.tif")[0]
        valid = (label != 255).astype(np.float32)
        for key, src in list(srcs.items()) + [("mapped", None)]:
            mask = valid if key == "mapped" else ((src == 2) & (label != 255)).astype(np.float32)
            tmp = np.zeros((r1 - r0, c1 - c0), np.float32)
            reproject(mask, tmp, src_transform=tr, src_crs=crs, dst_transform=dst_tr, dst_crs="EPSG:4326",
                      resampling=Resampling.average)
            target = mapped if key == "mapped" else layers[key]
            target[r0:r1, c0:c1] = np.maximum(target[r0:r1, c0:c1], tmp)
        if n % 50 == 0 or n == len(lids):
            print(f"  {n}/{len(lids)}", flush=True)
    # WorldPop people per grid cell (sum resampling), read only the window we need
    people = np.zeros((height, width), np.float32)
    with rasterio.open(a.worldpop) as pop:
        win = from_bounds(W, S, E, N, transform=pop.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, pop.width, pop.height))
        src = pop.read(1, window=win).astype(np.float32)
        src[(src == pop.nodata) | ~np.isfinite(src) | (src < 0)] = 0
        reproject(src, people, src_transform=pop.window_transform(win), src_crs=pop.crs,
                  dst_transform=from_origin(W, N, res, res), dst_crs="EPSG:4326", resampling=Resampling.sum)
    np.savez_compressed(out / "mosaic_emsr838.npz", mapped=mapped, people=people, bounds=np.array([W, S, E, N]), res=res,
                        aois=np.array(list(aoi_pts.keys())), aoi_xy=np.array([np.mean(v, axis=0) for v in aoi_pts.values()]),
                        **layers)
    pops = {}
    for m, _ in MODELS:
        d = pd.read_csv(preds / f"pop_{m}.csv")
        pops[m] = (d.pop_pred.sum(), d.pop_label.sum())
    panels = [("label", "CEMS official outline")] + MODELS
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.2), sharex=True, sharey=True)
    ext = (W, E, S, N)
    dens = np.log10(1 + people)
    for ax, (key, name) in zip(axes, panels):
        ax.imshow(dens, cmap="Greys", vmin=0, vmax=np.percentile(dens[dens > 0], 99.5) if (dens > 0).any() else 1,
                  extent=ext, interpolation="nearest")
        rgba = np.zeros((height, width, 4), np.float32)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.12, 0.47, 0.71  # mapped area tint (blue)
        rgba[..., 3] = 0.18 * (mapped > 0.05)
        ax.imshow(rgba, extent=ext, interpolation="nearest")
        rgba = np.zeros((height, width, 4), np.float32)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.84, 0.15, 0.16
        rgba[..., 3] = np.clip(layers[key] * 1.3, 0, 1)
        ax.imshow(rgba, extent=ext, interpolation="nearest")
        for aoi, pts in aoi_pts.items():
            x, y = np.mean(pts, axis=0)
            ax.annotate(aoi.replace("EMSR838-", "AOI "), (x, y), fontsize=6, ha="center", color="black", alpha=0.8)
        if key == "label":
            ax.set_title(f"{name}\naffected population {pops[MODELS[0][0]][1]:,.0f}", fontsize=10)
        else:
            p, lab = pops[key]
            ax.set_title(f"{name}\naffected population {p:,.0f} ({100 * (p / lab - 1):+.1f}%)", fontsize=10)
        ax.set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    fig.legend(handles=[Patch(color="#d62728", label="flooded (fraction of ~200 m cell)"),
                        Patch(color="#1f77b4", alpha=0.3, label="mapped area (AOI)"),
                        Patch(color="grey", label="WorldPop 2020 population density (log)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("EMSR838 - Pakistan monsoon floods 2022 (Sentinel-1, 10 m) - flood extent and affected population", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "fig_emsr838_overview.png", dpi=150)
    plt.close(fig)
    print("wrote", out / "fig_emsr838_overview.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(Path.home() / "pakflood/preds"))
    ap.add_argument("--root", default="data/geoid-flood")
    ap.add_argument("--worldpop", default=str(Path.home() / "pakflood/data/worldpop/pak_ppp_2020_constrained.tif"))
    ap.add_argument("--out", default=str(Path.home() / "pakflood/figs"))
    ap.add_argument("--res", type=float, default=0.002, help="mosaic cell size in degrees (~200 m)")
    ap.add_argument("--tiles", nargs="*", default=None, help="label ids for the tile panels (default: 4 test tiles with most flood)")
    ap.add_argument("--skip-overview", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_tiles(a, out)
    if not a.skip_overview:
        fig_overview(a, out)


if __name__ == "__main__":
    main()
