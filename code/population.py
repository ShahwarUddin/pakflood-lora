"""Affected-population estimate: overlay predicted and official (label) flood pixels on WorldPop.

Usage (from ~/pakflood/code, CPU):
  .venv/bin/python pakflood/population.py <pred_dir> <worldpop.tif> [--root data/geoid-flood] [--out pop.csv]

For every <label_id>_pred.tif in pred_dir: reproject the binary flood masks (prediction and label, both
restricted to mapped pixels) onto the WorldPop grid with average resampling, so each ~100 m population
pixel gets its flooded fraction; affected people = sum(fraction x population). Prints per-AOI and total
affected population for prediction vs official outline, flooded area in km2, and the percentage difference.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds


def flooded_people(mask: np.ndarray, src_transform, src_crs, pop_ds) -> float:
    """mask: float32 (H,W) in tile CRS, 1 = flooded. Returns people in flooded fraction of WorldPop pixels."""
    b = transform_bounds(src_crs, pop_ds.crs, *rasterio.transform.array_bounds(mask.shape[0], mask.shape[1], src_transform))
    win = from_bounds(*b, transform=pop_ds.transform).round_offsets().round_lengths()
    win = win.intersection(rasterio.windows.Window(0, 0, pop_ds.width, pop_ds.height))
    if win.width <= 0 or win.height <= 0:
        return 0.0
    pop = pop_ds.read(1, window=win).astype(np.float64)
    pop[(pop == pop_ds.nodata) | ~np.isfinite(pop) | (pop < 0)] = 0.0
    frac = np.zeros(pop.shape, dtype=np.float32)
    reproject(mask, frac, src_transform=src_transform, src_crs=src_crs,
              dst_transform=pop_ds.window_transform(win), dst_crs=pop_ds.crs, resampling=Resampling.average)
    return float((frac * pop).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("worldpop")
    ap.add_argument("--root", default="data/geoid-flood")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pred_dir, root = Path(a.pred_dir), Path(a.root)
    rows = []
    with rasterio.open(a.worldpop) as pop:
        for p in sorted(pred_dir.glob("*_pred.tif")):
            lid = p.name.replace("_pred.tif", "")
            ev = lid.replace("_label", "").rsplit("-", 1)[0]
            with rasterio.open(p) as r:
                pred = r.read(1)
                tr, crs, res = r.transform, r.crs, r.res
            with rasterio.open(root / ev / "label" / f"{lid}.tif") as r:
                label = r.read(1)
            mapped = label != 255
            m_pred = ((pred == 2) & mapped).astype(np.float32)
            m_lab = ((label == 2) & mapped).astype(np.float32)
            px_km2 = abs(res[0] * res[1]) / 1e6
            rows.append({
                "label_id": lid, "aoi": ev,
                "area_pred_km2": float(m_pred.sum() * px_km2), "area_label_km2": float(m_lab.sum() * px_km2),
                "pop_pred": flooded_people(m_pred, tr, crs, pop), "pop_label": flooded_people(m_lab, tr, crs, pop),
                "pop_mapped": flooded_people(mapped.astype(np.float32), tr, crs, pop),
            })
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
    agg = df.groupby("aoi")[["area_pred_km2", "area_label_km2", "pop_pred", "pop_label", "pop_mapped"]].sum()
    agg.loc["TOTAL"] = agg.sum()
    agg["pop_diff_pct"] = 100 * (agg.pop_pred - agg.pop_label) / agg.pop_label.replace(0, np.nan)
    pd.set_option("display.width", 160)
    print(agg.round(1).to_string())


if __name__ == "__main__":
    main()
