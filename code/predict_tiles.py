"""Predict full 1024x1024 GEOID-Flood tiles with a trained checkpoint (sliding window, averaged softmax).

Usage (from ~/pakflood/code, GPU):
  PYTHONPATH=. .venv/bin/python pakflood/predict_tiles.py <ckpt> <out_dir> [--events EMSR838 ...] [--split test|train|all]
          [--metadata data_tiles_s256_st128.csv] [--task geoid|pakflood]

Writes <out_dir>/<label_id>_pred.tif (uint8: 0 bg, 1 permanent water, 2 flood; same CRS/transform as the label)
and <out_dir>/per_tile.csv with the per-tile confusion matrix (3x3, ignore 255) for IoU and bootstrap CIs.
Preprocessing mirrors geoid_flood.datasets.geoid.GEOIDFloodDataset (dB conversion, nodata, normalisation).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

MEANS = np.array([-12.599, -20.293, -12.599, -20.293], dtype=np.float32)
STDS = np.array([5.195, 5.890, 5.195, 5.890], dtype=np.float32)
CROP, STRIDE, NCLS = 256, 128, 3


def read_s1_db(path: Path) -> np.ndarray:
    with rasterio.open(path) as r:
        img = r.read().astype(np.float32)
    np.nan_to_num(img, copy=False, nan=0.0)
    img[~np.isfinite(img) | (img <= 0) | (img > 1e3)] = 0.0
    img = np.maximum(img, np.finfo(np.float32).eps)
    return 10 * np.log10(img)


def load_task(ckpt: str, task: str):
    if task == "pakflood":
        from pakflood.tasks import FloodCEDiceTask as T
    else:
        from geoid_flood.tasks.flood_segmentation import FloodSegmentationTask as T
    m = T.load_from_checkpoint(ckpt, map_location="cpu")
    return m.eval()


@torch.no_grad()
def predict_tile(model, x: np.ndarray, device) -> np.ndarray:
    """x: (4,H,W) normalised. Returns argmax class map (H,W) from averaged softmax over 256 windows, stride 128."""
    _, H, W = x.shape
    ys = list(range(0, H - CROP + 1, STRIDE))
    xs = list(range(0, W - CROP + 1, STRIDE))
    wins = [(y, xx) for y in ys for xx in xs]
    batch = torch.from_numpy(np.stack([x[:, y : y + CROP, xx : xx + CROP] for y, xx in wins])).to(device)
    probs = torch.zeros((NCLS, H, W), device=device)
    count = torch.zeros((1, H, W), device=device)
    for i in range(0, len(wins), 32):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.model(batch[i : i + 32])
        logits = out.output if hasattr(out, "output") else out
        p = torch.softmax(logits.float(), dim=1)
        for j, (y, xx) in enumerate(wins[i : i + 32]):
            probs[:, y : y + CROP, xx : xx + CROP] += p[j]
            count[:, y : y + CROP, xx : xx + CROP] += 1
    return (probs / count).argmax(0).to(torch.uint8).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--events", nargs="+", default=["EMSR838"], help="event ids, or 'all'")
    ap.add_argument("--split", default="all", help="train|val|test|all (split of the given metadata CSV)")
    ap.add_argument("--metadata", default="data_tiles_s256_st128.csv")
    ap.add_argument("--root", default="data/geoid-flood")
    ap.add_argument("--task", default="geoid", choices=["geoid", "pakflood"])
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / a.metadata, usecols=["tile_id", "event_id", "modality", "image_time", "label_id", "split"])
    df = df[df.modality == "s1grd"]
    if a.events != ["all"]:
        df = df[df.event_id.isin(a.events)]
    if a.split != "all":
        df = df[df.split == a.split]
    df = df.drop_duplicates(["label_id", "image_time", "tile_id"]).sort_values(["label_id", "image_time", "tile_id"])
    pairs = {}
    for lid, g in df.groupby("label_id"):
        pre, post = g[g.image_time == "pre"].tile_id.tolist(), g[g.image_time == "post"].tile_id.tolist()
        if pre and post:
            pairs[lid] = (pre[0], post[0], g.split.iloc[0], len(pre), len(post))
    print(f"{len(pairs)} tiles to predict for {a.events} (split={a.split})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_task(a.ckpt, a.task).to(device)

    rows = []
    for k, (lid, (pre, post, split, npre, npost)) in enumerate(pairs.items(), 1):
        ev = lid.replace("_label", "").rsplit("-", 1)[0]
        x = np.concatenate([read_s1_db(root / ev / "s1grd" / f"{pre}.tif"), read_s1_db(root / ev / "s1grd" / f"{post}.tif")])
        x = (x - MEANS[:, None, None]) / STDS[:, None, None]
        pred = predict_tile(model, x, device)
        with rasterio.open(root / ev / "label" / f"{lid}.tif") as r:
            label = r.read(1)
            prof = r.profile
        prof.update(dtype="uint8", count=1, nodata=255, compress="deflate")
        pred_out = np.where(label == 255, 255, pred).astype(np.uint8)
        with rasterio.open(out / f"{lid}_pred.tif", "w", **prof) as w:
            w.write(pred_out, 1)
        valid = label != 255
        cm = np.bincount(label[valid].astype(np.int64) * NCLS + pred[valid].astype(np.int64), minlength=NCLS * NCLS)
        rows.append({"label_id": lid, "event": ev, "split": split, "n_pre": npre, "n_post": npost,
                     **{f"cm_{i}{j}": int(cm[i * NCLS + j]) for i in range(NCLS) for j in range(NCLS)}})
        if k % 20 == 0 or k == len(pairs):
            print(f"  {k}/{len(pairs)}", flush=True)

    with open(out / "per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cm = np.zeros((NCLS, NCLS), dtype=np.int64)
    for r in rows:
        for i in range(NCLS):
            for j in range(NCLS):
                cm[i, j] += r[f"cm_{i}{j}"]
    for c, name in enumerate(["background", "permanent_water", "flood"]):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        print(f"{name:16s} IoU={tp / max(tp + fp + fn, 1):.4f}  F1={2 * tp / max(2 * tp + fp + fn, 1):.4f}")
    print("wrote", out / "per_tile.csv")


if __name__ == "__main__":
    main()
