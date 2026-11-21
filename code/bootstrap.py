"""Bootstrap 95% confidence intervals for flood IoU from per-tile confusion matrices.

Usage:
  .venv/bin/python pakflood/bootstrap.py <name>=<per_tile.csv> [<name>=<per_tile.csv> ...] [--unit tile|event] [--split test] [--n 2000]

Each per_tile.csv comes from pakflood/predict_tiles.py (same tiles for every model). Resamples units (tiles or
events/AOIs) with replacement, recomputes flood IoU from the summed confusion matrices, and reports the 2.5-97.5
percentile interval per model and for every pairwise difference (paired bootstrap: same resample for all models).
"""
import argparse

import numpy as np
import pandas as pd


def iou_from(df: pd.DataFrame, cls: int = 2) -> float:
    cm = np.array([[df[f"cm_{i}{j}"].sum() for j in range(3)] for i in range(3)], dtype=np.float64)
    tp = cm[cls, cls]
    fp = cm[:, cls].sum() - tp
    fn = cm[cls, :].sum() - tp
    return tp / max(tp + fp + fn, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path/to/per_tile.csv")
    ap.add_argument("--unit", default="event", choices=["tile", "event"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tables = {}
    for m in a.models:
        name, path = m.split("=", 1)
        d = pd.read_csv(path)
        if a.split:
            d = d[d.split == a.split]
        d = d.sort_values("label_id").reset_index(drop=True)
        d["unit"] = d.label_id if a.unit == "tile" else d.event.str.extract(r"(EMSR\d+)")[0]
        tables[name] = d
    names = list(tables)
    ref = tables[names[0]]
    for n in names[1:]:
        assert list(tables[n].label_id) == list(ref.label_id), f"{n}: tile set differs from {names[0]}"
    units = sorted(ref.unit.unique())
    print(f"{len(ref)} tiles, {len(units)} {a.unit}s, {a.n} resamples")

    rng = np.random.default_rng(a.seed)
    groups = {n: {u: g for u, g in t.groupby("unit")} for n, t in tables.items()}
    cols = [f"cm_{i}{j}" for i in range(3) for j in range(3)]
    sums = {n: {u: g[cols].sum().values.astype(np.float64) for u, g in groups[n].items()} for n in names}

    def iou_vec(cmv: np.ndarray) -> float:
        cm = cmv.reshape(3, 3)
        tp = cm[2, 2]
        return tp / max(cm[:, 2].sum() + cm[2, :].sum() - tp, 1)

    boots = {n: np.empty(a.n) for n in names}
    for b in range(a.n):
        pick = rng.choice(units, size=len(units), replace=True)
        for n in names:
            boots[n][b] = iou_vec(sum(sums[n][u] for u in pick))
    print("\nflood IoU with 95% bootstrap CI")
    for n in names:
        lo, hi = np.percentile(boots[n], [2.5, 97.5])
        print(f"  {n:24s} {iou_from(tables[n]):.4f}  [{lo:.4f}, {hi:.4f}]")
    if len(names) > 1:
        print("\npairwise differences (row minus column), 95% CI; interval excluding 0 = significant")
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                d = boots[n1] - boots[n2]
                lo, hi = np.percentile(d, [2.5, 97.5])
                print(f"  {n1} - {n2}: {iou_from(tables[n1]) - iou_from(tables[n2]):+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
