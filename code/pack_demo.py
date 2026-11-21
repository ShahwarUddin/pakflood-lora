"""Pack the Hugging Face Space folder on the server: fp16 state dict + the 15 Pakistani official test tiles + app files.

Run ON THE SERVER:  cd ~/pakflood/code && .venv/bin/python pakflood/pack_demo.py --out ~/pakflood/demo
Then copy ~/pakflood/demo to the laptop and push it to the Space.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

CKPT = "workdir/pakflood/lora_r32_all/v1/checkpoints/best_state_dict.ckpt"
PER_TILE = Path.home() / "pakflood/preds/lora_r32_all_emsr838/per_tile.csv"
SPACE_SRC = Path("pakflood/space")  # app.py, requirements.txt, README.md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "pakflood/demo"))
    ap.add_argument("--root", default="data/geoid-flood")
    a = ap.parse_args()
    out, root = Path(a.out), Path(a.root)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    # 1. state dict, 'model.' prefix stripped, float16
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[len("model."):] if k.startswith("model.") else k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd, out / "assets" / "pakflood_lora_r32_state_dict_fp16.pt")
    print("state dict:", len(sd), "tensors,", sum(v.numel() for v in sd.values()) / 1e6, "M params")

    # 2. the 15 official Pakistani test tiles (pre, post, label), deflate-compressed copies
    df = pd.read_csv(root / "data_tiles_pk_test.csv", usecols=["tile_id", "event_id", "modality", "image_time", "label_id", "split"])
    df = df[(df.modality == "s1grd") & (df.split == "test") & (df.event_id == "EMSR838")].drop_duplicates(["label_id", "image_time", "tile_id"])
    iou = {}
    if PER_TILE.exists():
        pt = pd.read_csv(PER_TILE)
        for _, r in pt.iterrows():
            tp, fp, fn = r.cm_22, r.cm_02 + r.cm_12, r.cm_20 + r.cm_21
            iou[r.label_id] = float(tp / max(tp + fp + fn, 1))
    samples = {}
    for lid, g in df.groupby("label_id"):
        pre, post = g[g.image_time == "pre"].tile_id.iloc[0], g[g.image_time == "post"].tile_id.iloc[0]
        aoi = lid.replace("_label", "").rsplit("-", 1)[0]
        short = lid.replace("_label", "")
        d = out / "samples" / short
        d.mkdir(exist_ok=True)
        for name, src in [("pre", root / aoi / "s1grd" / f"{pre}.tif"), ("post", root / aoi / "s1grd" / f"{post}.tif"), ("label", root / aoi / "label" / f"{lid}.tif")]:
            with rasterio.open(src) as r:
                data, prof = r.read(), r.profile
            if name != "label":
                data = data[:2].astype(np.float32)  # VV, VH only
                prof.update(count=2, dtype="float32", compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
            else:
                prof.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(d / f"{name}.tif", "w", **prof) as w:
                w.write(data)
        with rasterio.open(d / "label.tif") as r:
            lab = r.read(1)
            px = abs(r.res[0] * r.res[1]) / 1e6
        samples[short] = {"pre": f"samples/{short}/pre.tif", "post": f"samples/{short}/post.tif", "label": f"samples/{short}/label.tif",
                          "flood_km2_label": float((lab == 2).sum() * px), "iou_r32": iou.get(lid)}
        print("packed", short, f"{samples[short]['flood_km2_label']:.1f} km2 flood, r32 IoU", iou.get(lid))
    (out / "samples.json").write_text(json.dumps(dict(sorted(samples.items(), key=lambda kv: -kv[1]["flood_km2_label"])), indent=1))

    # 3. app files
    for f in ["app.py", "requirements.txt", "README.md", ".gitattributes"]:
        if (SPACE_SRC / f).exists():
            shutil.copy(SPACE_SRC / f, out / f)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"demo folder ready: {out} ({total:.0f} MB)")


if __name__ == "__main__":
    main()
