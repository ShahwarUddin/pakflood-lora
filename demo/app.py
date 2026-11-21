"""PakFlood-LoRA demo: Sentinel-1 flood mapping with a LoRA-adapted TerraMind encoder (Gradio, CPU).

Run on Hugging Face Spaces or locally:   python app.py
Self-test (no UI):                        python app.py --selftest [tile id]
Files expected next to this script: assets/pakflood_lora_r32_state_dict_fp16.pt, samples/<tile>/{pre,post,label}.tif, samples.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
CKPT = HERE / "assets" / "pakflood_lora_r32_state_dict_fp16.pt"
SAMPLES = HERE / "samples.json"
MEANS = np.array([-12.599, -20.293, -12.599, -20.293], dtype=np.float32)
STDS = np.array([5.195, 5.890, 5.195, 5.890], dtype=np.float32)
CROP, STRIDE, NCLS, BATCH = 256, 128, 3, 8
PALETTE = np.array([[230, 230, 230], [31, 119, 180], [214, 39, 40], [255, 255, 255]], dtype=np.uint8)  # bg, water, flood, ignore
CLASS_NAMES = ["background (dry land)", "permanent water", "flood"]

# exactly the model_args of the trained run (pakflood/configs/lora_r32_all.yaml), without downloading pretrained weights
MODEL_ARGS = {
    "backbone": "terratorch_terramind_v1_base",
    "backbone_pretrained": False,
    "backbone_in_chans": 4,
    "backbone_modalities": ["sen1grd"],
    "backbone_bands": {"sen1grd": ["VV", "VH", "VV", "VH"]},
    "necks": [
        {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
        {"name": "ReshapeTokensToImage", "remove_cls_token": False},
        {"name": "LearnedInterpolateToPyramidal"},
    ],
    "decoder": "UNetDecoder",
    "decoder_channels": [512, 256, 128, 64],
    "num_classes": 3,
    "head_dropout": 0.3,
    "peft_config": {
        "method": "LORA",
        "replace_qkv": "qkv",
        "peft_config_kwargs": {"r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
                                "target_modules": ["q_linear", "k_linear", "v_linear", "proj", "fc1", "fc2", "fc3"], "bias": "none"},
    },
}

_MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    global _MODEL
    if _MODEL is None:
        from terratorch.models import EncoderDecoderFactory
        t0 = time.time()
        model = EncoderDecoderFactory().build_model(task="segmentation", **MODEL_ARGS)
        sd = torch.load(CKPT, map_location="cpu")
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] state dict: {len(missing)} missing, {len(unexpected)} unexpected keys", missing[:5], unexpected[:5])
        torch.set_num_threads(max(1, torch.get_num_threads()))
        _MODEL = model.to(DEVICE).eval()
        print(f"model ready in {time.time() - t0:.1f} s on {DEVICE}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
    return _MODEL


# ---------- data ----------
def read_s1_db(path):
    """2-band Sentinel-1 GRD GeoTIFF (VV, VH, linear intensity) -> (2, H, W) in dB, plus the raster profile."""
    with rasterio.open(path) as r:
        img = r.read().astype(np.float32)
        profile = r.profile
    if img.shape[0] < 2:
        raise ValueError("expected a 2-band GeoTIFF (VV, VH)")
    img = img[:2]
    np.nan_to_num(img, copy=False, nan=0.0)
    img[~np.isfinite(img) | (img <= 0) | (img > 1e3)] = 0.0
    img = np.maximum(img, np.finfo(np.float32).eps)
    return 10 * np.log10(img), profile


def to_input(pre_db, post_db):
    x = np.concatenate([pre_db, post_db]).astype(np.float32)
    return (x - MEANS[:, None, None]) / STDS[:, None, None]


def db_to_png(db):
    g = np.clip((db - (-25.0)) / 30.0, 0, 1)  # -25 .. +5 dB
    return Image.fromarray((g * 255).astype(np.uint8))


def cls_to_png(a):
    a = np.where(a == 255, 3, a).astype(np.uint8)
    return Image.fromarray(PALETTE[a])


# ---------- inference ----------
@torch.no_grad()
def predict(x):
    """x: (4, H, W) normalised. Sliding window 256 / stride 128, averaged softmax; returns class map (H, W) uint8."""
    model = get_model()
    _, H, W = x.shape
    ph = max(CROP, int(np.ceil(max(H - CROP, 0) / STRIDE)) * STRIDE + CROP)
    pw = max(CROP, int(np.ceil(max(W - CROP, 0) / STRIDE)) * STRIDE + CROP)
    xp = np.zeros((4, ph, pw), dtype=np.float32)
    xp[:, :H, :W] = x
    wins = [(y, xx) for y in range(0, ph - CROP + 1, STRIDE) for xx in range(0, pw - CROP + 1, STRIDE)]
    probs = torch.zeros((NCLS, ph, pw))
    count = torch.zeros((1, ph, pw))
    for i in range(0, len(wins), BATCH):
        chunk = wins[i:i + BATCH]
        batch = torch.from_numpy(np.stack([xp[:, y:y + CROP, xx:xx + CROP] for y, xx in chunk])).to(DEVICE)
        out = model(batch)
        logits = out.output if hasattr(out, "output") else out
        p = torch.softmax(logits.float(), dim=1).cpu()
        for j, (y, xx) in enumerate(chunk):
            probs[:, y:y + CROP, xx:xx + CROP] += p[j]
            count[:, y:y + CROP, xx:xx + CROP] += 1
    return (probs / count).argmax(0).to(torch.uint8).numpy()[:H, :W]


def metrics(pred, label):
    v = label != 255
    tp = int(((pred == 2) & (label == 2) & v).sum())
    fp = int(((pred == 2) & (label != 2) & v).sum())
    fn = int(((pred != 2) & (label == 2) & v).sum())
    iou = tp / max(tp + fp + fn, 1)
    return {"flood IoU": iou, "flood F1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def area_km2(mask, profile):
    res = profile.get("transform", None)
    px = abs(res.a * res.e) / 1e6 if res is not None else 1e-4
    return float(mask.sum() * px)


# ---------- samples ----------
def load_samples():
    if not SAMPLES.exists():
        return {}
    return json.loads(SAMPLES.read_text())


def run_pair(pre_path, post_path, label_path=None, progress=None):
    t0 = time.time()
    pre_db, prof = read_s1_db(pre_path)
    post_db, _ = read_s1_db(post_path)
    if pre_db.shape != post_db.shape:
        raise ValueError(f"pre and post tiles differ in size: {pre_db.shape} vs {post_db.shape}")
    x = to_input(pre_db, post_db)
    pred = predict(x)
    label = None
    if label_path and Path(label_path).exists():
        with rasterio.open(label_path) as r:
            label = r.read(1)
        pred_show = np.where(label == 255, 255, pred)
    else:
        pred_show = pred
    secs = time.time() - t0
    lines = [f"**Inference time:** {secs:.1f} s on {'GPU' if DEVICE.type == 'cuda' else 'CPU'} ({x.shape[1]} × {x.shape[2]} px, {len(range(0, max(x.shape[1] - CROP, 0) + 1, STRIDE)) ** 2 if x.shape[1] == x.shape[2] else '-'} windows)",
             f"**Predicted flood area:** {area_km2((pred_show == 2), prof):,.1f} km²  ·  permanent water {area_km2((pred_show == 1), prof):,.1f} km²"]
    if label is not None:
        m = metrics(pred, label)
        lines.append(f"**Against the CEMS analyst label (never used in training):** flood IoU **{m['flood IoU']:.3f}**, F1 {m['flood F1']:.3f}, precision {m['precision']:.3f}, recall {m['recall']:.3f}")
        lines.append(f"**Official flood area:** {area_km2((label == 2), prof):,.1f} km²")
    return pred_show, prof, db_to_png(pre_db[0]), db_to_png(post_db[0]), (cls_to_png(label) if label is not None else None), cls_to_png(pred_show), "\n\n".join(lines)


# ---------- UI ----------
def build_ui():
    import gradio as gr
    samples = load_samples()
    ids = list(samples)
    header = """# PakFlood-LoRA: flood mapping from Sentinel-1 radar
A frozen **TerraMind v1 base** encoder with a **rank-32 LoRA adapter** (5.7 M trained encoder weights) and a UNet decoder, trained on GEOID-Flood.
Input: the pre-event and post-event Sentinel-1 GRD images (VV, VH). Output: every 10 m pixel as dry land, permanent water or **flood**.
Official test flood IoU 0.525 (equal to full fine-tuning of all 101 M weights). Runs on CPU here, about 30 to 90 s per 1024 × 1024 tile.
"""
    legend = "Legend: grey = background, blue = permanent water, red = flood, white = outside the mapped area."
    with gr.Blocks(title="PakFlood-LoRA") as demo:
        gr.Markdown(header)
        with gr.Tab("Pakistan test tiles (EMSR838, August 2025)"):
            gr.Markdown("These are the 15 **official test tiles** of the Pakistan 2025 activation: the model never saw them in training. Pick one and run the model.")
            dd = gr.Dropdown(choices=[(f"{k}  (official flood {v.get('flood_km2_label', 0):.1f} km²)", k) for k, v in samples.items()], value=ids[0] if ids else None, label="Tile")
            with gr.Row():
                pre_i = gr.Image(label="pre-event S1 VV (dB)", type="pil", height=300)
                post_i = gr.Image(label="post-event S1 VV (dB)", type="pil", height=300)
                lab_i = gr.Image(label="CEMS official label", type="pil", height=300)
            run = gr.Button("Run PakFlood-LoRA on this tile", variant="primary")
            with gr.Row():
                pred_i = gr.Image(label="model prediction", type="pil", height=360)
                info = gr.Markdown()
            gr.Markdown(legend)

            def show(k):
                s = samples[k]
                pre_db, _ = read_s1_db(HERE / s["pre"]); post_db, _ = read_s1_db(HERE / s["post"])
                with rasterio.open(HERE / s["label"]) as r:
                    lab = r.read(1)
                return db_to_png(pre_db[0]), db_to_png(post_db[0]), cls_to_png(lab)

            def run_sample(k):
                s = samples[k]
                _, _, _, _, _, pred_png, text = run_pair(HERE / s["pre"], HERE / s["post"], HERE / s["label"])
                return pred_png, text

            if ids:
                demo.load(show, dd, [pre_i, post_i, lab_i])
            dd.change(show, dd, [pre_i, post_i, lab_i])
            run.click(run_sample, dd, [pred_i, info])

        with gr.Tab("Your own Sentinel-1 pair"):
            gr.Markdown("Upload a **pre-event** and a **post-event** Sentinel-1 GRD GeoTIFF of the same area (2 bands: VV, VH in linear intensity, 10 m, same size; 256 px or larger). The output map is also returned as a GeoTIFF.")
            with gr.Row():
                pre_f = gr.File(label="pre-event GeoTIFF", file_types=[".tif", ".tiff"])
                post_f = gr.File(label="post-event GeoTIFF", file_types=[".tif", ".tiff"])
            run2 = gr.Button("Run PakFlood-LoRA", variant="primary")
            with gr.Row():
                u_pre = gr.Image(label="pre-event VV (dB)", type="pil", height=280)
                u_post = gr.Image(label="post-event VV (dB)", type="pil", height=280)
                u_pred = gr.Image(label="model prediction", type="pil", height=280)
            u_info = gr.Markdown()
            u_file = gr.File(label="prediction GeoTIFF (0 background, 1 permanent water, 2 flood)")

            def run_upload(pre, post):
                if pre is None or post is None:
                    raise gr.Error("Please upload both the pre-event and the post-event GeoTIFF.")
                pre_p = pre if isinstance(pre, str) else pre.name
                post_p = post if isinstance(post, str) else post.name
                pred, prof, pre_png, post_png, _, pred_png, text = run_pair(pre_p, post_p)
                out = Path(pre_p).with_name("pakflood_prediction.tif")
                prof.update(count=1, dtype="uint8", nodata=255, compress="deflate")
                with rasterio.open(out, "w", **prof) as w:
                    w.write(pred.astype(np.uint8), 1)
                return pre_png, post_png, pred_png, text, str(out)

            run2.click(run_upload, [pre_f, post_f], [u_pre, u_post, u_pred, u_info, u_file])
            gr.Markdown(legend)

        gr.Markdown("""### About
MS AI deep learning course project, NED University of Engineering & Technology (Muhammad Shahwar Uddin, supervised by Dr. Sharaf Hussain).
Data: GEOID-Flood v1.1 (Fondazione LINKS, CC BY 4.0), built from Copernicus Emergency Management Service activations. Model: TerraMind v1 base (IBM and ESA) through TerraTorch, LoRA via Hugging Face PEFT.
Results on the official GEOID-Flood test split (2,417 tiles): frozen encoder 0.433, LoRA r16 0.493, **LoRA r32 0.525**, full fine-tune 0.525 flood IoU. On Pakistan's 15 official test tiles: LoRA r32 0.786.
The sample tiles are the official test tiles of activation EMSR838 (Punjab, August 2025) and were never used for training.""")
    return demo


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        samples = load_samples()
        k = sys.argv[sys.argv.index("--selftest") + 1] if len(sys.argv) > sys.argv.index("--selftest") + 1 else next(iter(samples))
        s = samples[k]
        print("self-test on", k, "| torch threads:", torch.get_num_threads(), "| cuda:", torch.cuda.is_available())
        _, _, _, _, _, pred_png, text = run_pair(HERE / s["pre"], HERE / s["post"], HERE / s["label"])
        pred_png.save(HERE / "selftest_prediction.png")
        print(text.replace("**", ""))
        print("expected flood IoU from the evaluation run:", s.get("iou_r32"))
    else:
        get_model()
        build_ui().launch()
