"""Derive PakFlood experiment configs from the authors' early-fusion S1 config.

Usage (from ~/pakflood/code):  .venv/bin/python pakflood/make_configs.py
Writes pakflood/configs/*.yaml. Checkpoints of a run <name> land in workdir/pakflood/<name>/v1/checkpoints/
as best.ckpt, best_state_dict.ckpt and last.ckpt.
"""
import copy
from pathlib import Path

import yaml

BASE = Path("configs/fusion/early_s1_frozen.yaml")
OUT = Path("pakflood/configs")
EPOCHS = 10
WORKERS = 8
TASK = "pakflood.tasks.FloodCEDiceTask"  # behaves like the authors' task unless loss: cedice / init_from_checkpoint


def lora_cfg(r: int, targets: list[str]) -> dict:
    return {
        "method": "LORA",
        "replace_qkv": "qkv",
        "peft_config_kwargs": {"r": r, "lora_alpha": 2 * r, "lora_dropout": 0.05, "target_modules": targets, "bias": "none"},
    }


LORA_QV = ["q_linear", "v_linear"]
LORA_ALL = ["q_linear", "k_linear", "v_linear", "proj", "fc1", "fc2", "fc3"]


def common(cfg: dict, name: str, metadata: str = "data_tiles_s256_st128.csv", epochs: int = EPOCHS) -> dict:
    cfg = copy.deepcopy(cfg)
    tr = cfg["trainer"]
    tr["max_epochs"] = epochs
    tr["logger"]["init_args"]["save_dir"] = "workdir/pakflood"
    tr["logger"]["init_args"]["name"] = name
    tr["logger"]["init_args"]["version"] = "v1"
    tr["default_root_dir"] = f"workdir/pakflood/{name}/v1/"
    for cb in tr["callbacks"]:
        if cb["class_path"].endswith("StateDictAwareModelCheckpoint"):
            cb["init_args"]["filename"] = "best_state_dict" if "state_dict" in cb["init_args"]["filename"] else "best"
    cfg["data"]["init_args"]["num_workers"] = WORKERS
    cfg["data"]["init_args"]["metadata_filename"] = metadata
    cfg["model"]["class_path"] = TASK
    m = cfg["model"]["init_args"]
    m["scheduler_hparams"]["T_max"] = epochs
    return cfg


def finetune(cfg):
    m = cfg["model"]["init_args"]
    m["freeze_backbone"] = False
    m["lr"] = 5e-5
    m["lr_overrides"] = {"encoder": 5e-6}
    return cfg


def lora(cfg, r=16, targets=LORA_QV, lr=1e-4):
    m = cfg["model"]["init_args"]
    m["freeze_backbone"] = False  # PEFT freezes the base weights itself; LoRA + decoder train
    m["model_args"]["peft_config"] = lora_cfg(r, targets)
    m["lr"] = lr
    return cfg


def cedice(cfg, weights=(0.3, 1.0, 2.0)):
    m = cfg["model"]["init_args"]
    m["loss"] = "cedice"
    m["class_weights"] = list(weights)
    m["dice_weight"] = 1.0
    m["dice_classes"] = [2]
    return cfg


def init_from(cfg, run_name: str):
    cfg["model"]["init_args"]["init_from_checkpoint"] = f"workdir/pakflood/{run_name}/v1/checkpoints/best_state_dict.ckpt"
    return cfg


def mem_probe(cfg):
    """Short run that records peak GPU memory (DeviceStatsMonitor) and it/s for the cost table."""
    tr = cfg["trainer"]
    tr["max_epochs"] = 1
    tr["limit_train_batches"] = 300
    tr["limit_val_batches"] = 20
    tr["callbacks"].append({"class_path": "lightning.pytorch.callbacks.DeviceStatsMonitor", "init_args": {"cpu_stats": False}})
    return cfg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE.read_text())
    v = {}
    # H1 regimes (frozen / finetune / lora_r16 already trained overnight as v2 via CLI; these configs are used for test + memory)
    v["baseline_frozen"] = common(base, "baseline_frozen")
    v["full_finetune"] = finetune(common(base, "full_finetune"))
    v["lora_r16"] = lora(common(base, "lora_r16"))
    v["lora_r32_all"] = lora(common(base, "lora_r32_all"), r=32, targets=LORA_ALL)
    # H2 loss ablation
    v["lora_r16_cedice"] = cedice(lora(common(base, "lora_r16_cedice")))
    # H3 South Asia: hold-out, then few-shot pass from the hold-out checkpoint
    v["lora_r16_sa_holdout"] = lora(common(base, "lora_r16_sa_holdout", "data_tiles_sa_holdout.csv"))
    v["lora_r16_sa_fewshot"] = init_from(
        lora(common(base, "lora_r16_sa_fewshot", "data_tiles_sa_fewshot.csv", epochs=8), lr=5e-5), "lora_r16_sa_holdout")
    v["full_finetune_sa_holdout"] = finetune(common(base, "full_finetune_sa_holdout", "data_tiles_sa_holdout.csv"))
    # memory / speed probes
    for name in ["baseline_frozen", "full_finetune", "lora_r16", "lora_r32_all"]:
        v[f"mem_{name}"] = mem_probe(copy.deepcopy(v[name]))
        v[f"mem_{name}"]["trainer"]["logger"]["init_args"]["name"] = f"mem_{name}"
        v[f"mem_{name}"]["trainer"]["default_root_dir"] = f"workdir/pakflood/mem_{name}/v1/"
    for name, cfg in v.items():
        p = OUT / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print("wrote", p)


if __name__ == "__main__":
    main()
