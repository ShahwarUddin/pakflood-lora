"""PakFlood task: the authors' FloodSegmentationTask plus
   - loss "cedice": class-weighted cross-entropy + Dice (on chosen classes)
   - init_from_checkpoint: load a weights-only checkpoint before training (few-shot pass from the hold-out model)
"""
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn

from geoid_flood.tasks.flood_segmentation import FloodSegmentationTask


class CEDiceLoss(nn.Module):
    """class-weighted CE + dice_weight * multiclass Dice (ignore_index aware)."""

    def __init__(self, class_weights, ignore_index: int, dice_weight: float = 1.0, dice_classes=None):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss("multiclass", classes=dice_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ce(y_hat, y) + self.dice_weight * self.dice(y_hat, y)


class FloodCEDiceTask(FloodSegmentationTask):
    """Extra init args (all optional):
    dice_weight (float), dice_classes (list[int], e.g. [2] = flood only), init_from_checkpoint (path to *_state_dict.ckpt).
    With loss != "cedice" it behaves exactly like FloodSegmentationTask.
    """

    def __init__(
        self,
        *args: Any,
        dice_weight: float = 1.0,
        dice_classes: list[int] | None = None,
        init_from_checkpoint: str | None = None,
        **kwargs: Any,
    ):
        self.dice_weight = dice_weight
        self.dice_classes = dice_classes
        super().__init__(*args, **kwargs)
        if init_from_checkpoint:
            ck = torch.load(init_from_checkpoint, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
            missing, unexpected = self.load_state_dict(sd, strict=False)
            n_loaded = len(sd) - len(unexpected)
            print(f"[pakflood] init_from_checkpoint {init_from_checkpoint}: loaded {n_loaded} tensors, "
                  f"missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
            if missing:
                print("[pakflood] first missing keys:", missing[:5], flush=True)
            if n_loaded < 0.9 * len(self.state_dict()):
                raise RuntimeError("init_from_checkpoint loaded less than 90% of the model; architecture mismatch?")

    def configure_losses(self) -> None:
        if self.hparams["loss"] != "cedice":
            return super().configure_losses()
        self.criterion = CEDiceLoss(
            self.hparams["class_weights"],
            self.hparams["ignore_index"],
            dice_weight=self.dice_weight,
            dice_classes=self.dice_classes,
        )
