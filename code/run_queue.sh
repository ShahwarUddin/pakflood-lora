#!/usr/bin/env bash
# PakFlood overnight queue: baseline_frozen -> full_finetune -> lora_r16 (10 epochs each, GPU 1).
# Run ON THE SERVER:  bash ~/pakflood/code/pakflood/run_queue.sh
# Survives logout (nohup + setsid, not tmux). Logs: ~/pakflood/run_*.log, each ends with EXIT=<code>.
set -u
cd "$HOME/pakflood/code" || exit 1
export HF_HOME="$HOME/pakflood/.cache/hf" CUDA_VISIBLE_DEVICES=1

COMMON=(--data.init_args.num_workers 8 --trainer.max_epochs 10
        "--model.init_args.scheduler_hparams={T_max: 10, eta_min: 1.0e-6, last_epoch: -1}"
        --trainer.logger.init_args.save_dir workdir/pakflood --trainer.logger.init_args.version v2)

LORA_ARGS='{backbone: terratorch_terramind_v1_base, backbone_pretrained: true, backbone_in_chans: 4, backbone_modalities: [sen1grd], backbone_bands: {sen1grd: [VV, VH, VV, VH]}, necks: [{name: SelectIndices, indices: [2, 5, 8, 11]}, {name: ReshapeTokensToImage, remove_cls_token: false}, {name: LearnedInterpolateToPyramidal}], decoder: UNetDecoder, decoder_channels: [512, 256, 128, 64], num_classes: 3, head_dropout: 0.3, peft_config: {method: LORA, replace_qkv: qkv, peft_config_kwargs: {r: 16, lora_alpha: 32, lora_dropout: 0.05, target_modules: [q_linear, v_linear]}}}'

run() {  # run <name> <config> [extra args...]
  local name="$1" cfg="$2"; shift 2
  nice -n 10 .venv/bin/terratorch fit --config "$cfg" "${COMMON[@]}" \
    --trainer.logger.init_args.name "$name" --trainer.default_root_dir "workdir/pakflood/$name/v2" "$@" \
    > "$HOME/pakflood/run_$name.log" 2>&1
  echo "EXIT=$?" >> "$HOME/pakflood/run_$name.log"
}

queue() {
  run baseline_frozen configs/fusion/early_s1_frozen.yaml
  run full_finetune   configs/fusion/early_s1_finetune.yaml
  run lora_r16        configs/fusion/early_s1_finetune.yaml \
      --model.init_args.lr 1e-4 --model.init_args.lr_overrides null \
      "--model.init_args.model_args=$LORA_ARGS"
}

if [[ "${1:-}" == "--inner" ]]; then queue; exit 0; fi

nohup setsid bash "$0" --inner > "$HOME/pakflood/queue.log" 2>&1 < /dev/null &
sleep 2
echo "Queue started in background (pid $!). Check with:"
echo "  tail -c 300 ~/pakflood/run_baseline_frozen.log"
echo "  pgrep -af 'terratorch fit' | cut -c1-80"
