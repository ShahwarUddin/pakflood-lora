#!/usr/bin/env bash
# H1/H2 follow-up chain for lora_r32_all (rank search) and lora_r16_cedice (loss ablation). Queue ON THE SERVER:
#   bash ~/pakflood/code/pakflood/run_job.sh cmd h12_chain bash pakflood/h12_chain.sh
# For each model: terratorch test on the official test split and on the PK official test (15 tiles),
# full-tile predictions on the official test split (2417 tiles, for bootstrap CIs / precision-recall),
# EMSR838 maps (177 tiles) + WorldPop population. ~15 min GPU. Log: ~/pakflood/jobs/h12_chain.log
set -u
cd "$HOME/pakflood/code" || exit 1
export HF_HOME="$HOME/pakflood/.cache/hf" CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$HOME/pakflood/code"
TT=.venv/bin/terratorch; PY=.venv/bin/python
PREDS="$HOME/pakflood/preds"
WP="$HOME/pakflood/data/worldpop/pak_ppp_2020_constrained.tif"
run() { echo; echo "#### $* ($(date +%H:%M))"; "$@"; echo "#### rc=$?"; }

for m in lora_r32_all lora_r16_cedice; do
  CFG=pakflood/configs/$m.yaml
  CK=workdir/pakflood/$m/v1/checkpoints/best.ckpt
  run $TT test --config $CFG --ckpt_path $CK
  run $TT test --config $CFG --ckpt_path $CK --data.init_args.metadata_filename data_tiles_pk_test.csv
  run $PY pakflood/predict_tiles.py $CK $PREDS/${m}_test --events all --split test --task pakflood
  run $PY pakflood/predict_tiles.py $CK $PREDS/${m}_emsr838 --events EMSR838 --split all --task pakflood
  run $PY pakflood/population.py $PREDS/${m}_emsr838 "$WP" --out $PREDS/pop_$m.csv
done
echo "#### chain done $(date)"
