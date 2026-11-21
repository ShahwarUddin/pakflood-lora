#!/usr/bin/env bash
# H3 + H4 chain for the South-Asia hold-out / few-shot models. Queue it ON THE SERVER (runs under the GPU-1 lock):
#   bash ~/pakflood/code/pakflood/run_job.sh cmd h3_chain bash pakflood/h3_chain.sh
# Steps (each prints "#### <cmd>" and "#### rc=<code>" in ~/pakflood/jobs/h3_chain.log):
#   1. test hold-out model on SA test (264 tiles), on the few-shot test subset (238) and on the PK official test (15)
#   2. few-shot fit (26 SA tiles, init from hold-out) -> test best.ckpt and last.ckpt on the same 238 + PK test
#   3. full-tile predictions on the 238 SA test tiles (hold-out vs few-shot) for event-level bootstrap CIs
#   4. Pakistan EMSR838 maps + WorldPop population (H4) for both models
#   5. full-tile predictions of the 3 global models on the official test split (2417 tiles) for H1 event-level CIs
set -u
cd "$HOME/pakflood/code" || exit 1
export HF_HOME="$HOME/pakflood/.cache/hf" CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$HOME/pakflood/code"
TT=.venv/bin/terratorch; PY=.venv/bin/python
HO_CFG=pakflood/configs/lora_r16_sa_holdout.yaml
HO_CK=workdir/pakflood/lora_r16_sa_holdout/v1/checkpoints/best.ckpt
FS_CFG=pakflood/configs/lora_r16_sa_fewshot.yaml
FS_DIR=workdir/pakflood/lora_r16_sa_fewshot/v1/checkpoints
PREDS="$HOME/pakflood/preds"
WP="$HOME/pakflood/data/worldpop/pak_ppp_2020_constrained.tif"
SA_EVENTS="EMSR205 EMSR459 EMSR629 EMSR631 EMSR838 EMSR851"
run() { echo; echo "#### $* ($(date +%H:%M))"; "$@"; echo "#### rc=$?"; }

run $TT test --config $HO_CFG --ckpt_path $HO_CK
run $TT test --config $HO_CFG --ckpt_path $HO_CK --data.init_args.metadata_filename data_tiles_sa_fewshot.csv
run $TT test --config $HO_CFG --ckpt_path $HO_CK --data.init_args.metadata_filename data_tiles_pk_test.csv

run $TT fit --config $FS_CFG
for ck in best last; do
  run $TT test --config $FS_CFG --ckpt_path $FS_DIR/$ck.ckpt
  run $TT test --config $FS_CFG --ckpt_path $FS_DIR/$ck.ckpt --data.init_args.metadata_filename data_tiles_pk_test.csv
done

run $PY pakflood/predict_tiles.py $HO_CK $PREDS/sa_holdout_satest --events $SA_EVENTS --split test --metadata data_tiles_sa_fewshot.csv --task pakflood
run $PY pakflood/predict_tiles.py $FS_DIR/last.ckpt $PREDS/sa_fewshot_satest --events $SA_EVENTS --split test --metadata data_tiles_sa_fewshot.csv --task pakflood

run $PY pakflood/predict_tiles.py $HO_CK $PREDS/sa_holdout_emsr838 --events EMSR838 --split all --task pakflood
run $PY pakflood/predict_tiles.py $FS_DIR/last.ckpt $PREDS/sa_fewshot_emsr838 --events EMSR838 --split all --task pakflood
run $PY pakflood/population.py $PREDS/sa_holdout_emsr838 "$WP" --out $PREDS/pop_sa_holdout.csv
run $PY pakflood/population.py $PREDS/sa_fewshot_emsr838 "$WP" --out $PREDS/pop_sa_fewshot.csv
for m in baseline_frozen full_finetune lora_r16; do
  run $PY pakflood/predict_tiles.py workdir/pakflood/$m/v2/checkpoints/best-epoch=5.ckpt $PREDS/${m}_test --events all --split test --task geoid
done
echo "#### chain done $(date)"
