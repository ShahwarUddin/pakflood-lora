#!/usr/bin/env bash
# PakFlood job launcher. Run ON THE SERVER from anywhere:
#   bash ~/pakflood/code/pakflood/run_job.sh fit  <name> <config.yaml> [extra terratorch args...]
#   bash ~/pakflood/code/pakflood/run_job.sh test <name> <config.yaml> <ckpt> [extra terratorch args...]
#   bash ~/pakflood/code/pakflood/run_job.sh cmd  <name> <program> [args...]   (any command, e.g. .venv/bin/python pakflood/predict_tiles.py ...)
# Each job is detached (nohup + setsid, survives logout) and jobs serialize on GPU 1 through a lock,
# so several can be queued at once. Log: ~/pakflood/jobs/<name>.log (first line START, last line EXIT=<code>).
set -u
if [[ $# -lt 3 ]]; then sed -n 2,8p "$0"; exit 1; fi
mode=$1; name=$2; cfg=$3; shift 3
cd "$HOME/pakflood/code" || exit 1
mkdir -p "$HOME/pakflood/jobs"
export HF_HOME="$HOME/pakflood/.cache/hf" CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$HOME/pakflood/code"
log="$HOME/pakflood/jobs/$name.log"
case "$mode" in
  fit)  cmd=(.venv/bin/terratorch fit --config "$cfg" "$@") ;;
  test) ckpt=$1; shift; cmd=(.venv/bin/terratorch test --config "$cfg" --ckpt_path "$ckpt" "$@") ;;
  cmd)  cmd=("$cfg" "$@") ;;
  *) echo "mode must be fit, test or cmd"; exit 1 ;;
esac
printf '%q ' "${cmd[@]}" > "$HOME/pakflood/jobs/$name.cmd"; echo >> "$HOME/pakflood/jobs/$name.cmd"
nohup setsid flock "$HOME/pakflood/jobs/gpu1.lock" nice -n 10 \
  bash -c 'echo "START $(date)"; "$@"; echo "EXIT=$? $(date)"' _ "${cmd[@]}" > "$log" 2>&1 < /dev/null &
echo "queued $name -> $log"
