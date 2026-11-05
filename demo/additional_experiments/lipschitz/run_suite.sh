#!/usr/bin/env bash
# Run the staged input suite of the Lipschitz experiment (see README.md).
# Usage: demo/additional_experiments/lipschitz/run_suite.sh [jobs]
# Each line below is one reproducible run; runs whose results.json already
# contains a "prune" section are skipped so an interrupted suite resumes.
set -euo pipefail
cd "$(dirname "$0")/../../.."
JOBS=${1:-3}
RUN="uv run --group lipschitz python demo/additional_experiments/lipschitz/run.py"
OUT=demo/additional_experiments/results/lipschitz
mkdir -p "$OUT/logs"

suite() {
  # analytic pilot
  for shape in sphere box torus shell; do
    for n in 12 24 40; do echo "$shape $n"; done
  done
  # repository pilot + scaling
  echo "bunny 24"; echo "bunny 30"; echo "bunny 40"
  echo "armadillo 30"; echo "fandisk 30"
  # extrapolation stress (separate domain, never pooled)
  echo "bunny 30 --expand 0.1"
  # conditioning variants (separate fields)
  echo "bunny 30 --reg 1e-6"; echo "bunny 30 --dedup-tol 5e-3"; echo "bunny 30 --reg 1e-4"
}

one() {
  local args="$1"
  local tag
  tag=$(echo "$args" | tr ' ' '_' | tr -d '-')
  echo "start $args"
  $RUN $args --resume > "$OUT/logs/$tag.log" 2>&1 && echo "done $args" || echo "FAILED $args"
}
export -f one
export RUN OUT
suite | xargs -P "$JOBS" -I{} bash -c 'one "{}"'
