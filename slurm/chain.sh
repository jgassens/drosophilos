#!/bin/bash
# Opt in: chain.sh N data/name.pt [submit.sh options] -- BLOCK [campaign options]
set -euo pipefail
cd "$(dirname "$0")/.."
usage() {
  echo "usage: $0 N data/checkpoint.pt [--cluster juno|g2] [sbatch options] -- BLOCK [kernel_campaign options]" >&2
  exit 2
}
[ "$#" -ge 4 ] || usage
count=$1; checkpoint=$2; shift 2
[[ "$count" =~ ^[1-9][0-9]*$ ]] || usage
# Relative paths outside data/ disappear when a cluster job's worktree is removed.
case "$checkpoint" in
  data/*|/*/drosophilos/data/*) ;;
  *) echo "chain.sh: checkpoint must be under data/ or an absolute .../drosophilos/data/ path" >&2; exit 2;;
esac
case "/$checkpoint/" in
  */../*) echo "chain.sh: checkpoint must not contain '..'" >&2; exit 2;;
esac
submit_options=(); dry_run=0
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do
  [ "$1" != --dry-run ] || dry_run=1
  submit_options+=("$1"); shift
done
[ "${1:-}" = -- ] || usage
shift
[ "$#" -gt 0 ] || usage
previous=""
for ((i=1; i<=count; i++)); do
  dependency=()
  [ -z "$previous" ] || dependency=("--dependency=afterany:$previous")
  output=$(slurm/submit.sh ${submit_options[@]+"${submit_options[@]}"} --signal=B:USR1@600 ${dependency[@]+"${dependency[@]}"} -- \
    drosophilos.bench.kernel_campaign "$@" --checkpoint "$checkpoint")
  printf '%s\n' "$output"
  if [ "$dry_run" -eq 1 ]; then
    previous=$((100000 + i)) # illustrative IDs; no jobs are submitted
  else
    previous=$(printf '%s\n' "$output" | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p')
    [[ "$previous" =~ ^[0-9]+$ ]] || { echo "chain.sh: cannot determine submitted job ID" >&2; exit 1; }
  fi
done
