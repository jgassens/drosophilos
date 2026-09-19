#!/bin/bash
# Fetch a cluster job's log and any result files it wrote under data/ back to this laptop.
#   slurm/fetch.sh [--cluster juno|g2] <jobid> [remote paths under ~/drosophilos, e.g. data/perf/x.json]
set -euo pipefail
cd "$(dirname "$0")/.."
cluster=juno
if [ "${1:-}" = "--cluster" ]; then cluster=$2; shift 2; fi
job=$1; shift
mkdir -p runs
scp -q $cluster:~/drosophilos/runs/slurm-$job.out runs/ 2>/dev/null || echo "no log yet for $job"
for p in "$@"; do mkdir -p "$(dirname "$p")"; scp -q "$cluster:~/drosophilos/$p" "$p" && echo "fetched $p"; done
ssh "$cluster" "sacct -j $job --format=JobID,State,Elapsed,MaxRSS,NodeList -P" 2>&1 | grep -v "post-quantum\|store now\|openssh.com"
