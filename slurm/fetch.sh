#!/bin/bash
# Fetch a G2 job's log and any result files it wrote under data/ back to this laptop.
#   slurm/fetch.sh <jobid> [remote paths under ~/drosophilos, e.g. data/perf/g2-h200.json]
set -euo pipefail
cd "$(dirname "$0")/.."
job=$1; shift
mkdir -p runs
scp -q g2:~/drosophilos/runs/slurm-$job.out runs/ 2>/dev/null || echo "no log yet for $job"
for p in "$@"; do mkdir -p "$(dirname "$p")"; scp -q "g2:~/drosophilos/$p" "$p" && echo "fetched $p"; done
ssh g2 "sacct -j $job --format=JobID,State,Elapsed,MaxRSS,NodeList -P" 2>&1 | grep -v "post-quantum\|store now\|openssh.com"
