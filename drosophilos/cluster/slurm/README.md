# Slurm launchers (UTD Juno)

`campaign.sbatch <block> [args]` runs one perturbation campaign of
`drosophilos.bench.a2_campaigns` on one GPU (`h200` partition by default; `-p h100`/`a30`
on the command line to override). The environment is the `drosophilos` conda env
(python 3.12, torch cu124, numpy, scipy, pyyaml) built by `juno_setup.sh` in the session
notes; the repo is cloned at `~/drosophilos` and updated with `git pull` before a run.

    sbatch drosophilos/cluster/slurm/campaign.sbatch machine --n 200 --batch 100
    sbatch drosophilos/cluster/slurm/campaign.sbatch ram --n 4000 --batch 100 --tx-per-chunk 20 --period 9000

Results land in `data/a2/<block>_4bit_B.jsonl` and `_summary.json` on the cluster; copy
the summaries into `docs/a2/` to record them.
