# Resumable kernel campaigns

Add `--checkpoint data/campaign.pt` to `python -m
drosophilos.bench.kernel_campaign ...`. It saves every 1,800 wall seconds;
`--checkpoint-every-s 300` changes that interval. SIGTERM and SIGUSR1 request a
save at the next simulation block boundary and exit **75**. Resume by running
the same command again. No flags means the existing execution and record format.

Checkpoints use a same-directory temporary file, file fsync, atomic rename and
directory fsync. The previous checkpoint survives an interrupted write. They
contain Python objects: load only trusted checkpoint files. Allow enough disk
space for both the old and new files, and enough time before the job limit to
write several GiB. SIGKILL cannot save; it loses work since the last save.
At startup, abandoned temporary files matching only this checkpoint's
`NAME.XXXXXXXX.tmp` pattern are removed. Use one writer per checkpoint.

Resume requires the same source content, topology, build options, seed, copies,
repeats, mix, dtype, backend, observation interval, device and neural-time limit.
Mismatches fail with `campaign checkpoint fingerprint mismatch`. Use the same
device model and software versions for exact results. The fingerprint hashes
source content, not the git commit: an unrelated documentation commit does not
prevent appending jobs. `--out` and the checkpoint interval may change.
The full simulator state (including its seeded stray-input
generator and delay ring), pending observation buffers, decoder, schedules,
retry/refusal state and NumPy generator are restored. On the same device and
dtype, campaign records match uninterrupted execution except `wall_s` and the
added checkpoint fingerprint metadata; runner
timing/profiling fields and injected wall timestamps are also wall-clock fields.
CPU fresh-process tests cover both backends, settling, mid-token state and a
queued watchdog retry. CUDA exactness has not been exercised by these CPU tests.

The final output is published atomically before replacing the checkpoint with a
small finished marker. Later jobs verify the fingerprint and exit zero without
simulating; they recreate a missing output from the marker. A matching complete
output also handles interruption between publishing the output and the marker:
checkpointed runs include `checkpoint_fingerprint` in the finished JSON record.
Without a matching fingerprint or finished marker, an existing `--out` is refused
with exit 2; move it aside or choose another output path. Matching run parameters
alone cannot establish that an old output came from the same code. Runs without
`--checkpoint` keep the original record format. Atomic outputs use the usual
umask permissions for new files and retain an existing file's permissions.
Use one chain per checkpoint; concurrent writers are unsupported. A run reaching
`--max-ms` is finished even if outputs are missing, as with existing campaigns.

## Slurm chains

`slurm/chain.sh N CHECKPOINT [submit.sh options] -- BLOCK [campaign options]`
submits N jobs with `afterany` dependencies. It adds `--signal=B:USR1@600`, giving
Python (the sbatch script execs it) about ten minutes to save before the limit.
It does not change `submit.sh` or the existing sbatch scripts. G2 preemption may
give much less notice; periodic checkpoints still apply. Submission needs a
clean, committed tree just like `submit.sh`. Optional `--dry-run` prints commands
with illustrative dependency IDs.
If saving takes longer than the Slurm warning and the job is killed mid-save,
the previous periodic checkpoint remains valid: only work since that checkpoint
is lost. Because dependencies use `afterany`, `scancel` of one job no longer
stops the chain. Cancel **all chain job IDs**, including queued jobs, to stop it.

For three 47-hour Juno jobs (a roughly 110-hour campaign):

```sh
slurm/chain.sh 3 data/perspective-1000.pt --cluster juno --time=1-23:00:00 -- \
  perspective --copies 100 --repeats 1000 --seed 115 --device cuda \
  --rate-robust --experimental-autapse --ordered-xor-oneshot \
  --f-width16-experimental --master-ignition --operand-xor-oneshot \
  --max-ms 100000000 --out data/perspective-1000.json
```

On G2, request a **specific GPU type via `--gres`**, as below, so every job uses
the same GPU model required by the fingerprint. An untyped `--gres=gpu:1` can
land on a different model and fail resume. Choose precision, wall limit and
number of jobs too:

```sh
slurm/chain.sh 12 data/perspective-g2.pt --cluster g2 --time=12:00:00 \
  --gres=gpu:nvidia_h100_nvl:1 -- \
  perspective --copies 100 --repeats 1000 --seed 115 --device cuda --fp32 \
  --rate-robust --experimental-autapse --ordered-xor-oneshot \
  --f-width16-experimental --master-ignition --operand-xor-oneshot \
  --max-ms 100000000 --out data/perspective-g2.json
```

Use `data/...` (the job worktree links it to `~/drosophilos/data`) or an absolute
path under the cluster's `~/drosophilos/data`. The chain helper rejects other
relative checkpoint paths. Job worktrees are disposable. Keep outputs under
`data/` too. More jobs can be appended using the same checkpoint and arguments;
already finished chains are cheap no-ops. Set `--max-ms` to cover the entire
chain's neural time, rather than a single job's wall time.

## Memory and checkpoint size

`make_perturbed_sim(..., n_steps)` does not allocate by `n_steps`: stray input is
drawn one device step at a time, without a time ceiling. The simulator delay ring
is always 101 slots. Normal observer history retains eight decode windows; the
Torch legacy trace is trimmed to a fixed number of recent chunks. There is no
allocation proportional to `max_ms` or elapsed simulation steps in a normal
campaign. Diagnostic spike capture and full traces deliberately retain history
and can grow without bound; avoid captures for long production campaigns.

Measured build shapes for the 16-bit perspective build with configuration F,
master ignition and operand XOR: **126,523 neurons, 260,114 edges**, 58 watched
neurons and a 94-step decode window. The checkpoint stores the delay ring as
flattened sparse COO: one int64 index and the unchanged int64 value per nonzero,
**16 bytes per occupied element**, restoring the identical dense tensor. There
is no floating conversion or approximation. The live dense ring remains
10,223,058,400 bytes at 100 copies; that allocation is not saved to disk.
TorchSim's already-ingested unfiltered legacy spike chunks are omitted, and the
saved observer's ingestion cursor is reset to zero for its empty restored lists.
The observer preserves the watched history, pending buffers and captures used
by decoding and the record; fresh-process resume tests cover this omission.

For **100 copies**, estimate the simulator tensor payload with
`100*N*(5*dtype_bytes + 4 + 1) + 100*E*8 + 16*R`, where `R` is the total count of
nonzero ring elements across all copies. A CPU measurement used one copy of the
build above, mix B, seed 17, float64, TorchSim, the eight perspective tokens and
`max_ms=1000`. Ring nonzero counts at steps 53 / 3107 / 6000 / 10000 were
**43 / 1020 / 1127 / 1133**. `torch.save(sim.snapshot(), BytesIO())` at step
10000 measured **7,801,730 bytes**, including sparse ring storage and metadata.
Extrapolating the largest sampled occupancy to 100 copies (`R=113300`) gives
**779,257,500 bytes (0.726 GiB) float64**, or **526,211,500 bytes (0.490 GiB)
float32**, before RNG state, queued events, observer/host histories and
serialization overhead. This is an activity-based estimate, not a measured
100-copy GPU checkpoint or an upper bound on later activity. Every additional
million occupied ring elements adds 16 MB. FastSim adds an eight-byte ring slot;
the observer buffer here is only `94*100*58` bytes.

`torch.save` stages CUDA tensors, including the sparse ring's indices and values,
in **host RAM**. Size Slurm **`--mem`** for that staging, the normal host state
and serialization overhead. Keep disk space for both checkpoint generations.
Snapshotting clones the other simulator tensors and creates sparse ring
indices/values, rather than cloning the dense ring; allow device workspace for
conversion. Restore densifies the ring on the device, temporarily requiring an
additional full dense ring alongside the live one (about 9.52 GiB extra at
100 copies). FastSim copies into the existing buffer to preserve graph addresses.
Use an H100/H200 at this size.

1,000 passes × 8 column tokens × 100 copies is 800,000 output pairs, plus at
least 800,000 load events, normalized schedule entries and load-step integers.
These lists scale with tokens and retries, rather than simulation steps: expect
several hundred MiB of Python host state (roughly 0.5–1 GiB including schedules,
events and scoring/JSON temporaries), small relative to the delay ring but not
constant space. Normalized schedules, host output/load/refusal lists and their
shared event references are saved.
Including that host estimate gives roughly **1.2–1.7 GiB float64** or
**1.0–1.5 GiB float32** per checkpoint for this workload, plus overhead and
activity-dependent growth.
The 8 column outputs are one output cell per token in this build. Checkpoint
size grows modestly with completed tokens; optional spike captures can dominate.
Output/refusal histories retain observed events, including spurious events;
pathological continuing activity can grow those lists beyond the token estimate.

## CPU hot-loop comparison

Measured on macOS arm64 with PyTorch 2.14.0 and one torch thread, without
`--checkpoint`. Both runners used the acceptance tests' two-bit MOV block,
one copy, token `[1]`, mix B, seed 17, float64 and `max_ms=2200`. The baseline
function was extracted by AST from `git show b176cb2:drosophilos/lib/kernel.py`
and executed with the same imports, simulator and helpers as the updated runner.
Pipeline/simulator construction was outside the timer; the timer enclosed each
`run_pipeline_batched` call. After one warm-up per version/backend, seven runs
per version alternated execution order; the table reports median steps/s.

| Backend | b176cb2 | Updated runner | Time slowdown | Steps per run |
| --- | ---: | ---: | ---: | ---: |
| torch | 4,837.65 | 4,812.95 | 0.51% | 15,426 |
| torch-fast (`observe_every=37`) | 8,326.31 | 8,295.02 | 0.38% | 15,432 |

Time slowdown is `baseline_steps_s / updated_steps_s - 1`. Both are below the
2% threshold, so no additional hot-loop refactor was needed.
