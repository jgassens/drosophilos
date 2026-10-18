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

Resume requires the same commit, source, topology, build options, seed, copies,
repeats, mix, dtype, backend, observation interval, device and neural-time limit.
Mismatches fail with `campaign checkpoint fingerprint mismatch`. Use the same
device model and software versions for exact results. `--out` and the checkpoint
interval may change. The full simulator state (including its seeded stray-input
generator and delay ring), pending observation buffers, decoder, schedules,
retry/refusal state and NumPy generator are restored. On the same device and
dtype, campaign records match uninterrupted execution except `wall_s`; runner
timing/profiling fields and injected wall timestamps are also wall-clock fields.
CPU fresh-process tests cover both backends, settling, mid-token state and a
queued watchdog retry. CUDA exactness has not been exercised by these CPU tests.

The final output is published atomically before replacing the checkpoint with a
small finished marker. Later jobs verify the fingerprint and exit zero without
simulating; they recreate a missing output from the marker. A matching complete
output also handles interruption between publishing the output and the marker.
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

For three 47-hour Juno jobs (a roughly 110-hour campaign):

```sh
slurm/chain.sh 3 data/perspective-1000.pt --cluster juno --time=1-23:00:00 -- \
  perspective --copies 100 --repeats 1000 --seed 115 --device cuda \
  --rate-robust --experimental-autapse --ordered-xor-oneshot \
  --f-width16-experimental --master-ignition --operand-xor-oneshot \
  --max-ms 100000000 --out data/perspective-1000.json
```

On G2, choose the GPU, precision, wall limit and number of jobs:

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
neurons and a 94-step decode window. At 100 copies, the simulator snapshot's
tensor payload is approximately **11,000,503,100 bytes (10.25 GiB) float64** or
**10,747,457,100 bytes (10.01 GiB) float32**, before host history and serialization
overhead. Formula: `100*N*(5*dtype_bytes + 4 + 1 + 101*8) + 100*E*8`.
The delay ring alone is 10,223,058,400 bytes. These are shape estimates, not a
measured GPU checkpoint file. Allow roughly **11 GiB plus token histories** for
FastSim. TorchSim also saves bounded recent unfiltered legacy spike chunks,
which can add several GiB depending on activity; reserve additional space for
that history. FastSim adds an eight-byte ring slot; observer
buffer size here is only `94*100*58` bytes. Snapshotting temporarily clones
simulator tensors, so allow roughly another snapshot's worth of device memory;
TorchSim's restore also stages tensors on the device before cloning them and
can briefly require two extra snapshots' worth. Use an H100/H200 at this size.

1,000 passes × 8 column tokens × 100 copies is 800,000 output pairs, plus at
least 800,000 load events, normalized schedule entries and load-step integers.
These lists scale with tokens and retries, rather than simulation steps: expect
several hundred MiB of Python host state (roughly 0.5–1 GiB including schedules,
events and scoring/JSON temporaries), small relative to the delay ring but not
constant space. Normalized schedules, host output/load/refusal lists and their
shared event references are saved.
The 8 column outputs are one output cell per token in this build. Checkpoint
size grows modestly with completed tokens; optional spike captures can dominate.
Output/refusal histories retain observed events, including spurious events;
pathological continuing activity can grow those lists beyond the token estimate.
