# Performance comparison suite

`python -m drosophilos.bench.perf_campaign` implements the fixed comparison suite in
`docs/perf_campaign.md` §3.  It writes one `<out>.json` full record and one `<out>.md`
comparison table. Primitive spike totals use `capture_spikes` for every neuron on node 0;
that is intentionally restricted to the short primitive runs because it adds host trace-copy
and memory cost. Render runs do not collect a full spike trace.

## Bare invocation is a smoke test

`uv run python -m drosophilos.bench.perf_campaign` with no arguments runs only:

- `--levels primitive`
- `--repeats 1`
- `--primitive-blocks cells,fanout,tick` (`cells` expands to separate ADD, AND, XOR, MOV streams)
- `--copies 1,8`
- `--tokens 16` (the documented minimum)

`perspective` stays selectable (`--primitive-blocks perspective` or `cells,fanout,perspective,tick`)
but is not in the default set: its 16-bit multiplier kernel needs ~6.6 neural seconds per token,
against ~1.2 for cells/fanout/tick, so it is the block most likely to need a longer cap or a
cluster GPU.

The `small` and `historical` levels are never on by default; name them explicitly with `--levels`.
On `--device cpu` they additionally refuse to run unless `--allow-cpu-renders` is given — these
are cluster workloads (see below), and the guard's error message says roughly what they cost: an
8 x 5, three-frame `doom2.c` render took ~2h43m of wall time on four CPU copies per `RESULTS.md`.

## Primitive neural-time cap

`_run_primitive` passes a neural-time cap to `run_pipeline_batched`, sized with
`--primitive-max-s` (default 150s), not a fixed 30s. 16 tokens of the `perspective` kernel need
~110s of neural time alone, so a fixed 30s cap cut that block off mid-run and reported it as a
missing/host-stall result rather than a slow one. Each primitive record carries the cap it ran
under as `max_neural_s`. When a block still hits the cap, its `host_stalls` field is already
true; the Markdown table additionally marks that row's `truncated` column `stalled@cap` so it
reads as a sizing problem, not a measurement.

## Incremental report

`write_report` is called after every completed configuration (all its repeats and any profile
run), not only once at the end. The JSON carries `complete` (`false` until the very last write,
`true` only then) and a `configurations_planned` / `configurations_done` pair; the Markdown
header shows `Status: partial — N/M configurations, still running.` until the run finishes. A
job killed by a laptop Ctrl-C or a Slurm `--time` limit still leaves the results computed so far,
readable at `<out>.json`/`<out>.md`.

## Sizing a run before you launch it

The default suite contains:

- `cells` (separate one-cell ADD, AND, XOR, and MOV streams), fanout, the perspective
  multiplier (pipelined and array variants, selectable), and tick, each with at least 16 tokens;
- three 8 x 5 frames of `doom2.c`, using `2,258,0`, and `doom4.c`, using `6,256,0` (named levels only);
- one and eight copies on the same explicitly selected Torch backend/device/dtype.

Use `--dry-run` to print the selected configurations plus a labelled, per-configuration
`estimated_neural_s` sizing estimate (`cost_estimate` in the JSON): primitive blocks are
tokens x a per-block per-token constant (cells/fanout/tick ~1.2s, perspective ~6.6s), capped at
`--primitive-max-s`; small/historical renders are pixels x frames x an unmeasured
~30 neural-seconds/pixel (doom4) or ~15 (doom2) at one copy, divided by the configuration's
copy count. This is a sizing estimate only, not a measurement — `cost_estimate_note` says so,
along with a wall-time multiplier note: wall time runs roughly 2-15x the neural-time estimate
depending on device. Use it to pick a `--time` limit before submitting a cluster job, not to
report a result.

The reference-frame coverage gate runs before every small-render timing. Doom2's turn then
forward sequence changes the rendered state, and its later reference frames differ. Doom4's
`6,256,0` is the sequence used by the render test: all three frames differ and each contains
at least one of the imp-only palette colours 15, 10, or 9. A static `0,0,0` Doom2 sequence is
rejected with an instruction to adjust the input sequence.

Run a quick smoke test with:

```
uv run python -m drosophilos.bench.perf_campaign --out data/perf/smoke
```

Run the full CPU primitive+small suite (accepting the render cost) with:

```
uv run python -m drosophilos.bench.perf_campaign \
  --levels primitive,small --device cpu --copies 1,8 --repeats 3 --allow-cpu-renders \
  --out data/perf/cpu-suite
```

Use `--dry-run` to print selected configurations and their cost estimate, `--profile-steps N`
for separately recorded diagnostic runs (never headline medians), and `--primitive-blocks cells`
for the quick one-cell check. The Markdown has unchanged-circuit simulator, changed-circuit, and
combined sections; repeat medians/min/max for wall time and first-frame latency are retained in
JSON.

`historical` selects 40 x 25 Doom2 (`2,258`, two frames) and 24 x 15 Doom4 sprite (`6,256`,
two frames). It is disabled without `--historical` and is H200-only: it can take hours.

## Running on the cluster

`small` and `historical` belong on a cluster, not a laptop. Use the G2 launcher from the repo
root:

```
slurm/submit.sh --time=4:00:00 -- \
  drosophilos.bench.perf_campaign --levels primitive --device cuda --out data/perf/<name>
```

pushes HEAD, checks it out on Juno (default; `--cluster g2` for ganymede2 with a `--gres` GPU
choice), and submits `slurm/<cluster>.sbatch`; `slurm/fetch.sh <jobid>` brings
the log and results back. Size `--time` from a `--dry-run` cost estimate on the same `--levels`
and `--copies` first, remembering the 2-15x wall/neural multiplier.

Measured cluster cost (Juno job 412136, one H200 NVL, commit 23a8966, 2026-09-18): the full
primitive level — cells, fanout, perspective (array and pipelined multiplier), tick; copies 1 and
8; one repeat — took **1 h 16 min of wall time** for ~640 s of neural time per copy set; the
largest block (perspective, pipelined, 8 copies) is 417 s wall. The report is
`docs/perf/juno-h200-primitive.{md,json}`. The `small` level has not completed yet (Juno 412137).
Every configuration finished with 0 wrong, 0 missing, nothing truncated.
