# Stage D exit: 1,000 ticks, canonical state equal every tick

`drosophilos/bench/stage_d.py`, `tests/test_stage_d.py`. Status (2026-09-25): harness built
and tested on the laptop (6 neural ticks on RefSim match every tick); **the 1,000-tick run has
not been run yet** — it is a cluster job, commands below.

## 1. The exit

docs/plan.md, Stage D: *"1,000 ticks of scripted and fresh random input, canonical state equal
to the reference every tick; a retried commit is applied once."* docs/perf_campaign.md §6 asks
for it on the explicitly named world-update program; docs/review-2026-09-13.md §5 asks for
canonical serialized state (field order, widths, byte order, padding) rather than a raw dump.

- **Program**: `examples/tick2.c`, the world update with fresh input every tick, in kernel form:
  `compile_kernel(prog, loop_body(prog), "i")` (the same kernel as
  `bench/kernel_campaign.py block("tick")`, 13 cells). One tick per input token; the token is
  the velocity. `main`'s own `i = 8` does not apply: the kernel runs as many ticks as it is
  given tokens, so 1,000 ticks = 1,000 tokens.
- **Neural run**: `build_pipeline(..., outputs=[c4_sel, c9_sel, c12_sel])` — the kernel's two
  outputs (px, mx) plus health's carrier cell, so the host decodes every committed state word.
  Inputs use the runner's ordinary unpaced token schedule. The kernel is a pipeline: it can
  accept tick *t*+1's input while tick *t*'s downstream state outputs remain in flight.
  Consequently input-load intervals are not state-output windows.
  Each canonical field commits once per tick, so the k-th commit of a field is tick k's value.
  Tick *k* is complete when all three fields have a k-th commit. The runner's final (last-retry)
  load event for tick *k* is only a causal lower bound: every k-th field commit must occur after
  it. The next tick's load is never an upper bound. Each field must have exactly as many commits
  as requested ticks; extras are duplicates, while a trailing shortfall is incomplete.
- **Pass**: every copy's canonical state equals the reference's at every one of the 1,000 ticks,
  with exactly one commit from every state cell per tick, no missing or extra commits, and no
  run-level fault or timeout latch. A run ended at `--max-ms` cannot pass. Verdict `"exit met"`
  only then. `faults` and `timeouts` are batch counters from `lib/kernel.py`, not values that can
  be attributed to an individual copy; the record labels that scope explicitly. Refusals and
  retry counts are reported with the verdict counters but do not alone fail an otherwise correct
  resend.

## 2. Canonical state (`stage-d-canonical-v1`)

| offset | field | state cell | init | width | encoding |
|---|---|---|---|---|---|
| 0 | `px` | `c4_sel` | 20 | 8 bits, 1 byte | unsigned, big-endian |
| 1 | `mx` | `c9_sel` | 90 | 8 bits, 1 byte | unsigned, big-endian |
| 2 | `health` | `c12_sel` | 100 | 8 bits, 1 byte | unsigned, big-endian |

- Order: declaration order in `tick2.c`. `load_kernel` checks it against the compiler's
  `state_cells` (sorted by variable address) and refuses to run if they ever differ.
- Width: the program's word width (`prog.width` = 8); each field takes ceil(width/8) bytes.
  No padding between or after fields; at wider widths the unused top bits are zero.
- Excluded: `vel` (the token itself), `dist` and `contact` (written before they are read every
  tick, so they carry nothing between ticks), `i` (the loop counter; the kernel form has none).
- `canonical_state(values)` raises on a missing field, a non-integer (an undecodable neural word
  is `None`) or a value out of range — a comparison never passes by masking.

Example: after tick 33 of the scripted prefix the state is px 66, mx 66, health 246 → `4242f6`.

## 3. The reference and its cross-check

- **Oracle**: `compiler.kernel.kernel_outputs` with the three state cells as outputs — the IR
  semantics the compiler lowers, evaluated cell by cell with feedback reads taking the previous
  tick's value.
- **Three-point check** (review §5: portable C ↔ IR interpreter ↔ neural), run by the CLI before
  the neural run, which it refuses to start if the references disagree:
  - IR interpreter (`isa.ir.interpret`) on **every** tick;
  - portable C (`compiler.golden.run_golden`, clang `-fsanitize=undefined`) on the first
    `--c-ticks` (default 100).
  Both run a rewritten `tick2.c`: the state initializers set to the chunk's starting state, the
  loop count set to the chunk length (≤ 50 ticks: `i` is `u8` and the C shim holds 64 inputs,
  256 outputs), and `out_pixel(px); out_pixel(mx); out_pixel(health);` added before `i = i - 1`.
  A test checks that the unmodified `tick2.c` agrees with the same reference, so the rewrite
  cannot hide a difference.
- Measured: oracle = IR on 1,000 ticks and = C on 100 ticks (seed 1), ~8 s on the laptop.

## 4. Inputs

`tokens_for(ticks, seed)`: a 53-tick scripted prefix, then fresh bytes from
`numpy.random.default_rng(seed)`. The prefix, with the reference state it produces:

| ticks | tokens | what it exercises |
|---|---|---|
| 0–7 | 5 5 250 3 0 40 40 40 | the campaign tokens; west-wall wrap at 7 (107 + 40 → px 0) |
| 8–12 | 0 1 127 128 255 | edge values; 1 + 127 = 128 wraps |
| 13 | 66 | px beside the monster: contact, health 100 → 90 |
| 14–33 | 0 × 20 | monster steps onto px every other tick: health 90 → 0 → **246** (unsigned wrap) |
| 34–41 | 2 × 8 | player and monster move together: contact every tick, health 236 → 166 |
| 42–46 | 127 1 126 129 254 | wrap, creep, px = 127 (the largest), wrap, wrap |
| 47–52 | 3 × 6 | steady walk east, the monster closing from above |

The east wall (`px == 200 → 199`) cannot be reached at 8 bits (px < 128 after the west check);
its SEL cell still runs every tick. Over 1,000 ticks (seed 1) health changes 75 times.

## 5. Reporting (§6 discipline)

Per copy: `requested` (ticks), `completed` (all three fields have their ordinal commit), `matched` (ticks equal
before the first mismatch), `wrong` (1 at the first mismatch; counting stops there — a diverged
state makes every later tick wrong and says nothing more), `unscored` (completed after it),
`missing`, `duplicates` (commits beyond the requested ticks), `invalid` (undecodable words),
`refusals`, `retries`, the first mismatch (`tick`, `token`, `field`, `expected`, `got`, both
whole states, step, and a class: `state lags a tick` / `previous word applied twice` /
`commit before input load` / `commit count` / `other`),
and `tick_ms` (neural ms from the first load to each tick's last commit). `invalid` also counts
commits that precede their own tick's final load. An extra commit is a first mismatch and names
its field and observed count; a missing trailing commit remains an incomplete liveness outcome.

Status per copy is one of `matched`, `mismatch`, `stalled`, or `truncated`. `mismatch` is the
first wrong value or commit-count/order violation. A nonmatching copy is `stalled` if its last
commit was more than `--stall-ticks` × the calibrated tick time before the run ended; its
`stopped_at_tick` identifies the last fully committed tick. It is `truncated` only if it was
still making progress when the run hit `--max-ms`. Thus a copy that ceased committing long before
a batch's time cap is not mislabeled as truncated.

Run-level: `faults`, `timeouts`, `bad_outputs`, `blocked_nodes` (the runner counts these for the
whole batch, not per copy; `fault_timeout_scope` records this), `build_options` (the build's `pl.build_options`), `tokens`,
`reference_canonical_hex`, `three_point_check`, `max_ms` and its source, the calibration,
`tick_wall_s_copy0`, `verdict_accounting` (faults, timeouts, refusals, retries, and scope), and
`verdict`.

New records retain compact `commit_events`, unpaced `load_events`, `run_end_step`, and `dt_ms` so a
completed long job can be re-judged without simulation:

```
uv run python -m drosophilos.bench.stage_d --recheck data/stage_d/mixB_s108_c100.json
```

The command rewrites that record by default (or writes `--out FILE`), recomputing per-copy status
and verdict. Records with both saved load and commit events are fully rechecked; no pacing
evidence is required. Records from before those events were retained are still
rechecked positionally for faults, timeouts, stored value mismatches, and liveness; their
`recheck` note explicitly says the exactly-once commit-count rule was not checkable. Treating
unpaced input injections as both lower and upper output boundaries was the cause of the former
six-tick false failure (`completed=2`, `matched=0`): only the lower bound is causal.

**Sizing**: without `--max-ms`, a nominal single-copy run of `--calibrate-ticks` (4) ticks on the
same backend measures the first-tick latency and the per-tick time; `max_ms = (first + ticks ×
per_tick) × --margin (1.5)`. Both the calibration and the resulting `max_ms` are recorded.

**"A retried commit is applied once"**: checked only as far as the host's refusal resend goes
(`lib/kernel.py` `retry_refused`). Per copy the record lists the retried ticks, whether each was
among the matched ticks, and `applied_twice` (a duplicate commit, or a first mismatch equal to the
previous word applied again); `retried_commit_applied_once` is true when no copy shows either.

## 6. Running it on the cluster

VPN first (GlobalProtect). From a clean tree on the laptop (`submit.sh` pushes HEAD; Juno is
the default cluster, partition h200, 1-day default time, 2-day limit):

```
# nominal, one copy (~1.75 h)
slurm/submit.sh --time=6:00:00 -- drosophilos.bench.stage_d --ticks 1000 --seed 1 \
    --backend torch-fast --device cuda --copies 1 --mix none --out data/stage_d/nominal_s1.json

# mix B, 100 perturbed copies (~18 h)
slurm/submit.sh --time=2-00:00:00 -- drosophilos.bench.stage_d --ticks 1000 --seed 108 \
    --backend torch-fast --device cuda --copies 100 --mix B --out data/stage_d/mixB_s108_c100.json

# results
slurm/fetch.sh <jobid> data/stage_d/nominal_s1.json
```

On G2 add `--cluster g2 --gres=gpu:nvidia_h200_nvl:1` (not a 3090: slow at float64).
`--backend torch` (TorchSim) is the comparison backend: ~5.9× slower (docs/perf_final_comparison.md).

Estimates. Measured on the laptop (2026-09-25, RefSim, nominal, the unpaced 6-tick test): tick
commits at 12.1 / 17.6 / 23.9 / 29.5 / 35.8 / 41.3 s neural after the first load — first tick
**12.1 s**, then **5.84 s per tick** on average (alternating ~5.5 and ~6.35 s); a 2-tick FastSim
run gave the same first two ticks. The throughput is faster than first-tick latency because ticks
overlap: later inputs are accepted while prior downstream commits are still in flight. The
commit-order scorer preserves that resident-kernel behavior and uses loads only as lower bounds.

1,000 ticks ≈ 12.1 + 999 × 5.84 ≈ **5,850 s neural** (1.6 h). Applying the measured FastSim
ratios from `docs/perf_final_comparison.md` gives about **1.75 h** wall for one H200 copy and
**18 h** for 100 copies. The calibrated 1.5× neural ceiling is ≈ 8,700 s, so a 100-copy run cut
there would take ≈ 27 h; `--time=2-00:00:00` leaves margin. On the laptop, RefSim ran at ~5.4×
wall/neural under heavy load (226 s for 6 ticks).

## 7. Diagnosing a stalled copy

The seed-108, 100-copy mix-B runs on `ad4046d` ended with four stalled base copies
(18 after 1 completed tick, 34 after 31, 74 after 323, 17 after 728) and five stalled
rate-robust copies (43 after 450, 28 after 535, 34 after 599, 24 after 654, 11 after 864).
Both had zero wrong values and only one batch fault. A stall therefore needs its own
request/commit evidence; a batch fault count cannot explain every stalled copy.

After integrating and committing the dump support, replay base copies 18 and 34 on Juno
from the beginning with 40 tokens. Keep **all 100 copies** in the simulator:

```sh
slurm/submit.sh --cluster juno --time=4:00:00 -- drosophilos.bench.stage_d \
    --ticks 40 --seed 108 --backend torch-fast --device cuda --copies 100 --mix B \
    --max-ms 450000 --dump-copies 18,34 \
    --dump-pre-ms 30000 --dump-post-ms 10000 \
    --dump-out data/stage_d/mixB_s108_c100_replay40 \
    --out data/stage_d/mixB_s108_c100_replay40.json
```

Use the integrated commit containing the same netlist as `ad4046d`; `--ref ad4046d` itself
predates these dump flags. This is the base replay; retain `--rate-robust` when replaying
the rate-robust record, and use enough ticks to reach its stalled copies.

The command writes `mixB_s108_c100_replay40_copy18.npz` and
`mixB_s108_c100_replay40_copy34.npz`. Each contains int32 `step` (int64 for steps beyond
2^31−1), int32 `neuron`, and a single `uniq`/`role_of` table mapping observed neuron IDs to
roles. `kernel_campaign` writes the same compact format. Both diagnostics also accept
old dumps containing one `role` string per spike and old compact dumps.

Juno job 428569 exposed two problems in the original writer: it retained the entire run
and expanded each spike into a fixed-width NumPy Unicode role. The current base netlist's
longest role is 37 characters, so that array alone cost **148 bytes per spike**, before
step/neuron storage and write-time copies. Continuously firing stalled latches made both
costs grow for hundreds of seconds. Its truncated file stopped at exactly 960 MiB; the
precise external I/O/stdio failure cannot be established from exit code 120 alone. These
writers neither close nor redirect stdout or stderr. The new numeric payload costs
**8 bytes per spike**, or 12 with int64 steps, plus a small role table and metadata.

`--dump-pre-ms` defaults to **30000 neural ms** before a selected copy's last output
commit. A packed ring advances with commits and retains the intervening silence while
Stage D's existing `stall_ms` watch decides whether that copy has stalled. Anchoring at
the last commit preserves the failure onset even when `stall_ms` exceeds 30 seconds.
Once the watch detects a stall, `--dump-post-ms` (default **10000 neural ms**) records
another 10 seconds and then freezes that copy. Its dump is written immediately and its
buffer released, while the other copies continue. With `--stall-ticks 0`, captures are
ordinary rolling pre-window rings; completed copies also keep only their latest ring.
Non-dump execution and the netlist are unchanged. A dump run's global stall stop allows
selected copies' post windows to finish, subject to the original `--max-ms` ceiling and
the runner's completion stop.

The maximum retained duration is approximately `dump_pre_ms + stall_ms + dump_post_ms`,
plus the runner's observation/decode latency. For `N` continuously firing selected
neurons at 213 Hz, expect `N × 213 × duration_seconds × 8` bytes per copy. For example,
1,000 active neurons over 160 seconds produce about **273 MB**. As a conservative sizing
example, if all **5,674** neurons selected by the current base filter fired continuously,
a 160-second window (30 + 120 + 10) would occupy **1.55 GB per copy**. Five such rings
use about **7.7 GB**, regardless of whether the run is 40 or 870 ticks. Writes concatenate
one copy at a time; allow another two copy payloads for concatenation and NumPy's unique
ID scratch space (about 3.1 GB in this example), plus simulator/runner memory. In the
40-tick command above, the uncalibrated watch is 225 seconds, so the conservative bound
is **2.56 GB per copy** over 265 seconds. Most selected roles do not fire continuously,
so actual sizes are smaller. int64 steps increase these numeric payload estimates by 50%.

`window_json` in each NPZ and the record's `dump_windows[copy]` give inclusive step/ms
bounds, the last commit, stall-detection step, configured pre/post durations, spike count,
and whether the full post window froze. A run ending early writes the available partial
window with `frozen: false`. `capture_ids` identifies selected neurons, including silent
ones; diagnostics report the capture window and use its end even if the tail is silent.
Each NPZ is written to a temporary file in its destination directory, flushed, and
atomically renamed. A failed write leaves the previous complete target intact (or no
target), prints a traceback to stderr, and records `spike_dump_errors[copy]`; other copies
still write. The final JSON is written even when dump writes fail, before printing the
final stdout summary. A process killed during writing can leave a `.tmp` file, never a
partial replacement `.npz`.

`--dump-roles REGEX` defaults to the filter documented by `stall_diag`:

```text
\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)
```

Fetch the replay record and both dumps (`<jobid>` is the submitted job ID), then diagnose:

```sh
slurm/fetch.sh <jobid> data/stage_d/mixB_s108_c100_replay40.json
slurm/fetch.sh <jobid> data/stage_d/mixB_s108_c100_replay40_copy18.npz
slurm/fetch.sh <jobid> data/stage_d/mixB_s108_c100_replay40_copy34.npz

uv run python -m drosophilos.bench.stall_diag data/stage_d/mixB_s108_c100_replay40_copy18.npz \
    --campaign data/stage_d/mixB_s108_c100_replay40.json --node 18 \
    --out docs/stage_d_copy18_stall.md
uv run python -m drosophilos.bench.stall_diag data/stage_d/mixB_s108_c100_replay40_copy34.npz \
    --campaign data/stage_d/mixB_s108_c100_replay40.json --node 34 \
    --out docs/stage_d_copy34_stall.md
```

`fault_diag` accepts the same Stage D record and dump with `--campaign` and `--node`.
Both tools rebuild Stage D's `[c4_sel, c9_sel, c12_sel]` outputs from its program,
`build_options`, `datapath`, and `rate_robust`, and verify the recorded neuron/role mapping.
Existing `kernel_campaign` records still use their own output selection and legacy defaults.
The original `mixB_s108_c100_rr_ad4046d.json` predates circuit-version recording, although
`ad4046d` already contains the current v2 rate-reader reset wiring. Use the replay's new
record for diagnosis: it records `build_options.rate_robust_version` and neuron/edge counts.
An unmodified versionless rate-robust record without a circuit hash or both counts remains
rejected by the existing identity guard; matching neuron roles alone cannot prove that its
synapses match. For an archived record independently confirmed to come from `ad4046d`,
a copy annotated with `build_options.rate_robust_version: 2` identifies that circuit.

Replay invariants: `tokens_for(N, seed)` is a prefix of `tokens_for(1000, seed)`, including
the random suffix after tick 52. Weight, threshold, bias, and stray-seed draws use a fresh
NumPy generator for each main run. The perturbation builder ignores its historical
`n_steps` argument; neither `max_ms` nor `stall_ms` affects draws. Calibration uses a
separate nominal single-copy simulator. The explicit `--max-ms` above skips it, recorded as
`calibration: null`, `max_ms_source: given`, and `replay.calibration_rng: skipped (--max-ms)`.
The stray stream draws a `(copies, neurons)` mask on every simulated step: changing
`--copies` to 2 to dump two copies changes that stream. Retain the seed, full batch size,
netlist, backend, device, and dtype; CPU mask-prefix tests do not certify a CUDA replay
across different Torch/CUDA versions. Time ceilings affect how much evidence is collected,
so allow time after the last healthy tick for the stall watch.

New records retain `last_commit_step_by_field` for every copy and `stall_evidence` for each
stalled copy: stop step, per-field last commits, fields still awaiting commits, and whether
the runner lists that copy in `blocked_nodes`. The runner does not currently expose its
internal pending-cell/request state; those fields are explicitly `null`, rather than an
empty list that would claim nothing was pending. Raw load and commit events remain available
for rechecking, which preserves the saved stall evidence.

## 8. Campaign D result (2026-10-09)

**`--rate-robust --experimental-autapse` met the Stage D 1,000-tick mix-B exit on generic seeds 108, 109 and 111 and specialized-datapath seed 108; it did not meet it on generic seed 110.** The exit remains the definition above: no faults or timeouts, no wrong or missing values, not truncated, and exactly-once commits checkable from saved events.

Campaign D used generic `tick2.c`, 1,000 ticks × 100 copies, mix B, commit `f71538c` (merged `28caa66`). The integrator reran `stage_d --recheck` on both passing records. Seed 108 (Juno 443026) and seed 109 (Juno 443063) each report 100/100 copies matched, 100,000/100,000 ticks, and 0 wrong, faults, timeouts, stalls, refusals, or retries. The build has 30,643 neurons and 6,430 added autapse synapses versus the rate-reader build; nominal measured steady tick time is about 5,645 ms.

For seed 108 with the same scorer: no fix (Juno 426357, `ad4046d`) had 96 matched, 4 stalled and 1 fault; `rate_robust` only (Juno 426358) had 95 matched, 5 stalled and 1 fault; `zero_once` only (Juno 441839, `7ec3ddf`) had 94 matched, 6 stalled and 0 faults; campaign C, `zero_once + robust_request_clear + experimental_register_reset` (Juno 442659, `c8e94c6`), had 99 matched, 1 stalled (copy 77 at 517 ticks), 3 faults, 1 refusal and 1 retry, so did not meet exit. The older build (Juno 425033) had 91 matched, 9 stalled and 5 faults.

Additional commit `63c41f8`, same `tick2.c`/mix-B/1,000-tick/100-copy configuration: generic seed 110 (Juno 449070) **did not meet exit** — 99 matched, copy 82 stalled after 352 completed ticks, 1 run-level fault, 99,352 matched ticks, 0 wrong, 648 missing, 0 refusals/retries. Generic seed 111 (Juno 449071) **met exit** — 100/100, 0 wrong/faults/timeouts/stalls. Specialized datapath seed 108 (Juno 449072) **met exit** — 100/100, 0 wrong/faults/timeouts/stalls, 26,683 neurons. Across generic seeds 108--111 and specialized seed 108: 4 of 5 100-copy campaigns met exit; 1 stalled copy in 500 (0.2%); 0 wrong values in 499,352 matched ticks.

For 1 stalled copy of 500, the exact one-sided 95% Clopper--Pearson upper bound on the per-copy 1,000-tick stall probability is 0.009452282208 (0.9452282208%). The one-sided Fisher exact test for fewer autapse stalls, 1/500 versus the pooled no-autapse seed-108 baselines 15/300 (4+5+6), gives p=0.00000344134. The observed autapse stall rate is about 25× lower, not zero. The seeds draw different noise; changed builds also re-draw noise by index, so these are rate comparisons, not paired-copy comparisons. Replay of generic seed-110 copy 82 with spike dumps is queued for diagnosis.

All reviewed `stage_d_*_stall.md` classes reduce to a fast or entrained latch surviving a four-pulse clear: stage Z0, request latch, or master rail. Completion's mechanism was confirmed by capture Juno 441837. The autapse caps the failure at the latch itself.

Nominal: `tick2.c`, 1 copy, mix none, seed 1, 1,000 ticks, `--rate-robust --experimental-autapse`, Juno 448740, commit `474a8ee`: exit met; 1,000/1,000 ticks matched, 0 wrong, faults, timeouts, missing, refusals or retries; 30,643 neurons, 5,736.6 s neural, 3 h 10 min wall on one H200. The configuration's mix-B exit is met on generic seeds 108, 109 and 111 and specialized seed 108, not generic seed 110.

Open items: the autapse is formally unqualified on the isolated 44-step rate floor and reload margin; it is applied to all 3,215 kernel latches while the 40,000-trial surveys cover only request, master and stage cases; three generic seeds and one specialized campaign have passed; the option remains experimental and non-default. Review follow-ups remain: restrict the transform to the reviewed kernel construction order (after all consumers/controllers, including compact resets, and before any later reader installation), and add a version key for the experimental-autapse netlist so diagnostic rebuild/recheck can reject an ambiguous future transform. Toy tick results remain labeled as such.

## 9. What this does NOT cover

- **TMR and the commit log**: the plan's Stage D puts `minidoom` `p_*` on control nodes with
  triple modular redundancy and a commit log. Neither exists; this is the resident-kernel form
  on one simulated node per copy. "A retried commit is applied once" is checked only for the
  host's input-refusal resend, not for a commit-log replay.
- **doom4 / minidoom**: `tick2.c` is a toy world update (three 8-bit state words). A pass here
  must stay labeled as the toy tick (§6: "A toy tick result must remain labeled as such rather
  than being generalized to doom4"); doom4's netlist nondeterminism is a separate open item.
- **Per-copy faults and timeouts**: the runner reports them for the batch only.
- **Timestep refinement and CUDA transaction-level certification**: outstanding (§6).
