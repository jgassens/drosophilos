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
  Tick *t* is complete when all three cells have committed their *t*-th word.
- **Pass**: every copy's canonical state equals the reference's at every one of the 1,000 ticks,
  with no commit beyond the requested ticks. Verdict `"exit met"` only then.

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

Per copy: `requested` (ticks), `completed` (all three fields committed), `matched` (ticks equal
before the first mismatch), `wrong` (1 at the first mismatch; counting stops there — a diverged
state makes every later tick wrong and says nothing more), `unscored` (completed after it),
`missing`, `duplicates` (commits beyond the requested ticks), `invalid` (undecodable words),
`refusals`, `retries`, the first mismatch (`tick`, `token`, `field`, `expected`, `got`, both
whole states, step, and a class: `state lags a tick` / `previous word applied twice` / `other`),
and `tick_ms` (neural ms from the first load to each tick's last commit).

Status per copy: `matched`, `wrong`, `blocked` (a word refused `max_retries` times: fail-stop),
`truncated` (the run hit `--max-ms`, a resource limit), `stalled` (no output from any unfinished
copy for `--stall-ticks` × the calibrated tick time), `unfinished`.

Run-level: `faults`, `timeouts`, `bad_outputs`, `blocked_nodes` (the runner counts these for the
whole batch, not per copy), `build_options` (the build's `pl.build_options`), `tokens`,
`reference_canonical_hex`, `three_point_check`, `max_ms` and its source, the calibration,
`tick_wall_s_copy0`, and `verdict`.

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

# mix B, 100 perturbed copies (~18 h; a run cut at max_ms could take ~27 h)
slurm/submit.sh --time=2-00:00:00 -- drosophilos.bench.stage_d --ticks 1000 --seed 108 \
    --backend torch-fast --device cuda --copies 100 --mix B --out data/stage_d/mixB_s108_c100.json

# results
slurm/fetch.sh <jobid> data/stage_d/nominal_s1.json
```

On G2 add `--cluster g2 --gres=gpu:nvidia_h200_nvl:1` (not a 3090: slow at float64).
`--backend torch` (TorchSim) is the comparison backend: ~5.9× slower (docs/perf_final_comparison.md).

Estimates. Measured on the laptop (2026-09-25, RefSim, nominal, the 6-tick test): tick commits
at 12.1 / 17.6 / 23.9 / 29.5 / 35.8 / 41.3 s neural after the first load — first tick **12.1 s**,
then **5.84 s** per tick on average (alternating ~5.5 and ~6.35 s); a 2-tick FastSim run gave the
same first two ticks. 1,000 ticks ≈ 12.1 + 999 × 5.84 ≈ **5,850 s neural** (1.6 h). One H200,
FastSim: single copy at ~1.08× wall/neural (tick-kernel row, docs/perf_final_comparison.md) ≈
**6,300 s ≈ 1.75 h**; 100 copies at ~11× ≈ **64,000 s ≈ 18 h**. The calibrated default `max_ms`
is ≈ 8,700 s neural, so a 100-copy run cut there would take ≈ 27 h: hence `--time=2-00:00:00`.
On the laptop, RefSim ran at ~5.4× wall/neural under heavy load (226 s for 6 ticks).

## 7. What this does NOT cover

- **TMR and the commit log**: the plan's Stage D puts `minidoom` `p_*` on control nodes with
  triple modular redundancy and a commit log. Neither exists; this is the resident-kernel form
  on one simulated node per copy. "A retried commit is applied once" is checked only for the
  host's input-refusal resend, not for a commit-log replay.
- **doom4 / minidoom**: `tick2.c` is a toy world update (three 8-bit state words). A pass here
  must stay labeled as the toy tick (§6: "A toy tick result must remain labeled as such rather
  than being generalized to doom4"); doom4's netlist nondeterminism is a separate open item.
- **Per-copy faults and timeouts**: the runner reports them for the batch only.
- **Timestep refinement and CUDA transaction-level certification**: outstanding (§6).
