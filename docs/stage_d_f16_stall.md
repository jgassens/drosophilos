# Perspective F16: copy 75 loses a master-rail ignition

The supplied five 100-copy mix-B campaigns freeze **8/500 copies** (seed 110:
75, 85; 114: 43, 51, 57; 116: 7; 117: 6, 20), all within the first six of
16 outputs, with no wrong outputs, faults, or timeouts. The nominal run is
16/16. This analysis establishes the mechanism for **seed 110, copy 75** only.

**Copy 75 is a reset-recovery / master-write stall, not the tick2 master-survivor
stall.** On r4's **first transaction**, the selected COPY arm fires, master
`c4_mulp.r4.M.b40r1.u` (**17531**) fires once at **112,208**, and its partner
`M.b40r1.v` (**17532**) never fires. Bit 40 is carried operand **B5**, not result
bit 40. Its missing sustained rail prevents valid40, six completion joins, and
DONE. No old master rail, valid latch, or completion latch survives a clear:
this master has never held a word before this transaction.

A small RefSim experiment at reconstructed copy-75 static draws reproduces the
missing partner spike with the captured reset arrivals and observed u spike.
Removing those reset inputs makes v fire. Thus the established first break is
failed rail establishment; **residual reset hyperpolarisation at a weak,
high-threshold, negatively biased partner is a sufficient reproduced mechanism**.
The dump contains spikes, not voltages or GPU stray events, so the exact captured
membrane trajectory is not reconstructed.

## Artifacts and method

Inputs are under
`/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:

- Campaign: `perspective_f16_cal_s110_9206797.json`.
- Full capture: `perspective_f16_s110_copy75_dump.npz`.
- Initial localization: `reports/perspective_f16_s110_copy75.md`.
- Adjacent replay metadata: `perspective_f16_s110_copy75_replay.json` (a 30 s
  capture run, shorter than the calibration campaign).

`build_tick_pipeline` reconstructs **123,323 neurons / 249,940 synapses** from
the record: rate readers v2, experimental autapses v1, ordered XOR one-shots v1,
F16 opt-in, ordinary four-tap register resets at 0.75 loop strength. The compact
dump has **103,539,612 spikes**, **35,467 firing IDs**, explicit capture of all
**123,323 IDs**, and steps **23–299,999**, at **0.1 ms/step**. `load_dump` verifies
all recorded roles with **zero ID drift**. Silent partners and gates below are
captured, not missing probes. A single-copy GPU replay is not used: the stray
stream depends on the original batch of 100 copies.

The new read-only `completion_view` in `drosophilos/bench/stall_diag.py` exposes
cell-local rails, valid/tree/control probes, COPY arms, pre-window counts, and
reset fan-out. It deliberately reports spike counts rather than treating one u
spike as a live latch. `arrival_alignment` brackets inhibitory arrivals with
target spikes without inferring voltage or within-step ordering. For example:

```python
import json
from pathlib import Path
from drosophilos.bench.stall_diag import (
    build_tick_pipeline, load_dump, completion_view,
)

base = Path('/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d')
_, _, pl = build_tick_pipeline(json.loads(
    (base / 'perspective_f16_cal_s110_9206797.json').read_text()))
dump = load_dump(base / 'perspective_f16_s110_copy75_dump.npz', pl.net.roles)
view = completion_view(pl, dump, 'c4_mulp.r4', 110405)
print(view['bits'][40])
print(view['signals'][17531], view['signals'][17532])
print(view['reset_emissions'])
```

Counts below span the complete capture unless a window is stated. All names in
the next sections have prefix `c4_mulp.r4.` unless specified otherwise.

## From commit to the first missing spike

| Probe | ID | Captured evidence |
|---|---:|---|
| `Q.comp.c5_0.L.u` | 17320 | First 109,610; holds through 299,998 |
| `autocommit.edge` | 18902 | 109,655 |
| Commit command (`cell.commit_pulse`) | 122411 | 110,405 |
| `commit_in` | 18456 | 110,445 |
| `commit.u` | 18454 | First 110,485; holds through 299,963 |
| `guardd.d11` | 18468 | 111,057 |
| `grant.edge` / `grant.L.u` | 18471 / 18469 | 111,101 / first 111,141 |
| `M.reset` | 18432 | 111,181, exactly once |
| `M.ready` | 18453 | 112,036, exactly once |
| `copy.u` / `copy.v` | 18474 / 18475 | First 112,076 / 112,133; both hold |
| `cp40r1.driver.edge` | 18800 | 112,119, exactly once |
| `cp40r1.edge` | 18802 | 112,164, exactly once |
| `M.b40r1.u` | 17531 | **112,208 only** |
| `M.b40r1.v` | 17532 | **No spikes** |
| `M.b40r1.u.rate` | 17937 | 112,236; 112,263; 112,294; 112,336 only |
| `M.valid40.or` | 17933 | **No spikes** |
| `M.valid40.L.u/v` | 17934 / 17935 | **No spikes** |
| `M.valid40.ign.edge/edge_inh/hold_inh` | 17938 / 17939 / 17940 | **All silent** |
| `M.comp.c5_0.L.u/v` (W_M) | 18425 / 18426 | **Both silent** |
| `done.edge` / `done.edge_inh` | 18884 / 18885 | **Both silent** |
| `Q.reset` / `Q.reset_inh` / `Q.ready` | 17327 / 17328 / 17348 | **All silent** |

The COPY arm delivers **+4,655 nominal q, delay 18**, arriving at u at
**112,182**; u fires 26 steps later. Its only positive partner drive is the loop
edge to v, **+3,621 nominal q, delay 18**, arriving at **112,226**. There is no
v spike and hence no return loop spike to sustain u. The brief rate-reader burst
does not light the validity OR. The selected stage rail `Q.b40r1.u/v`
(**16426/16427**) starts at **105,259/105,310** and holds to
**299,988/299,957**; the opposite stage rail and `cp40r1.veto` (**18803**) are
silent. The selected input and COPY qualification did their jobs.

All **51 selected COPY arms fire exactly once**, between **112,151–112,170**;
all 51 opposite arms stay silent. All selected master-u members fire, but **only
bit 40 fails to establish its loop**. All other selected master rails hold, all
opposite master rails are dark, and the other **50 valid latches** start between
**112,518–112,576**. Q's 51 valid latches and all 50 Q completion latches are
present and firing. The selected packed word has ones at **8, 25, 37, 40, 41**:
R=256, C/Z/V=0, A=64, B=100. The terminal master is missing B5.

Exactly these master tree latches (both members) never ignite; the other **44 of
50** tree latches do:

| Join | u/v IDs |
|---|---|
| `M.comp.c0_20.L` (valid40 AND valid41) | 18193 / 18194 |
| `M.comp.c1_10.L` | 18313 / 18314 |
| `M.comp.c2_5.L` | 18377 / 18378 |
| `M.comp.c3_2.L` | 18401 / 18402 |
| `M.comp.c4_1.L` | 18417 / 18418 |
| `M.comp.c5_0.L` (root) | 18425 / 18426 |

The root's other input, `M.comp.c4_0.L.u` (**18409**), starts at **115,074**.
This is one missing leaf propagating up its actual dependency branch, not a
global completion-tree or DONE-inhibitor failure. DONE is downstream of W_M;
stage reset and IDLE re-arm require DONE. Their absence leaves the stage busy.
The neighboring r3 subsequently completes its third stage at **128,237** but
cannot commit, consistent with downstream backpressure. The blocked DONE remains
absent for **189,594 steps** after the command, exceeding stall_diag's
**26,320-step** workload threshold. Capture truncation alone is not the diagnosis.
All r4 fault probes are silent. A missing rail does not satisfy the double-rail
fault AND, and these internal staged commits have no producer watchdog
(`_NoProducer`); missing completion therefore need not produce a fault or timeout.
Copy 75 has published **zero outputs**.

## No reset survivor: what the inhibitory pulses actually did

Before its reset at **111,181**, **no r4 master data rail, validity latch, or
completion-tree latch has ever fired**. This is its first START (**104,219**),
first stage completion, and first commit. Therefore there is no surviving master
period and no refractory-phase entrainment to measure. The stage is active, with
Q bit-40 member periods of **63 steps** around master reset; it receives **no Q
reset at all**, so retaining Q does not indicate a failed stage clear.

The master reset inhibitor **18433** emits **five** spikes (four configured taps
plus the trailing spike). Every edge to a master rail member has nominal
**−2,716 q / 18 steps**. There are **711 negative fan-out edges**, including
rate-reader reset mirrors. At both bit-40 members the arrivals are:

| Inhibitor emission | Arrival | Steps before the only u spike (112,208) |
|---:|---:|---:|
| 111,231 | 111,249 | 959 |
| 111,275 | 111,293 | 915 |
| 111,319 | 111,337 | 871 |
| 111,366 | 111,384 | 824 |
| 111,439 | 111,457 | 751 |

Neither member has any earlier spike; **none of these pulses lands in a
post-spike refractory interval**. The final arrival precedes the COPY input to u
by **725 steps (72.5 ms)** and the loop input to v by **769 steps (76.9 ms)**.
They inhibit already empty storage. Current decays with tau_s=5 ms, but the
voltage perturbation recovers on tau_m=20 ms; a negligible remaining synaptic
current does not imply full membrane recovery.

## Reduced reproduction and limits

Static draws were regenerated in `make_perturbed_sim` order using
`default_rng(110)`, B=100: log-normal weights in the **sorted topology edge
order** `(100, 249940)`, then threshold and bias normals `(100, 123323)`; copy
index **75**. Rail neurons have zero nominal bias. These yield:

| Parameter | u (17531) | v (17532) |
|---|---:|---:|
| Threshold, mV | −45.12705001632755 | **−44.42221141037541** |
| Bias, mV | +0.03693153370274249 | **−0.7152384644970224** |
| Incoming reset edge, q | −2618 | −2707 |
| Self-inhibition, q / delay 0 | −737 | −692 |

The u→v loop is **3281 q** (nominal 3621, −9.39%); v→u is **3792 q**. The
COPY ignition into u is **4661 q** (nominal 4655). v's threshold is +2.89 sigma
and its bias is −3.58 sigma: this partner has a narrow first-spike margin.

`test_copy75_partner_handoff_at_reconstructed_draws` isolates **v alone** in
RefSim, initially at its long-empty resting equilibrium **−52.7152384645 mV**.
It prescribes the five captured inhibitory arrivals at −2707 q and the observed
u spike's **3281-q loop arrival at 112,226**. No unrecorded strays are supplied:

- With reset: **no spike**. v peaks at **−44.46283430295 mV at step 112,322**,
  about **0.04062 mV below threshold**. At the loop arrival it is still at
  **−53.72250690493 mV**, roughly 1.0073 mV below its resting equilibrium.
- Without reset: v **fires at 112,287** with the identical loop input and static
  parameters.

An additional two-neuron RefSim check, using both reconstructed loop/autapse
edges, both reset trains, and the COPY input, produces only u at **112,210**.
Removing reset establishes the loop (first u/v **112,205/112,284**); delaying
COPY by 1,000 steps also establishes it (**113,205/113,284**). Nominal parameters
with reset establish it (**112,211/112,271**). Each check is 4,500 steps on two
neurons and takes seconds. The two-step difference from captured u=112,208 is
consistent with omitted stray input; **that explanation is a hypothesis**, not
an exact GPU trajectory match.

v's autapse cannot directly cause its first-spike failure: it has never fired
and has delivered no self-inhibition. Likewise wider outgoing fan-out does not
divide the loop weight or consume source voltage in this simulator. All wired
inputs to this v are its partner, its reset inhibitor, and its own autapse.
There is no other gate or wide-word loading input to explain the handoff.

## Relation to tick2, early failures, and the target of a fix

The [tick2 completion stall](stage_d_completion_stall.md) has an old fast rail
surviving master reset in a near-coincident ~34-step mode. Its validity OR
resumes while the valid ignition one-shot remains inhibited. Here the old
master is empty, five reset spikes occur, the new rail does not establish,
**valid40.or and its edge inhibitor never fire**, and no retained latch is
observed. The shared symptom is commit without renewed W_M/DONE; the initiating
failure is different. The tick2 stage-Z0 survivor and request-clear entrainment
also require previously active storage, absent here. This is neither an XOR
double-rail fault nor a missing commit request.

The broader **reload-margin problem was already identified** during reset and
experimental-autapse qualification (see [stable latch recovery](stable_latch.md)
and the completion-stall document's reload frontier). Copy 75 is direct dump
evidence of that class in F16's ordinary master reset/COPY path, including a
failed first load into an empty master. It does not establish that experimental
autapses themselves cause the missing v spike, or that the same static draw
would exist in a differently sized netlist without that option.

Early occurrence needs no accumulating damage: thresholds, biases, and weights
are fixed per copy, and the master reset precedes **even its first write**.
`perspective` uses heading=3; its first column has r=64, and MULP carries the
constant B=100 (B5=1). Thus this vulnerable rail is selected immediately. A
single lost COPY ignition is not retried: COPY and the one-shot driver's
inhibitor continue firing, but `cp40r1.edge` never emits a second pulse. Small
stray kicks never restart this rail through capture end.

**Hypothesis for campaign-wide early clustering:** sixteen 51-bit MULP row
masters expose 816 selected rail latches per transaction, plus their stages
and control, increasing opportunities to encounter an adverse fixed draw early.
The enlarged reset fan-out is delivered edge by edge without a load-dependent
weight penalty; width may change exposure and sampled extremes, rather than
introducing a new fan-out physics effect. The other **seven frozen copies have
not been traced here**; their shared cause and the 8/500 population's early-time
distribution remain unproven. The nominal 16/16 run does not challenge these
perturbed reload margins.

A fix would need to target the **reset-to-COPY recovery budget and reliable
one-pulse establishment of the selected master loop**, including independent
weak-loop / high-threshold / negative-bias corners and initially empty storage.
Any retry or READY verification would need to detect incomplete new storage;
making valid or DONE fire without that storage would conceal the failure.
Stronger clearing alone could worsen this mechanism. Both old-data extinction
and new-data establishment need qualification under actual COPY timing, along
with the existing reset-survivor tests. No circuit or builder was changed here.

Validation **passed**:
`uv run pytest tests/test_stall_diag.py -q -m 'not slow' -p no:warnings`.
This isolated worktree used a writable `UV_CACHE_DIR`, the existing shared
`UV_PROJECT_ENVIRONMENT` with `UV_NO_SYNC=1`, and `PYTHONPATH="$PWD"` so pytest
imports this worktree rather than the shared environment's editable checkout.
The new helper was also exercised on the complete external dump: no unavailable
probes, the five reset emissions above, and exactly one u spike / zero v spikes.
