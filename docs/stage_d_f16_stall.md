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

## 2026-10-14: rejected v1 selected-master simultaneous ignition

**Rejected v1 history.** The candidate opt-in was `build_pipeline(...,
master_ignition=True)`: the
**existing selected-rail COPY one-shot drives both u and v**, each at
`drive.ignite` with the ordinary 18-step delay. The original u connection is
retained; one additional synapse connects that same arm to v. This avoids
relying on the weaker u-to-v loop for the first partner spike while v is still
recovering from reset. It adds no driver, enable, retry, validity pulse or DONE
pulse. The opposite-rail and fault vetoes and the required selected stage rail
are shared by both member ignitions. Both members belong to one logical rail;
this does not energize the other rail of a bit.

### Recovery-only candidate rejected

A first candidate changed the sixteen master READY delays from 18 to 100 steps,
adding 1,312 steps / 131.2 ms, without changing any weights. Copy 75 and eight
40,000-trial first-load/reload surveys passed (nominal, weak and fast corners
in both load scenarios, slow reload and mix-B reload; 320,000 trials with zero
missing, double rails or reset survivors). **It was nevertheless rejected.**
After a granted master reset at step 1,000, a stage discard at 1,500 releases
its next DATA through ordinary stage READY. The delayed old master READY then
COPYs this new word although no new COMMIT was issued: its master u/v first
fire at 3,354 / 3,407. Ordinary timing produces no master spikes in that test.
`test_rejected_delay_candidate_copies_an_uncommitted_next_word` retains this
counterexample and checks that dual-member ignition also leaves the next word
unwritten. The delay-only circuit is not an available build option. Extending
recovery alone would need additional transaction cancellation, not merely a
larger analogue settling margin.

This selects direction (a). Storage loop weights, autapses, reset charge and
all READY/controller delays retain their old values. Driving both members can
change the storage orbit, so the earlier single-member clear results were
not treated as sufficient evidence: dual-member clear surveys are repeated
below. This is neither stronger reset nor the rejected verification/READY
interlock. No storage veto, certificate or feedback is added to READY.

### Build policy and reconstruction

The option requires default physics/drive, rate readers, experimental autapses,
selected-rail COPY and ordinary four-tap resets. It is allowed with complete
configuration F, including `f_width16_experimental=True`; the stronger
experimental register-reset combination is rejected. Generic and specialized
datapaths are supported. The input master and every cell master, including all
MULP row masters, receive partner ignition. Stage, producer, request, RAM-word
and power-up image writers retain their old wiring.

`kernel_campaign` and `stage_d` expose `--master-ignition`. Historical enabled
records for this rejected v1 candidate carried `master_ignition=true` and
`master_ignition_version=1`; the current v2 candidate records version 2.
`kernel_campaign.rebuild_record`, Stage D recheck and `stall_diag` reject an
enabled record with a missing, mistyped or changed version. Missing option
keys mean off, as in historical records. Off preserves existing topology and
build-option records; frozen digest expectations are not edited.

The diagnostic frozen-draw hand-off machinery is reused as an adverse control:
the same reset arrivals still prevent the original 3,281-q hand-off. Adding a
partner ignition at the original COPY arrival, weakened by exp(-0.12), makes
v fire. The two-member test retains both reconstructed loop and autapse weights,
both thresholds/biases, both reset trains and the 4,661-q u ignition. Off gives
only u at relative step 1,210; on sustains both members through the final
observation window. A new partner edge has no historical captured draw; it is
tested at the weak three-sigma weight and sampled independently in surveys.
These are conditional RefSim mechanism probes, not a replay of GPU strays.

### Bounded RefSim qualification

`tests/test_master_ignition.py` uses unchanged float64 RefSim, the real ordinary
four-tap controller, real READY chain, rate readers and both real required-rail
COPY arms. Each trial begins with empty storage or a live opposite old rail,
resets, COPYs the selected new rail once, checks sustained establishment of
both members, and clears again. The terminal ordinary clear also clears the
source stage/COPY so a legitimate subsequent rewrite cannot masquerade as a
reset survivor. Both storage members and readouts must be dark after clear;
a surviving old orbit cannot count as establishment. Strays cover every
fixture neuron, including dark rails and control/qualification neurons, and
clear phases are randomized. Fast tests cover absent stage rails, either
selected rail, both stage rails, fault veto, no COPY, and stage discard followed
by new DATA with no new COMMIT.

Corner weights use log offsets +/-0.12; thresholds and biases use +/-0.6 mV.
Fast/slow correlated corners include adverse incoming clear/autapse weights;
weak-loop, high-threshold and negative-bias cases are also separate. The
independent grid enumerates all 4,096 signs of two loop, two autapse, two clear,
two threshold, two bias and both ignition weights, repeated ten times per
scenario (40,960 trials). These corners apply to master storage and ignition;
controls/readouts are nominal except in mix B, which samples every edge and
neuron independently. Copy-75 surveys retain its reconstructed storage draws
and u ignition, with independent mix-B partner-edge draws and strays.

The five earlier captured clear scenarios (request28, master74, master17,
master24, stage18), fast/slow corners and independent mix B are rerun at 40,000
trials each. They reuse `test_stable_latch.clear_fixture` and its reconstruction,
reader/reset mirrors and added-edge draw law, but ignite **both members** for
the four close initial pulses before the existing 300-q kick and random clear.
Both members must be live before reset. New dual-member clear evidence is
required even though all clearing edges are structurally unchanged.

**V1 qualification was rejected.** This simultaneous-ignition circuit produced
**127 reset survivors in 40,000 fast-corner first-load trials**, despite the
earlier smoke, grid, and captured-clear passes. V2 qualification results are
recorded in [master_ignition v2: cluster qualification results](#master_ignition-v2-cluster-qualification-results);
the v1 evidence below cannot qualify it.

### Costs and latency

Measured configuration-F builds, generic datapath and ordinary resets:

| Build | Neurons, off / on | Synapses, off / on | Added neurons / synapses | Added controller latency |
|---|---:|---:|---:|---:|
| `tick2.c` | 31,363 / 31,363 | 64,238 / 64,540 | 0 / 302 | 0 ms |
| `render.c` | 11,768 / 11,768 | 24,228 / 24,332 | 0 / 104 | 0 ms |
| perspective (`render2.c`, F16) | 123,323 / 123,323 | 249,940 / 251,794 | 0 / 1,854 | 0 ms |

There is exactly one added synapse per master rail (two per bit). Ordered neuron
roles/biases and every pre-existing edge weight/delay are identical off/on.
READY and COPY times are unchanged. The nominal one-bit fixture has READY at
1,872 and the selected COPY arm at 2,009 in both builds; u first fires at 2,056.
The partner first fires at 2,116 off versus 2,056 on: **6.0 ms earlier**, with
no additional reset-to-COPY waiting in any of the three builds. The direct
partner pulse supplies ignition margin within that existing recovery budget. This is a
primitive timing measurement, not an end-to-end kernel throughput benchmark;
pipeline overlap and completion timing determine output latency. No full GPU
campaign or end-to-end throughput comparison was run here. Finite surveys do
not prove indefinite holding or safety under unbounded Gaussian tails, remove
F's separate rate-floor limitation, or diagnose the other seven freezes.

## 2026-10-14 continuation: staggered partner ignition, version 2

**Version 1 above is rejected.** Although simultaneous ignition passed both
40,000-trial copy-75 scenarios, nominal first-load/reload, the independent
4,096-corner grids, master-74 clearing and the two-cell kernel, its fast-corner
first-load survey produced **127 reset survivors in 40,000 trials**. Igniting
u and v together can establish a fast storage orbit that survives the later
ordinary clear. Successful establishment alone was insufficient. Those results
belong to the rejected simultaneous circuit and do not qualify its replacement.

The opt-in remains `master_ignition=True` / `--master-ignition`, now recorded as
`master_ignition_version=2`. The selected, qualified COPY arm retains its
original `drive.ignite` connection to u at delay 18 and adds one connection to
v at **delay 78**: the partner input arrives **60 steps / 6.0 ms later**.
This staggers the direct member inputs while supplying enough charge for the
reset-depressed copy-75 partner to fire. The partner assist follows the same
selected-rail and fault qualification as the original u input. There is no
new enable, retry, completion shortcut or change to READY/rail-selection
timing. The storage loop, autapses, reset weights and readers are unchanged.
The nominal timing regression retains READY=1,872, COPY=2,009 and u=2,056;
the assisted partner fires after u, before the unaided partner's step 2,116.
The simultaneous v1 timing measurements above are historical.

Version checks in campaign rebuild, Stage D recheck and stall diagnostics now
reject enabled v1 records, missing versions and mistyped versions. The option
is still off by default; disabled topology and build-option records retain
their frozen fingerprints. The structural comparison checks every existing
edge, neuron role and bias against off and permits only the delay-78 partner
edges. Costs remain **0 neurons and 302 / 104 / 1,854 synapses** for tick2 /
render / perspective F16, with no added controller delay.

### Cluster jobs and laptop smoke policy

All long ignition cases in `tests/test_master_ignition.py` are marked `slow`
and skipped unless `DROSO_IGNITION_CLUSTER=1`. Each parameter is an independent
pytest case so a scheduler can run it as one job. No 40,000-trial survey was
run on this laptop during this continuation. The load surveys each check
establishment, live old-data extinction on reload, opposite-rail darkness,
exactly one COPY pulse, and **clear-after-load survivors** in both members
and the readouts. They use real COPY arms, the original READY chain and
ordinary reset controllers, with random reset/clear phases and strays on all
fixture neurons. Copy-75 keeps its reconstructed draws and independently
samples the added partner weight. The independent grid uses a coprime stride
of 17 so a small smoke also varies the upper parameter axes; a complete
4,096-case pass still visits every sign combination once.

For **each ID** in the following table, run this command in a separate cluster
job, substituting that ID for `CASE_ID` (keep the brackets quoted):

```sh
DROSO_IGNITION_CLUSTER=1 uv run pytest -q -s -m slow -p no:warnings \
  'tests/test_master_ignition.py::test_master_ignition_cluster_survey[CASE_ID]'
```

| First-load case ID | Reload case ID | Trials per case |
|---|---|---:|
| `first-load-nominal` | `reload-nominal` | 40,000 |
| `first-load-fast3sigma` | `reload-fast3sigma` | 40,000 |
| `first-load-slow3sigma` | `reload-slow3sigma` | 40,000 |
| `first-load-weak_loop` | `reload-weak_loop` | 40,000 |
| `first-load-high_threshold` | `reload-high_threshold` | 40,000 |
| `first-load-negative_bias` | `reload-negative_bias` | 40,000 |
| `first-load-mix_b` | `reload-mix_b` | 40,000 |
| `first-load-copy75` | `reload-copy75` | 40,000 |
| `first-load-independent3sigma` | `reload-independent3sigma` | 40,960 |

The separate clear stress cases retain the captured/controller fixtures,
reader mirrors, four close initial pulses, the 300-q kick and randomized clear
phases. Their partner inputs are shifted by 60 steps too. They check that
both members were live before clear and that neither storage nor readers
survives. Run each of these IDs as a separate job with the next command:
`request28`, `master74`, `master17`, `master24`, `stage18`, `fast3sigma`,
`slow3sigma`, `mix_b` (40,000 trials each).

```sh
DROSO_IGNITION_CLUSTER=1 uv run pytest -q -s -m slow -p no:warnings \
  'tests/test_master_ignition.py::test_master_ignition_clear_cluster_survey[CASE_ID]'
```

The two deterministic, stray-free grid jobs have IDs `first-load` and `reload`,
4,096 trials each:

```sh
DROSO_IGNITION_CLUSTER=1 uv run pytest -q -s -m slow -p no:warnings \
  'tests/test_master_ignition.py::test_all_independent_corners_establish_without_strays[first-load]'
DROSO_IGNITION_CLUSTER=1 uv run pytest -q -s -m slow -p no:warnings \
  'tests/test_master_ignition.py::test_all_independent_corners_establish_without_strays[reload]'
```

The end-to-end two-cell check is also cluster-only, and expects outputs
`[0, 3, 1]` with no faults/timeouts/bad outputs:

```sh
DROSO_IGNITION_CLUSTER=1 uv run pytest -q -s -m slow -p no:warnings \
  tests/test_master_ignition.py::test_master_ignition_two_cell_kernel_computes_and_reloads
```

Laptop smoke uses **256 trials per scenario**, enabled only with
`DROSO_IGNITION_SMOKE=1`. The 18 load cases and eight separate clear cases have
the same parameter IDs as their cluster counterparts. Each case prints its
counts and wall/CPU time with `-s`. A fixed adverse-control smoke compares the
rejected delay-18 partner edge against delay 78 on the same 256 fast-corner
first-load trials (seed 17,531): **2 simultaneous survivors versus 0 staggered
survivors**. This is a bounded regression demonstrating the removed failure
mechanism, not a qualification claim.

```sh
DROSO_IGNITION_SMOKE=1 uv run pytest tests/test_master_ignition.py \
  -q -s -m 'not slow' -p no:warnings
```

Local smoke results: **27 cases passed**. All 18 load/reload scenarios
(4,608 trials) returned zero missing establishment, double rails, old survivors,
clear-after-load survivors, non-live pre-clear observations and repeated/missing
COPY arms. All eight clear stress scenarios (2,048 trials), including
`master74`, returned zero storage/reader survivors and zero non-live pre-clear
observations. The adverse-control case additionally passed the expected
2-versus-0 survivor comparison on 256 trials per design. The isolated smoke
command was:

```sh
DROSO_IGNITION_SMOKE=1 uv run pytest tests/test_master_ignition.py \
  -q -s -m 'not slow' -k 'smoke or counterexample' -p no:warnings
```

For this isolated checkout, commands used a writable `UV_CACHE_DIR` under the
task temp directory, the existing shared `UV_PROJECT_ENVIRONMENT` with
`UV_NO_SYNC=1`, and `PYTHONPATH="$PWD"`. A separate run with the cluster flag
unset skipped **all 29 slow cases**, confirming the default laptop guard.

The required foreground regression **passed**, with the smoke flag enabled so
the new 256-trial cases ran too. This includes the unchanged frozen default
netlist hashes and the enabled-record reconstruction/version checks:

```sh
DROSO_IGNITION_SMOKE=1 uv run pytest tests/test_build_options.py tests/test_master_ignition.py tests/test_config_f_kernels.py tests/test_stall_diag.py -q -m 'not slow' -p no:warnings
```

### master_ignition v2: cluster qualification results

On 2026-10-14, all 26 opt-in cluster cases (`DROSO_IGNITION_CLUSTER=1`,
`tests/test_master_ignition.py::test_master_ignition_cluster_survey` and
`tests/test_master_ignition.py::test_master_ignition_clear_cluster_survey`)
were run at commit `cbb4cc1` on Juno and G2 CPU nodes. Every case passed, every
failure counter was zero, and the cases completed **1,041,920 trials** in
total. The reported counter values and trial counts are:

| Case | Trials | Result |
|---|---:|---|
| ignition clear 2 fast3sigma | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition clear 2 master24 | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition clear 2 mix_b | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition clear 2 slow3sigma | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition 2 copy75 empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 high_threshold empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 mix_b empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 nominal empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 weak_loop empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 fast3sigma reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 independent3sigma reload | 40,960 | `copies=40960, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 negative_bias reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 slow3sigma reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition clear 2 master17 | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition clear 2 master74 | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition clear 2 request28 | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition clear 2 stage18 | 40,000 | `copies=40000, survivors=0, not_live_before_clear=0, reader_survivors=0` |
| ignition 2 fast3sigma empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 independent3sigma empty | 40,960 | `copies=40960, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 negative_bias empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 slow3sigma empty | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 copy75 reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 high_threshold reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 mix_b reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 nominal reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |
| ignition 2 weak_loop reload | 40,000 | `copies=40000, selected_rail_not_established=0, double_rails=0, old_rail_survivors=0, survivors=0, multiple_copy=0` |

The scheduler job IDs were: `fast3sigma` Juno 457262;
`first-load-copy75` Juno 457266; `first-load-fast3sigma` G2 4018677;
`first-load-high_threshold` Juno 457267; `first-load-independent3sigma` G2
4018678; `first-load-mix_b` Juno 457268; `first-load-negative_bias` G2
4018679; `first-load-nominal` Juno 457269; `first-load-slow3sigma` G2
4018680; `first-load-weak_loop` Juno 457270; `master17` G2 4018673;
`master24` Juno 457263; `master74` G2 4018674; `mix_b` Juno 457264;
`reload-copy75` G2 4018681; `reload-fast3sigma` Juno 457271;
`reload-high_threshold` G2 4018682; `reload-independent3sigma` Juno 457272;
`reload-mix_b` G2 4018683; `reload-negative_bias` Juno 457273;
`reload-nominal` G2 4018684; `reload-slow3sigma` Juno 457274;
`reload-weak_loop` G2 4018685; `request28` G2 4018675; `slow3sigma` Juno
457265; `stage18` G2 4018676.

V2 now passes its bounded RefSim qualification. It has not yet run in a full
kernel campaign; the next step is a 100-copy 16-bit perspective campaign with
it enabled. The copy-85/57 operand-XOR freeze class is not covered by this
option.
