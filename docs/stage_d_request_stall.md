# Stage D: the request/clear stall of seed 108, `rate_robust` copy 28

**2026-10-04 update:** `robust_request_clear=True` is now an opt-in compact
request-only clear. The candidate evaluation and qualification are below under
"Compact request clear". Defaults and the control machine retain their exact
netlists. The fifth-tap proposal is rejected on its slow-corner reload margin.

**Mechanism established.** The false ("consumed") rail of `c4_sel`'s request from
`c2_sel` (latch 15701/15702) survived the four-pulse DONE clear at step 30,423,365. It was
not re-lit afterwards. This copy's noise draws make the latch asymmetric: its `u → v`
edge is +11.8 % and `v`'s threshold is −0.50 mV. Single background strays therefore keep
throwing it into a recurring fast excursion, about three times a second. In each excursion
the two members slide into near-coincident firing, with a 35–38-step period. The clear
arrived at the one phase of that excursion where a 4 × 0.75 train fails. Each pulse landed
0–5 steps after both members had fired, delayed them by roughly the inter-pulse spacing,
and so kept them just ahead of the next pulse: the latch was **entrained, not killed**.
Both request rails stayed live, the false rail vetoed `c4_sel`'s go chain, and the cell
never started again.

No repair pulse is expected in this situation. The ACT^d repair is driven by START,
re-lights a *dark* false rail, and is vetoed by a live one. `rate_robust` does not touch
this circuit; it matters here only because it gives the copy a different set of noise
draws.

This is the known kill-margin class (seed-108 copy 8, seed-110 copy 73), now against the
shipped 4 × 0.75 train. Reconstructing copy 28's own draws reproduces it with
`add_latch`, the real `add_kill_train` and RefSim: a kicked orbit survives
**5 / 1,500** clear phases at four taps, versus **0 / 1,500** with an added fifth tap.

**A fifth request-clear tap is rejected as the standalone fix.**
Four configured taps nominally emit **five inhibitor spikes**, and five taps emit
**six**: the final spike is residual charge, not noise. Copy 28's failing controller
emits only four. The earlier 0.15 % survival estimate is conditional on four
inhibitor arrivals; it is not the rate for every shipped four-tap controller.
The real five-tap train has a **−40-step reload margin at the slow corner**, so
request reload/repair timing needs qualification before any policy change.
No netlist default or `request_clear_pulses` wiring is changed here.

## Inputs and rebuild

All read-only, under `/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:
`mixB_s108_c100_rr_replay870.json`, `mixB_s108_c100_rr_replay870_copy28.npz` and
`reports/rr_copy28.md`. The record's options are: generic datapath, true guards v2,
`start_relight_hops=5`, `relight_repair_delay`, `copy_requires_rail`, request and kernel
clears 4 × 0.75, `rate_robust` version 2 and no `retry_clear`. It rebuilds
(`stall_diag.build_tick_pipeline`) to **30,643 neurons / 55,936 synapses**, which matches
the record. The dump window is 30,138,019–32,262,185 (steps of 0.1 ms). The last commit is
at 30,438,019.

| Neuron | Role | In dump |
|---:|---|---|
| 15701 / 15702 | `c4_sel.req.c2_selr0.u` / `.v` (false, "consumed") | yes |
| 15703 / 15704 | `c4_sel.req.c2_selr1.u` / `.v` (true, "pending") | yes |
| 15706 | `c4_sel.req.c2_sel.received` | yes |
| 15709 | `c4_sel.req.c2_sel.k1.inh` (the clear train) | yes |
| 15868 / 15869 / 15870 | `c4_sel.relight.c2_sel.edge` / `.edge_inh` / `.veto` (ACT^d repair) | yes |
| 29297 / 15780 | `…r0.u.rate` / `…r1.u.rate` (`rate_robust` readouts) | yes |
| 15840 | `c4_sel.start.fd.d4` (START's delayed re-light) | **not captured** |

Wiring into the false latch (netlist, delay 18 steps on every edge):

| Edge | Nominal q | Copy 28 q |
|---|---:|---:|
| `u → v` (loop) | 3,621 | **4,050 (+11.8 %)** |
| `v → u` (loop) | 3,621 | 3,672 (+1.4 %) |
| `k1.inh → u` | −2,716 | −2,893 (+6.5 %) |
| `k1.inh → v` | −2,716 | −2,741 (+0.9 %) |
| `start.fd.d4 → u` (re-light) | 4,655 | 4,522 |
| `relight.edge → u` (repair) | 4,655 | 4,565 |

The thresholds are `u` −44.897 mV and `v` **−45.504 mV** (nominal −45.0); the biases are
−0.242 and +0.074 mV. There are no other inputs. Copy 28's draws can be reconstructed
exactly because `make_perturbed_sim` draws on the host from `np.random.default_rng(seed)`
in a fixed order: first the weights `(100, nnz)` in topology edge order, then V_th
`(100, n)`, then bias `(100, n)`. Only the stray stream is device-drawn.

## What the dump shows

### The latch's history in the window

The false `u` trains (gap ≤ 141 steps):

| Train | Lit by | Ended by |
|---|---|---|
| 30,148,856–30,198,089 | START (before the table) | clear 30,197,984 |
| 30,209,437–30,251,598 | START | clear 30,251,449 |
| 30,262,882–30,309,446 | START 30,262,573, +309 | clear 30,309,343 |
| 30,320,796–30,363,097 | START 30,320,486, +310 | clear 30,362,971 |
| **30,374,396–32,262,157** | START 30,374,085, +311 | **clear 30,423,365 fails** |

Every re-light came 309–311 steps after START, as the five-hop re-light is meant to. The
last train is continuous through the end of the dump: 45,474 `u` spikes, with no gap
longer than 104 steps. **The latch survived; it was not re-lit.** Nothing could have re-lit
it: START never fired again, and the repair edge 15868 has no spike in the window.

Every train settles to a **43-step period with `v` firing 18 steps after `u`**. It then
repeatedly leaves that orbit along the same path. The `u → v` offset runs 17, 16, 15, …, 2,
1; `v` passes `u`; the offset returns as 32, 29, 26, 24, 22, 20, 19, 18. Meanwhile the
period drops to 35 and comes back to 43. Excursion onsets come at 3.05, 3.79, 3.44, 3.07 and
**2.99 per second** in the five trains. Over the last train, the `u` intervals are
distributed as follows (steps: count):

> 34: 6, 35: 2,320, 36: 2,143, 37: 1,175, 38: 1,672, 39: 1,136, 40: 1,702, 41: 2,137,
> 42: 5,392, 43: 27,233, 44: 528

### The failed clear against the four successful ones

The clear train's `k1.inh` spikes were at 30,423,456, 502, 551 and 600, which is +91, +137,
+186 and +235 after `received`. The spacing is 46/49/49 steps, the same in all five clears
of the window. Each pulse reaches both members 18 steps later. The table counts steps
relative to `received` R = 30,423,365.

| | `u` spikes | `v` spikes | pulse arrivals |
|---|---|---|---|
| before the clear | −36, +0, +36, +72 (intervals 39, 38, 38, 37 → 36) | −30, +6, +41, +76 | |
| during the train | +107, +150, +204 | +112, +152, +199, +253 | **+109, +155, +204, +253** |
| after the train | +308, +400, then intervals 92, 62, 53, 49, 47, 45 … back to 43 | +361 … | |

- Before the clear, the latch was in an excursion, at offsets 6, 6, 5, 4 with a 35–37-step
  period.
- **Pulse 1** arrives 2 steps after `u` and 3 steps before `v`.
- **Pulse 2** arrives 5 and 3 steps after the pair.
- **Pulse 3** arrives 5 steps after `v`, on `u`'s spike step.
- **Pulse 4** arrives on `v`'s spike step.

Every pulse lands in the members' post-spike refractory period (n_ref = 22 steps), where
it can only delay them. The delays grow from about 7 to about 20 steps, so the pair's
period stretches to 43–54 steps. That stretch locks the pair to the train's 46–49-step
spacing, and the pair keeps firing just ahead of each next pulse. After the fourth pulse
the latch has slowed to intervals of about 100 steps, then recovers.

The healthy clears look different. At 30,309,343, for example, the latch sat on the
43-step orbit (`v` −6, `u` +19, `v` +36, `u` +61, `v` +78, `u` +103, `v` +126). Pulse 1
arrived at +111, 8 steps after `u`. It delayed `v`, and pulses 1 and 2 together stopped `u`
before `v`'s spike could re-fire it. The other healthy clears end the same way, within the
second pulse. The 30,251,449 clear also met the latch in an excursion, at offset ≈ 20 on
the return leg, and killed it on the third pulse.

### Why no repair fired, and what held

- **The repair cannot fire here, by design.** The repair (`kernel.py`, §10.5) starts from
  START's ACT^d chain. It re-*lights* a dark false rail and is vetoed by a live one
  (`relight.veto` ← false `u` and true `u`, −1,810 q onto the edge). Its delayed tap fired
  on schedule after each START: `edge_inh` 15869 at START + 1,140, + 1,131 and + 1,130
  steps (113 ms). The edge stayed silent each time because the false rail had re-lit, which
  is correct behaviour. After the failed clear no START occurred, so no tap occurred.
  Firing it would not have helped either, since it ignites the same rail that is stuck.
- **The fault stayed a fail-stop, not a wrong value.** The go chain's vetoes
  (`go.g0.pb.veto`, `go.g1.pa.veto`, `go.g1.pb.veto`) read false `u` directly, at 3,621 q.
  The producer `c2_sel` cannot overwrite the unconsumed value either. Its commit gate
  `c2_sel.cg.g1.pa` *requires* the false readout 29297, but it is *vetoed* by
  `c4_sel.req.c2_selr1.u` (pending), and that rail is live. No commit-gate edge fired.
  Accounting: 0 wrong, 0 duplicates, 1,005 outputs unfinished.

### `rate_robust`'s part: none in the mechanism

`rate_robust` leaves the request latch, its k1 train, the START re-light and the repair
relay unchanged; `kernel.py` pass 2 does not consult it. The netlist has **no edge from
any `.rate` readout into any request-latch member**. The readouts are feed-forward taps that
inherit k1's inhibition:

- 29297 drives only `c2_sel.cg.g1.pa.require0`, at 155 q.
- 15780 drives `c4_sel`'s go `require1` and `pb.edge`, also at 155 q.

The vetoes that block START, and the veto that keeps the repair away from a live rail, read
the latch member at full pulse drive. The guard timing that matters here is therefore the
default build's. The repair tap's +113 ms matches the `relight_repair_delay` design.
`rate_robust` matters only as a different netlist. Edge indices shift, so copy 28 of the
default build is a different set of draws, and that copy completed.

## Reproduction from copy 28's own draws (RefSim, real primitive)

`tests/test_request_clear_entrainment.py` builds `add_latch` and `add_kill_train`,
then applies the reconstructed draws to both the latch and its clear controller.
It uses the full-precision thresholds/biases, all controller edges (including
the source edge detector), the loop weights above and the captured kill fan-out.
The added `h4` and its two edges have nominal draws, since that hypothetical
controller has no capture. Only the latch ignition, excursion kick and source
activation are external events; inhibitor arrivals are generated by the primitive.

Ignite `u` with 4,522 q at step 10, kick `v` with 300 q at 5,020, and activate
`received` with one ignition event at each step 5,000–6,499, one clear per node:

| controller | kick | inhibitor spikes per clear | survivors / phases |
|---|---|---:|---:|
| copy 28, 4 taps | none | 4 | 0 / 1,500 |
| copy 28, 4 taps | 300 q on `v` | 4 | **5 / 1,500** |
| copy 28 plus nominal fifth tap | 300 q on `v` | 6 | **0 / 1,500** |
| nominal, 4 taps; copy-28 latch | 300 q on `v` | 5 | 0 / 1,500 |
| nominal, 5 taps; copy-28 latch | 300 q on `v` | 6 | 0 / 1,500 |

The survivors start at source-event steps 5,507 and 5,538–5,541; both members
fire within six steps of one another at the first arrival. They remain live
150 ms after activation. This reproduces the entrainment mechanism, not the exact
CUDA stray stream or absolute stall step. Source activation and primitive timing
also differ slightly from the recorded arrivals (the four-spike sweep has
47/49/49-step spacing, versus the capture's 46/49/49).

A smaller real-primitive stray probe (`_strayed_survival(taps, copies=4000)`,
seed 108, batches of 1,000, uniformly random source events at 3,000–4,999,
5 Hz × 150 q strays on both latch members through step 6,800) gives **4 / 4,000**
survivors at four taps and **0 / 4,000** at five. Survival is still measured
150 ms after source activation. This is conditional on copy 28's controller
draws and a nominal added tap; zero survivors in this smaller sample is weaker
evidence than the earlier 60,000 prescribed five-arrival trials. The new
40,000-clear-per-setting slow regression was added but not run for this update.

Noise-free `add_kill_train` and `add_reset` both produce five spikes for four
taps and six for five taps. For the nominal kill controller, arrivals relative
to its first arrival are **0/45/89/136/215** and **0/45/89/136/185/252**.
The last spike is a longer residual-charge tail. In the reviewed captures,
71–82 % of four-tap bursts contain five spikes; rr copy 28 has **99 / 140**
(71 %). Six of its 25 request inhibitors always emit four, including the failing
one; sixteen always emit five. A configured tap count is not an arrival count.

### Earlier prescribed-arrival probes (conditional evidence)

These earlier scratch probes are not committed. Each one is one `add_latch` with the table's
copy-28 weights, thresholds and biases. It is ignited by 4,522 q on `u`. The clear is
injected as `k1.inh` arrivals at the recorded offsets 0/46/95/144 (+49 per extra pulse),
with −2,893 q on `u` and −2,741 q on `v`. Their clear statistics apply to these
prescribed arrivals only. They must not be interpreted as a census of real
four-/five-tap controllers, which generally emit an extra spike.

1. **Free run, no strays.** The latch has a 44-step period with `v` 19 steps after `u`. Its
   ignition transient (intervals 73, 56, 50, 48, 46, 45, 45, 44) matches the dump's train
   starts step for step. Without noise, no excursion occurs.
2. **Excursion trigger.** A single 150 q kick never starts an excursion, at any of 44
   phases, on either member (175 q: 0 / 44). Kicks on `v` do start it: 200 q at 7 / 44
   phases, 225 q at 22, 250 q at 31, 300 q at 40. Two 150 q strays on `v` 5–40 steps apart
   start it at 41–44 / 44 phases. Kicks on `u` never start it, up to 300 q. Once started,
   the excursion follows the dump's trajectory exactly: offsets 17, 16, …, 1, then 32, 29,
   26, 24, 22, 20, 19.
3. **With mix-B strays** (5 Hz × 150 q on both members, 40 runs × 4.8 s), the steady
   period becomes 43. Excursions start at **2.80/s** (dump: 2.99/s), and the interval
   histogram reproduces the dump's: 35: 2,221, 36: 2,052, 37: 1,144, 38: 1,586, 39: 1,084,
   40: 1,626, 41: 2,031, 42: 5,013, 43: 28,365, 44: 978. Strays near threshold are what turn
   single strays into triggers.
4. **The failing phase.** A 300 q kick on `v` is followed by the recorded 4 × 0.75 train,
   started at each of 1,500 consecutive steps.
   - On the steady orbit: 0 / 1,500 survive.
   - On the excursion orbit: **5 / 1,500 survive.** All five have a `u → v` offset of 3–5,
     a preceding interval of 35–36, and the first pulse arriving within 3 steps of a `u`
     spike (1–3 steps after it, or 0–2 steps before the next one).
   - Copy 28 at its failure: offset 5, interval 35, lag 2, which is inside that window.
5. **Monte Carlo with strays and a uniformly random clear time.** Each run is ignited, gets
   strays throughout, and is cleared at a random step in 3,000–5,000. It counts as a
   survivor if `u` still fires 150 ms later.

   | clear | survivors / clears |
   |---|---:|
   | 3 × 0.75 | 1,866 / 40,000 (4.7 %) |
   | **4 inhibitor arrivals × 0.75** | **59 / 40,000 (0.15 %)**; a second, independent run gives 28 / 20,000 |
   | 5 × 0.75 | **0 / 40,000** and 0 / 20,000 (95 % upper bound 5.0 × 10⁻⁵ per clear) |

   Conditional on four-arrival clears, the pair is cleared once per `c4_sel` transaction. Over the ≈535 clears before the
   stall, 0.15 % gives a 1 − (1 − 0.0015)^535 ≈ **55 %** chance of at least one survival.
   A 1,000-tick run gives ≈ 77 %. This is consistent with the captured four-spike
   controller, not an overall stall prediction for the shipped four-tap policy.

### How rare such a latch is

All 25 request false latches of each of the 100 copies were taken from this record's draws,
2,500 latches in total. Their noise-free periods have minimum 43, p1 44 and median 47.

- Only **6 latches** enter an excursion after a 300 q kick at any of 16 phases. They belong
  to copies 19, 24, 28, 59, 87 and 88. Copy 28's latch is the most susceptible: 15 / 16
  phases, against 1–13 for the others. Every one of the six has one member clearly favoured
  (an input edge +8 to +13 % and/or a threshold 0.4–0.7 mV low).
- At 200 strayed clears per latch, 4 × 0.75 left **0 / 500,000** survivors. At 20,000
  clears each, on the six susceptible latches plus the six fastest of the rest, **only copy
  28's latch survives** (28 / 20,000). Each of the other eleven gives 0 / 20,000.

**Census caveat:** these probes reconstruct the latch draws but prescribe exactly
four arrivals; they do not reconstruct each latch's own clear-controller draws.
Thus the 2,500-latch census identifies susceptibility under that train, not a
per-copy failure rate for the implemented primitive. The real-controller sweep
above confirms copy 28's mechanism; it does not repeat the full census.

Under the prescribed train this mechanism accounts for copy 28 and no other copy. Copy 24's
latch, `c11_sub.req.input`, is susceptible to the excursion but is killed in every trial,
so copy 24's stall needs its own explanation.

## Relation to the kill-margin class

The class shares its shape with the earlier cases. A request's false rail outlives its DONE
clear, both rails stay live, the false rail vetoes go, and no repair is involved. The cases
differ as follows.

| | Seed-108 copy 8 (base, 2026-09-20) | Seed-110 copy 73 | **Stage D rr copy 28** |
|---|---|---|---|
| clear | 3 × 0.75 | 3 × 0.75 | **4 × 0.75** |
| latch period | 33–41, wandering; 34 at the failure | 41 | **43 steady**; 35–38 in stray-triggered excursions |
| failure | slipped between pulses | survived the first clear | **entrained**: every pulse 0–5 steps after both members fired |

The earlier four-arrival census killed every latch of this realization except
copy 28's, and killed copy 28's in all but 0.15 % of its prescribed clears.
That evidence does not measure the output distribution of a four-tap controller.
The residue is not a statically fast loop, which `test_kill_margin.py` and the 10,000-copy
survey cover. It is a **dynamic** fast state: an asymmetric latch whose two members drift
into near-coincidence, at a period (35–38) at the edge of what four pulses kill. The
surveys missed it because they draw a symmetric loop scale and threshold offset, and their
kills start on the latch's steady orbit.

The START re-light delay and the delayed ACT^d repair are not involved. The re-light worked
on all five STARTs, and the repair was correctly vetoed by a live false rail. The true
guards turned the fault into a fail-stop, as intended.

## Fifth-tap candidate and reload constraint

**Do not dispatch a five-tap default from the arrival-only probes.**
`request_clear_pulses=5` in `lib/kernel.py build_pipeline` remains an isolated
candidate for the k1 train of each request pair in pass 2. The option exists and
is recorded in `build_options`. Its six-spike nominal tail improves clear margin
but consumes the slow-corner reload margin; any shipping decision must qualify
the START re-light and delayed repair together. The other kernel kill trains
(`kernel_kill_pulses=4`) and machine clears are outside this candidate's scope.

- **Cost:** one relay neuron and two synapses per request pair. On `tick2.c` that is +25
  neurons / +50 synapses: 29,375 / 52,808 → 29,400 / 52,858 by default, and
  30,643 / 55,936 → 30,668 / 55,986 with `rate_robust`. The added edges carry ordinary
  pulse drive, and the existing `k1.inh` mirrors cover the readouts.
- **Latency:** nothing on the handshake path waits for the end of the clear, but
  reload can miss and require the delayed repair. The nominal real train's last
  arrival is 252 steps after its first at five taps, versus 215 at four taps.
  Anchoring its first arrival at the capture's R + 109 gives R + 361 / R + 324;
  the earlier R + 302 / R + 253 values omitted the nominal tail.
- **Ordering, the hazard behind the 2026-09-20 wrong values:** the false rail's earliest
  re-light is START + 5 hops. In this copy, receipt → START was at least 686 steps (over
  129 receipts; median 1,834). Ignition reaches `u` about 25 steps before its first spike,
  at START + ~285, so ≥ ~971 steps after receipt (862 after the captured first
  clear arrival at R + 109). The nominal sixth arrival leaves only ~610 steps
  before that ignition. The constraint is residual hyperpolarisation even when
  no inhibitor spike arrives near the fresh word.
- **Reload, re-measured with the real primitive:** `add_kill_train` with nominal
  controller draws, 0.75 strength and 4/5 taps (5/6 actual spikes), followed by
  a START-like 4,655 q event on `u`. The latch uses copy 28's reconstructed draws
  or the stated symmetric loop/threshold corner. Initial ignition is 4,522 q at
  step 10; source activation is at 2,000 plus phases 0, 4, …, 40 (11 phases).
  Reload offsets are swept in 50-step increments after the last **actual**
  arrival. Success requires a sustained train 150 ms later, not a transient
  spike. These are conservative grid bounds, not exact recovery thresholds.

  | latch | 4 taps: after last / first arrival | 5 taps: after last / first arrival | margin vs 862 (4 / 5 taps) |
  |---|---:|---:|---:|
  | copy 28 | 250 / 465 | 300 / 552 | +397 / +310 |
  | nominal | 350 / 565 | 400 / 652 | +297 / +210 |
  | loop −8 % / V_th +0.4 mV | 500 / 715 | 550 / 802 | +147 / +60 |
  | loop −12 % / V_th +0.6 mV | 600 / 815 | 650 / 902 | +47 / **−40** |

  At the slow corner the extra tap adds 87 steps from the first arrival, not
  just the tap spacing. The earlier **+20-step** five-pulse margin is withdrawn:
  six nominal spikes need 902 steps against 862 available. Reload at 650 steps
  after the true last arrival succeeds at all phases, but that is too late for
  the earliest START. A failed early re-ignition is the case the delayed ACT^d
  repair is intended to handle (`tests/test_relight_repair.py`); this isolated
  test does not prove that the kernel repair resolves it under noise.
- **Interactions:** with true guards, a dark pair still blocks rather than passes. The
  repair's live-false veto is unchanged. `copy_requires_rail` is unaffected because it is
  not a request path. `rate_robust` readouts and `require` neurons already mirror every
  `k1.inh` spike, so every emitted spike clears them too. A rebuilt netlist is a new noise
  realization, so the pinned tick SHA-256 regressions in `tests/test_build_options.py` move
  and campaigns must be compared by per-seed totals.
- **Rejected alternatives:**
  - Earlier prescribed-arrival probes found `retry_clear` cures it in isolation
    (0 / 1,500 on the kicked orbit). It costs more
    per pair: a 12-hop delay chain, a gate and a 3 × 1.5 train. It also has two campaign
    records of turning stalls into wrong values.
  - 4 × 1.0, or the four pulses at 24-step spacing, also give 0 / 1,500. Both change the
    clear's charge or timing more than one extra pulse does, and stronger trains have
    failed in the kernel before.
  - Making the latch symmetric is not a fix at this depth: the asymmetry is the noise model.

**Validation required before shipping:**

1. Qualify START re-light and ACT^d repair at the slow corner with the six-spike
   tail before selecting the candidate. Then run Stage D on Juno with
   `request_clear_pulses=5` via explicit experimental build options. `stage_d`
   has no CLI flag for it; exposing one is separate work, not a reason to change
   the default for a diagnostic campaign:
   `python -m drosophilos.bench.stage_d --ticks 1000 --seed {108,109,110} --backend torch-fast --device cuda --copies 100 --mix B [--rate-robust]`,
   for both builds.
2. The machine and multiplier suites (`test_machine.py`, `test_mul_diag.py`), and the
   relight-repair and true-guard suites, run against any proposed policy change.

## Regression test

Implemented in `tests/test_request_clear_entrainment.py`, in RefSim with no kernel:

- **Fast:** reconstruct the full copy-28 clear controller and latch, drive its
  source, and sweep 1,500 phases on steady/kicked orbits. Four taps leave survivors
  only on the kicked orbit, with pair offset ≤ 6; an added fifth tap leaves none.
  Also sweep the nominal controller against the kicked latch (zero survivors at
  either setting) and check both real primitives' extra nominal output spike.
- **Slow:** 40,000 randomly timed clears per tap setting with 5 Hz × 150 q strays
  on both latch members and the real controller. Assert four taps leave survivors
  and five leave none. This is a conditional copy-28 probe, not a controller census;
  it is excluded by `-m 'not slow'`. The earlier 59-survivor count belongs to the
  prescribed-arrival probe and is not pinned as a real-primitive measurement.
- **Reload:** sweep the real nominal controller at all four latch corners and
  11 phases, including the residual-charge tail, and pin the table above.
  Five taps reload at last-arrival +650 at the slow corner, while their grid bound
  first-arrival +902 exceeds the earliest START budget of 862. A separate probe
  at last-arrival +610 (first-arrival +862) confirms missed early reloads.

These regressions reproduce entrainment and the reload constraint from static draws.
They do not replay the device stray stream, survey the full noisy kernel, or establish
that the delayed repair makes a fifth-tap policy safe. No new campaign is claimed here.

## Compact request clear (2026-10-04)

**Choice:** opt-in `robust_request_clear=True`, implemented by
`control.add_request_clear`. Keep the real four-tap `add_kill_train`, shorten only
the three links into `h1/h2/h3` from 18 to **0 synaptic-delay steps**, and use
**1.1 × loop** inhibition on the false request latch. Zero-delay delivery still
integrates on the following step in RefSim; these are ordinary supported
synapses, not injected inhibition or simulator state edits. All other controller
edges retain their original delay and drive. Both latch members receive the same
train, and the existing inhibition mirrors cover rate readouts and guard inputs.

The compact controller emits **four actual spikes**, with nominal arrival
offsets **0/43/81/138**; copy 28's reconstructed controller emits four at
**0/47/88/151**. Its excitation overlaps sooner, avoiding the ordinary
controller's late fifth spike. Stronger inhibition then stops fast loops without
paying for the ordinary train's late charge. This is a measured bounded-corner
result, not a proof for arbitrarily fast or arbitrarily perturbed latches.

### Candidate evaluation with real primitives

`tests/test_robust_clear.py::test_candidate_evaluation_table` reproduces the
tables. All candidates use `add_latch`, `add_kill_train`, the actual edge detector,
and RefSim. The candidates are concrete circuits:

- **Phase:** retain 0.75 strength; set successive tap-link delays to 0/36/18
  steps. Nominal inhibitor offsets become 0/57/105/147 (four spikes).
- **Verify:** retain the ordinary train, then use a 12-hop delay from `received`,
  a coincidence neuron taking 0.5 single-pulse need from that tap and 0.35
  rate need from **each** latch member, and a real one-tap 0.75 clear. The
  detector fires on the failing orbit: at source event 5,540 its delayed tap,
  detector and extra inhibitor fire at 6,200 / 6,258 / 6,353. The survivor
  withstands that extra spike too. Healthy nominal clears produce no extra
  spike. This rejects this single-extra-pulse detector, not every possible
  closed-loop design; the older multi-pulse `retry_clear` remains a separate,
  previously unsafe experiment.
- **Stagger:** ordinary train and strength; delay inhibition of `v` by another
  22 steps (one refractory period), leaving `u` unchanged.
- **Stronger:** ordinary four-tap timing, 1.0 strength; five nominal spikes.
- **Compact:** the selected zero-delay tap links and 1.1 strength, four spikes.

For the survival screen, apply copy 28's reconstructed controller draws and its
fan-out multipliers to the candidate's strength. Ignite at step 10, kick at
5,020, and sweep all 1,500 source-event steps 5,000–6,499. The mirror corner
exchanges the latch's loop weights, thresholds and biases, and kicks `u`
instead of `v`; ignition and controller draws are otherwise unchanged. The
symmetric fast corners are +20% / −0.8 mV (~38-step steady period) and
+30% / −1.2 mV (~33 steps), including the extra-kick excursion. Slow corners
are those in the reviewed reload table. Entries are **survivors / 1,500**:

| candidate | copy 28 | mirrored 28 | nominal | −8% / +0.4 | −12% / +0.6 | +20% / −0.8 | +30% / −1.2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary 4 × 0.75 | 5 | 1 | 0 | 0 | 0 | 276 | 1,500 |
| phase | 0 | 0 | 0 | 0 | 0 | 432 | 1,500 |
| verify | 5 | 1 | 0 | 0 | 0 | 276 | 1,500 |
| stagger | 0 | 0 | 0 | 0 | 0 | 468 | 1,500 |
| stronger 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **compact 1.1** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

Reload uses **nominal controller draws**, all actual inhibitor spikes, a
4,655-q ignition, 11 phases, and 50-step offsets after the last arrival, as in
the reviewed table. No-reload controls must go dark at all 11 phases before a
recovery threshold is credited. A successful reload must sustain firing 150 ms
later, at every phase and every later grid offset. Margins below are
**862 − (last arrival from first + recovery offset)** in steps; `—` means the
nominal controller did not clear the no-reload control, so survival cannot be
credited as re-ignition.

| latch | ordinary | phase | verify | stagger | stronger 1.0 | compact 1.1 |
|---|---:|---:|---:|---:|---:|---:|
| copy 28 | +397 | +465 | +397 | +375 | +247 | **+374** |
| mirrored 28 | +297 | +365 | +297 | +225 | +197 | **+274** |
| nominal | +297 | +365 | +297 | +225 | +197 | **+274** |
| loop −8% / V_th +0.4 mV | +147 | +215 | +147 | +125 | +47 | **+124** |
| loop −12% / V_th +0.6 mV | +47 | +115 | +47 | +25 | **−53** | **+24** |
| loop +20% / V_th −0.8 mV | +497 | — | +497 | — | +347 | **+424** |
| loop +30% / V_th −1.2 mV | — | — | — | — | +447 | **+524** |

The compact clear's all-phase recovery bounds after the last / first arrival
are respectively **350/488, 450/588, 450/588, 600/738, 700/838, 300/438,
200/338** steps in that table's order. The slow corner also reloads at the
actual earliest budget, first arrival +862, at every phase. With **copy 28's
controller and fan-out draws** as well as the slow latch corner, the bound is
700 after last / 851 after first: still **+11 steps**, including its real
four-spike train. These are conservative grid bounds, not exact thresholds.

### Stray qualification and why 1.1

The selected real primitive produced **0 / 40,000 survivors at each of the seven
corners above: 0 / 280,000 with copy 28's controller**. Additional nominal-controller
runs gave **0 / 40,000** each for the copy-28 and fast-30% latches, for
**0 / 360,000 total**. These runs reuse the reviewed RNG protocol:
seed 108, batches of 1,000, random clear events at steps 3,000–4,999, and
5 Hz × 150-q strays on both members through step 6,800. Either member firing
150 ms after its clear counts as survival. The same random schedules are used
at each corner; these are nine fixed latch/controller settings, not 360,000
independently drawn controllers. The per-setting zero-failure 95% upper bound
is about 7.5 × 10⁻⁵ per clear.

Phase sweeps alone were insufficient: a compact 1.0 train with 4-step tap
links passed all 1,500 phases but left **206 / 4,000** strayed fast-30% latches
alive. Shorter links and 1.1 strength were therefore qualified with the larger
stray sweep. The final nominal-controller phase sweeps cover the same seven
corners as well.

The earlier 4 × 1.5 / 3 × 1.5 changes increased prolonged inhibition in other
control domains and lost reloads. Here the control machine is untouched, and
the selected train's total nominal inhibitory charge per member is **4.4 loop**
(four actual spikes × 1.1), versus **3.75 loop** for the ordinary five-spike
0.75 train, or **5 loop** for the rejected ordinary 1.0 train. The compact
train finishes 77 steps earlier than the ordinary nominal train. The measured
slow-corner reload stays inside the budget; the ordinary 1.0 train and the
ordinary fifth-tap proposal both fail it. This addresses the measured reload
hazard; it does not establish safety of a noisy full Stage D campaign.

### Build option, costs and validation

`build_pipeline(..., robust_request_clear=False)` records the effective flag in
`Pipeline.build_options`. `stage_d` and `kernel_campaign` expose
`--robust-request-clear`; records, `stall_diag` rebuilds and Stage D `--recheck`
honour it, including old records' missing-option fallback to false. The option
requires request-priority storage, four configured request taps, base kill
strength 0.75, and `retry_clear=False`; unsupported combinations raise rather
than silently building an unqualified circuit. It changes only DONE-side
request `k1` edges. START clears, re-light delays, repair, the other kernel
trains, and machine trains retain their existing circuits.

- **Cost per request and per kernel:** **+0 neurons, +0 synapses**. There are
  three shorter tap-link delays and stronger inhibitor fan-out weights per
  request (including existing mirrors). Tick remains **29,375 / 52,808**,
  or **30,643 / 55,936** with `rate_robust`.
  FastSim now has two delay groups (0 and 18), adding one sparse delivery
  product per step; its campaign wall-time cost has not been measured.
- **Clear-to-READY path:** **0 added hops / 0 added delay**; nothing waits for
  the train to finish. The nominal first-to-last span falls 215 → 138 steps
  (−7.7 ms); copy 28's span changes 145 → 151 (+0.6 ms). The slow-corner
  conservative clear-to-reload bound changes 815 → 838 steps (+2.3 ms),
  retaining the +24-step nominal margin.
- **Fast tests:** zero survivors for kicked phase sweeps; real nominal and
  reconstructed pulse counts; positive reload margins, including the slow
  corner with copy 28's controller; a 2-bit ADD kernel streams 0/1/3 → 1/2/0
  in RefSim with both rate-reader settings; CLI/record/rebuild/recheck coverage;
  and unchanged pre-existing ordered-netlist SHA-256 assertions, including
  both machine sizes. The original entrainment regressions remain intact.
- **Scope:** no CUDA stray-stream replay or new full-kernel noise campaign is
  claimed. This remains opt-in pending campaign qualification.

Validation completed: the requested three-file `pytest -m 'not slow'` suite,
the expanded `test_robust_clear.py` fast suite, all nine 40,000-clear settings,
and the candidate table. Additional CPU probes at the copy-28 failing phase
and the fast-30% corner gave identical full traces in RefSim, TorchSim and
FastSim, including the zero-delay links.

The literal requested `uv` invocation initially failed because the sandbox
denied its default cache. It passed after setting `UV_CACHE_DIR` and pytest's
`--basetemp` to the writable task temporary directory, `UV_PROJECT_ENVIRONMENT`
to the existing repository environment, `UV_NO_SYNC=1`, and `PYTHONPATH=.` to
import this worktree. The test command itself remained:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_robust_clear.py tests/test_request_clear_entrainment.py tests/test_build_options.py -m 'not slow'
```
