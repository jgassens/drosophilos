# Stable storage with rate-conditioned readers: measured frontier

**None of the evaluated combinations meets all six qualification requirements.**
The 0.2-loop autapse is available for Stage D measurement as the recorded,
explicitly EXPERIMENTAL `build_pipeline(..., experimental_autapse=True)` opt-in.
It **requires `rate_robust=True`** and remains **unqualified on the 44-step rate
floor and reload margin**. Defaults retain their ordered netlist fingerprints;
the control machine is unchanged.

## Campaign result (2026-10-09)

The option fails two isolated proxy requirements — the 44-step rate floor and reload margin — and meets the Stage D exit on generic seeds 108, 109 and 111 and specialized-datapath seed 108, but not generic seed 110. Campaign D, `tick2.c` mix B, 1,000 ticks × 100 copies, `--rate-robust --experimental-autapse` (`f71538c`, merged `28caa66`), was rechecked by the integrator on generic seed 108 (Juno 443026) and seed 109 (Juno 443063): each was 100/100 matched and 100,000/100,000 ticks, with 0 wrong, faults, timeouts, stalls, refusals and retries. This is a Stage D result, not qualification or a default change.

The build has 30,643 neurons and +6,430 synapses versus rate readers; the autapse is applied to all 3,215 kernel latches. It caps the reviewed four-pulse-clear failure classes at the latch (stage Z0, request and master rail; completion capture Juno 441837), but 40,000-trial surveys cover only those request/master/stage cases. Additional commit `63c41f8`: generic seed 110 (Juno 449070) **exit not met** — 99 matched, copy 82 stalled after 352 completed ticks, 1 run-level fault, 99,352 matched ticks, 0 wrong, 648 missing, 0 refusals/retries; generic seed 111 (Juno 449071) **exit met** — 100/100, 0 wrong/faults/timeouts/stalls; specialized datapath seed 108 (Juno 449072) **exit met** — 100/100, 0 wrong/faults/timeouts/stalls, 26,683 neurons. Total: 4 of 5 100-copy campaigns met exit; 1 stalled copy in 500 (0.2%); 0 wrong values in 499,352 matched ticks. The exact one-sided 95% Clopper-Pearson upper bound from 1/500 is 0.009452282208 (0.9452282208%); one-sided Fisher exact 1/500 versus pooled no-autapse seed-108 baselines 15/300 is p=0.00000344134. The observed stall rate is about 25× lower, not zero; a spike-dump replay of generic seed-110 copy 82 is queued. Nominal: `tick2.c`, 1 copy, mix none, seed 1, 1,000 ticks, `--rate-robust --experimental-autapse`, Juno 448740, commit `474a8ee`: exit met; 1,000/1,000 ticks matched, 0 wrong, faults, timeouts, missing, refusals or retries; 30,643 neurons, 5,736.6 s neural, 3 h 10 min wall on one H200. The configuration's mix-B exit is met on generic seeds 108, 109 and 111 and specialized seed 108, not generic seed 110. The option remains experimental and non-default, and the isolated 44-step rate floor and reload margin remain unqualified. Keep the transform restricted to the reviewed construction order — after all consumers/controllers, including compact resets, and before any later reader installation — with version-gated rebuild/recheck. Toy tick results remain labeled as such.

Both `stage_d` and `kernel_campaign` expose `--experimental-autapse` with
`--rate-robust`. Campaign records retain the choice in build options,
`stall_diag` rebuilds it, and Stage D `--recheck` honours saved options.
Build options also record `experimental_autapse_version = 1`: gain 0.2,
delay zero. Diagnostic rebuild and Stage D recheck reject a version differing
from the current circuit. Records without the version key mean version 1,
because commits f71538c..63c41f8 built exactly that circuit; missing boolean
flags still default to false.

Autapse **alone** still rejects memories/RAM, phases/pacing, multiple input
streams and multipliers. The complete configuration F has the narrower,
per-shape qualification described below; this does not qualify those shapes
with autapse alone.
Both generic and specialized `tick2.c` datapaths remain allowed; specialized
Stage D seed 108 (Juno 449072) met exit with 100/100 copies and 26,683 neurons.
Same-seed autapse and legacy campaigns are
not paired copy for copy: the extra edges change the weight-draw shape and
shift subsequent weight, threshold and bias draws. Compare aggregate rates.

All eight combinations of
`zero_once`, `robust_request_clear` and `experimental_register_reset` are
allowed, subject to those options' existing restrictions. Each combination
builds and computes 0, 3, 1 on the 2-bit ADD → XOR kernel; two-copy full mix-B
checks cover autapse alone and all three options together. The rejected
`robust_register_reset` and `verified_register_reset` policies remain rejected.
These finite checks authorize measurement, not qualification.

The earlier claim that the autapse fails with `rate_robust` readers was a **probe
artifact**. `_modify` passed local inhibitory feedback to `net.synapse`, which
also copied it at gain 16 into existing readers and into required-rail qualifiers.
A latch spike consequently suppressed its own consumers. The fixed transform
builds all consumers first, suppresses mirroring only while adding local feedback,
and preserves genuine reset/kill mirrors. A regression checks pre-existing resets
and resets added after the transform. The kernel builder applies the same
transform after all consumers and controllers are wired, including compact
register resets. No reader is installed afterward.

With that correction, autapse + readers computes **0, 3, 1 with zero faults** in
the multi-cell kernel, including with the other options enabled. It remains
unqualified because storage admits a 36-step independent-corner interval and
reload margins are worse than legacy. High-gain storage fixes the rate floor,
but ordinary clears leave live storage, the slow corner still misses gate
inputs, and the kernel stops after its first output.

## The six requirements

All six are qualification gates. The experimental campaign opt-in does not claim
to pass them:

1. **Rate floor:** settled storage intervals are at least 44 steps, including
   independent three-sigma weight, threshold and bias corners, both ignition
   schedules and the captured-style kick. A good random sample cannot override
   a deterministic corner failure.
2. **Holding:** an ignited train does not drop out during the exposure. Failure
   to establish a train is reported separately from loss after establishment;
   silent copies are never omitted from the success denominator.
3. **Ordinary clear:** the actual four-tap, 0.75-loop controller leaves zero
   storage or reader survivors in at least 40,000 trials **per scenario**,
   including the five captured entrained primitives, fast/slow three-sigma
   storage and independent mix B. Storage must be live before clear.
4. **Reload/re-ignition:** recovery margins are no worse than legacy, asserted
   against identical legacy controls. A train that never cleared cannot count
   as successful re-ignition. Passing the isolated comparison would still
   require qualification of actual stage/COPY/request paths.
5. **Rate-mode gates:** no false faults and no missed legitimate inputs, with
   legacy controls, dark/one-live/two-live inputs, noisy exposures and corner
   probes. Readers must be exercised on the candidate storage itself.
6. **Kernel correctness:** a multi-cell kernel produces all expected outputs,
   with zero faults, timeouts and bad decodes, including the combination with
   `zero_once`, `robust_request_clear` and `experimental_register_reset`.

Finite stochastic probes do not prove indefinite holding or absolute bounds
under unbounded Gaussian tails. Failures below are concrete refutations; passes
are limited to the stated exposures and conditions.

| Requirement, with readers | Legacy | Autapse | High gain | More rate margin |
|---|---|---|---|---|
| 1. Storage interval ≥44 | fails: 32 | fails: 36 | measured floor 46 | measured floor 54 |
| 2. Holding | no observed dropout | no observed dropout | no observed dropout | no observed dropout |
| 3. Ordinary clear | fails captures: 51 / 1220 / 10 / 39 / 80; mix B: 0 | zero survivors in all eight surveys | fails | fails |
| 4. Recovery versus legacy | comparison baseline | worse in all eight comparisons | undefined after failed clears in seven | undefined after failed clears in seven |
| 5. Gate inputs | passes tested cases | passes tested cases | misses slow-corner conjunctions | misses slow-corner conjunctions |
| 6. Multi-cell kernel | correct with options off/on | correct with options off/on | only first output | only first output |

Clear counts are survivors per 40,000 trials, ordered request28, master74,
master17, master24, stage18. Autapse improves clears over legacy; its remaining
qualification failures are the absolute rate floor and lost reload margin.

Campaign E (`831016a`) used `--rate-robust --experimental-autapse --zero-once --robust-request-clear --experimental-register-reset` on `tick2.c`, mix B, 1,000 ticks × 100 copies. Juno H100 seed 110 (449853; 30,487 neurons) and seed 112 (449854) each met exit: 100/100, 0 faults/timeouts/wrong/missing, not truncated; 2/2 campaigns and 0 stalled/200 copies. Campaign F (`c2464d7`, merged `da15590`) used `--rate-robust --experimental-autapse --ordered-xor-oneshot`: Juno H200 seed 110 (451082; 31,363 neurons) and seed 113 (451083) each met exit, 100/100, 0 faults/timeouts/wrong/missing, not truncated; 2/2 campaigns and 0 stalled/200 copies. F's exact one-sided 95% binomial upper bound is 1.4867039% (0/200); E+F pooled is 0.7461356% (0/400), though they are different builds. Fisher exact F 0/200 versus pooled no-autapse 15/300, one-sided, is p=0.000407242. Nominal: `tick2.c`, 1 copy, mix none, seed 1, 1,000 ticks, `--rate-robust --experimental-autapse --ordered-xor-oneshot`, Juno H100 451587, commit `ded224b`: exit met; 1,000/1,000 ticks matched, 0 wrong, faults, timeouts, missing, refusals or retries; 31,363 neurons, 4 h 23 min wall on one H100. Stage D exit is met with configuration F: nominal plus 100-copy mix B on seeds 110 and 113. The autapse-only residual (seed 110 copy 82; Juno 449070/449820/450661) was a real R6 double rail from an ordered-XOR sum relay escaping its veto on a second carry pulse; `ordered_xor_oneshot` fixes the primitive (0 double rails/120,000 trials; Juno 451099 slow tests: 5 passed). F is the leanest configuration meeting exit on every seed tried: `rate_robust` plus opt-in `experimental_autapse` and `ordered_xor_oneshot`; it remains experimental/non-default, has two seeds per configuration, is toy tick only, and the default-build decision is the user's.

## Circuits, methods and search coverage

Probes use real `add_latch`, `add_reset`, `add_kill_train`, AND/OR/majority, veto
and required-rail primitives, integrated by unchanged float64 RefSim. There is
no prescribed latch train, synthetic inhibitory arrival sequence, rate clipping,
changed refractory period or additional recovery delay.

The original loop is 3,621 q. Delays are 0.1 ms steps. Ignition and external
clear strengths retain their legacy values. The best clearable autapse and the
two fastest independent-corner passers at the lowest passing loop gain are
followed through all six requirements:

| Candidate | Loop gain | u→v / v→u delay | Local feedback on each member |
|---|---:|---:|---|
| Legacy control | 1 | 18 / 18 | none |
| Autapse | 1 | 18 / 18 | −0.2 loop, delay 0 |
| High gain (`HIGH_GAIN`) | 8 | 22 / 22 | −6 loop, delay 22 |
| More rate margin (`HIGH_MARGIN`) | 8 | 26 / 26 | −6 loop, delay 22 |

The latter two minimize loop gain among corner passers, then nominal period.
This is a reproducible selection criterion, not a claim of global optimality.

Mix B is independent `round(q * exp(N(0, .04)))` per edge, independent threshold
and tonic-bias sigma 0.2 mV per neuron, and Bernoulli 5 Hz × 150 q strays on every
neuron, including readers and dark rails. `MixProbe` samples this campaign law;
it does not replay the CUDA random bitstream. Adding edges changes random draw
assignment, so random legacy/candidate comparisons are not paired trajectories.

`independent_corners` enumerates both signs of every edge's 0.12 log-weight
offset and every neuron's 0.6 mV threshold and bias offsets. With a reader this
includes its input edge, threshold and bias: 1,024 legacy or 4,096 candidate
initial-state/corner cases. Single weak ignition at step 10 uses
`round(drive.ignite * exp(-.12))`; the other schedule adds ordinary ignitions at
11, 14 and 23. A 300-q kick at 2,520 challenges coincidence. Measured intervals
start at step 2,000 and include the kick transient.

The committed search covers **all 3,227 configurations**, not just the earlier
twenty-setting shortlist. It runs 5,000 steps at five correlated electrical
corners and both ignition schedules; screen intervals start at 3,000. A spike
in the final 200 steps is only the screen's liveness filter.

| Grid | Configurations | Live in all screen cases | Live and screen floor ≥44 |
|---|---:|---:|---:|
| Lower gain / asymmetric delays | 1,089 | 847 | 0 |
| Direct / shared delayed inhibition | 704 | 415 | 5 |
| Bias / loop / delay | 384 | 208 | 0 |
| High gain with inhibition | 600 | 242 | 39 |
| Refined low-gain autapses | 450 | 295 | 35 |

Every one of the **79** screen passers now has a committed independent-corner
follow-up. Every setting that meets the rate floor and establishes all initial
trains is also rejected by **actual reader-enabled ordinary clears** (32/32
survivors at fast three sigma each). Its nominal raw fault gate misses two live
inputs too, but that is not the reason to reject a combination with readers.
The other settings fail the independent rate floor or fail to establish hold.
Very slow circuits receive clear tests even if they exceed the 200-step
liveness window: a long continuous period must not alone be called dropout.
Full 40,000-trial surveys cover the selected two high-gain settings and autapse.
The finite search does not show that a qualifying circuit is impossible.

## Rate floor and holding

| With readers | Independent minimum | Non-ignition | Loss of established hold |
|---|---:|---:|---:|
| Legacy | 32 | 0 | 0 |
| Autapse | 36 | 0 | 0 |
| High gain | 46 | 0 | 0 |
| More rate margin | 54 | 0 | 0 |

These values are asserted. Readers are feed-forward and cannot repair storage
rate violations. The autapse's sub-44 failure is at independent extreme corners;
it passes the floor in the committed random mix-B exposures. Random surveys use
10,000 latches for 6,000 steps; long exposures use 1,000 reader-equipped latches
for 100,000 steps (10 s) plus extended independent storage corners.

Holding diagnostics define establishment as at least two spikes in [500, 1000),
after ignition and before the kick. They separately report non-establishment,
an established train's subsequent gap over 200 steps (including terminal
silence), and no spike in the final 200 steps. A late kick cannot erase a prior
gap. This operational check is stronger than looking only at the last window;
it is not an indefinite-holding proof.

For example, the earlier zero-delay representative has 512 independent cases:
98 never establish, 77 lose established hold, and 175 are dark at the end.
Its interval minimum is 34. The separate `silent == 173` statistic means no
measured intervals after step 2,000; it must not be called 173 holding losses.
The old unasserted percentile, interval-count and auxiliary-corner tables have
been removed. Probe output still includes those descriptive statistics.

## Ordinary clears and recovery

`clear_survey` uses the real four-tap, 0.75-loop controller; configured taps can
emit more than four spikes. Each trial includes four close ignitions, the kick,
random clear at steps 3,000–4,499, and strays everywhere. Storage and reader
survivors fire more than 1,500 steps after the trigger; a separate check requires
live storage in the 200 steps before it. A suppressed reader does not excuse
live storage behind it.

The five captures are request RR copy 28, master base copies 74/17, master RR
copy 24 and stage Z0 base copy 18. Storage/controller edges, thresholds and
biases retain their reconstructed seed-108, B=100 draws. Feedback edges receive
independent mix-B draws. Readers and their edges are rebuilt and sampled
independently, including master24, whose capture already had rate readers;
its captured reader parameters are not replayed.
Reader reset mirrors are genuine controller edges. New mirrors start at nominal
weights, including in request 28, whose fixture stores old captured weights
directly in its netlist. The captured old draws are reapplied only to old edges.
This is a conditional
mechanism probe, not a captured CUDA trajectory replay or stall-rate estimate.
Fast/slow corners perturb storage and its incoming clears; their controller
and reader parameters are otherwise nominal.

Reader-enabled surveys test legacy and all three candidates at **40,000 trials
in each of eight scenarios**, 320,000 trials per design. Autapse has zero storage
and reader survivors in every scenario. Both high-gain candidates leave
40,000/40,000 storage survivors at fast three sigma; legacy leaves 14/40,000,
despite clearing all 256 smoke trials. At slow three sigma, high gain clears
all trials, but the longer-delay candidate leaves 6/40,000 survivors.
Captured and mix-B scenarios refute both high-gain designs as well. Legacy
storage-survivor counts on the five captures and mix B are asserted as
51 / 1220 / 10 / 39 / 80 / 0. Exact matching reader-survivor counts are asserted
for legacy, autapse and the slow-corner high-gain cases; the other high-gain
cases assert storage and reader survivors are nonzero, without claiming equal
counts. Autapse's clear success cannot remove its separate
rate-floor and recovery failures.

Recovery uses the same ordinary controller and a 25-step offset grid, with 15
clear phases (0…56 by 4). These sweeps use fixed captured/corner parameters
without strays; they measure deterministic recovery, not clear probabilities.
A single reload must sustain storage and its reader
1,500 steps later at every phase and later tested offset. No-reload controls
must first prove the old orbit cleared. The synthetic 15-hop READY is a
comparison clock; the request fixture's actual START path is not modeled.

| Scenario | Legacy recovery | Autapse recovery | Lost margin |
|---|---:|---:|---:|
| Nominal | 675 | 700 | 25 |
| Fast three-sigma | 425 | 500 | 75 |
| Slow three-sigma | 1,075 | 1,100 | 25 |
| Request 28 | 500 | 600 | 100 |
| Master 74 | 525 | 575 | 50 |
| Master 17 | 475 | 525 | 50 |
| Master 24 | 550 | 625 | 75 |
| Stage 18 | 600 | 650 | 50 |

All eight rows, zero no-reload survivors, common READY times and strictly worse
autapse margins are asserted **with readers**. Nominal READY is +872; slow-corner
margins are −203 legacy / −228 autapse. At slow three sigma the high-gain
candidates recover at 900 / 950, better than legacy's 1,075. In the other seven
scenarios all 15 no-reload controls survive, so recovery and margin are reported
as **undefined**, not as successful re-ignition.

## Gates and multi-cell correctness

Gate tests include legacy controls. Without readers, nominal autapse storage
misses a legitimate two-rail fault; at the slow corner it also loses completion
AND and one-rail OR. Legacy's raw fault gate also misses at that slow corner, so
that particular failure is not attributed to the candidate. Noisy raw autapse
gates miss true inputs while the legacy controls respond to all of them.

With readers, legacy and all three candidates detect all intended inputs with
zero false faults in the 2,000-copy-per-condition mix-B tests: fault/completion
AND, OR, veto, majority and required-rail one-shot. The latter rejects each
absent input and never emits multiple pulses. Autapse passes nominal and
correlated fast/slow storage corners too. **Both high-gain designs miss both
conjunctions at the slow corner even with readers**; OR and veto still respond.
Thus the existing reader is useful but does not normalize arbitrarily slow
storage. These are measured gate cases, not an exhaustive independent-corner
enumeration of an entire gate network.

The synthetic copy-73 false-fault probe uses actual source latches: loop ×1.3,
threshold −1.2 mV, bias +0.6 mV; the gate has threshold −0.8 mV, bias +0.5 mV and
a 150-q stray at each of 47 offsets. This exceeds three sigma and is not a
reconstructed capture. All four reader-enabled controls reject false faults
with the opposite rail dark. Passing this probe cannot compensate for missed
legitimate inputs elsewhere.

The whole-kernel transform covers all **249** actual latch pairs, including
constants, requests, ALU, registers and image state in a 2-bit ADD → XOR kernel.
Inputs 0, 1, 3 must yield 0, 3, 1. “Other options” enables `zero_once`,
`robust_request_clear` and `experimental_register_reset` together.

| Readers | Other options | Legacy final step | Autapse final step |
|---|---|---:|---:|
| off | off | 47,304 | 55,616 |
| off | on | 50,728 | 59,215 |
| on | off | 45,499 | 46,147 |
| on | on | 48,825 | 49,570 |

All rows assert the complete output sequence, zero faults/timeouts/bad decodes,
and these final steps. Reader-enabled autapse overhead is **64.8 ms / 74.5 ms**,
versus 831.2 ms / 848.7 ms with raw gates. Both high-gain combinations produce
only `[0]` by 60,000 steps: one fault with other options off; zero counters with
them on. Missing outputs remain failure even with zero counters. The autapse's
successful nominal kernel cannot qualify its independent-corner rate floor or
actual-path reload behavior.

## Costs and validation

Autapse and both high-gain designs add **0 neurons / 2 feedback synapses per
latch**. An isolated candidate is 2 neurons / 4 synapses, or 3 / 5 with one reader
before reset fan-out. The 2-bit ADD → XOR kernel adds **0 neurons / 498 synapses**,
with or without readers. The earlier +630 count included 132 erroneous feedback
mirrors and is withdrawn. Existing reader/qualifier reset fan-out stays intact.
High gain also multiplies both excitatory loop weights by 8 and adds two 6-loop
inhibitory weights; its charge cost is much larger despite the same edge count.

The 8-bit tick kernel has **3,215 latch pairs**. With rate readers and the
experimental autapse it has **30,643 neurons / 62,366 synapses**: **+0 neurons /
+6,430 synapses** versus the rate-reader baseline (30,643 / 55,936), or +1,268 /
+9,558 versus the default kernel. Ordered probe equality and these counts are
asserted.

A nominal float64 RefSim Stage D run, generic datapath, other options off,
tokens 5, 5, 250, 3 (`tokens_for(4, 0)`), matches all four canonical states with
zero faults, timeouts and bad decodes. Tick completion times relative to the
first input load are 11,802.7 / 17,170.8 / 23,369.8 / 28,738.4 ms. The first tick
takes **11,802.7 ms**; mean subsequent per-tick latency is **5,645.2 ms**
((last minus first) / 3). This short nominal measurement is not Stage D
qualification. Reproduce with:

```sh
python -m drosophilos.bench.stage_d --backend ref --mix none --ticks 4 --seed 0 --c-ticks 0 --max-ms 120000 --rate-robust --experimental-autapse
```

The tick-kernel transform is checked edge-for-edge against the committed probe,
including ordered edges, weights, delays, biases, image and mirror registry.
Default kernel and control-machine fingerprints remain regression checks.
`tests/test_experimental_autapse.py` covers the allowed combinations and small
noisy kernels; `tests/test_build_options.py` exercises both CLIs, saved records,
diagnostic rebuilds and recheck.

Required fast validation:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_experimental_autapse.py tests/test_stable_latch.py tests/test_build_options.py -m 'not slow'
```

Slow qualification probes retain search-passer refutations, 40,000-trial
surveys, holding/distribution exposures, gate controls and recovery comparisons.

## Configuration F: wider kernel qualification (2026-10-11)

F means `rate_robust=True, experimental_autapse=True,
ordered_xor_oneshot=True`, with the existing default timing/reset policy.
These are **bounded RefSim composition qualifications for experimental builds**,
not a new Stage D exit, a proof over all mix-B draws, or a resolution of the
isolated rate-floor/reload frontier above. The control machine and every
option-off netlist are unchanged. Circuit versions remain autapse v1 and
ordered-XOR v1: no weights, delays, readers or reset trains were retuned.

| Shape | Added circuitry and reset ownership | Decision for F |
|---|---|---|
| ROM / LOAD | Exact address-mismatch veto neurons, read relays into operand-token latches; generic LOAD retains the PASSB mux and unused ordered adder, specialized LOAD retains only PASSB. Operand latches clear with the LOAD stage; no stored ROM word latch. | Allowed, widths 1–8. Autapse/one-shot apply unchanged. |
| RAM / STORE / LOAD | Word registers contain dual rails, bit-valid ORs, completion trees, conditioned readers and ordinary four-tap resets/READY chains. Each write port has a COPY latch cleared by its STORE stage. Shared ports add per-word marks and a completion-driven kill train for marks and all COPY latches. Address mismatch vetoes are combinational; reads feed LOAD's token latches. | Allowed, widths 1–8, under the existing valid-address, initialized-read and non-overlapping-write contract. No new reset policy. |
| MUL array | Partial-product rails; successive ordered ripple adders; completion-tree retiming between rows. All row sums, carries, retiming latches and one-shot spent latches share the cell's stage reset. | **2026-10-11: only widths 1–2 allowed; above 2 rejected** after the nominal primitive survey. **2026-10-12 follow-up: widths 1, 2, 4 allowed**, with wide-array bypass v1 for 4; see the appended timing diagnosis and qualification below. |
| MULP / MULP_ROW | Each row carries accumulator, A/B and flags in its stage/master (`3*n+3` bits), forms shifted partial products and uses one ordered ripple adder. Each row has its own request, commit, reset and one-shot spent latches. No between-row `Gates.retime` tree. | Allowed, widths 1–8. Existing row timing and reset ownership suffice in the tested envelope. |
| Multiple streams | One producer/stage/master, creq pair, watchdog and conditioned readers per stream, with the same reader-request/commit guards as the single stream. | Allowed with F; repeated two-stream operand and independent-output kernels complete. |
| Phases / pacing | Phase OK kill pairs gate input commits. Ring lines are kill pairs; multiple completion sources get separate rings, wrapped pairs and a guarded join. No XOR datapath is added by the pacing itself. | Allowed with F. Local autapses cover these pairs; ordinary four-tap kill trains are unchanged. Preserve the existing host schedule contract; phase gates do not supply missing tokens or fix cyclic dependencies. |

The autapse transform still runs **after** RAM ports, shared-port marks, phase
guards and rate readers are installed. It adds exactly one zero-delay
`-round(0.2*loop)` edge per latch member while suppressing feedback mirroring;
genuine clears retain their reader mirrors. `Gates.ordered_xor_oneshot` reaches
every `ripple_adder_ordered`, including MULP rows, and its spent latches are
included in that row's `extend_reset`. The default control-machine ALU never
opts in. Tests check reset ownership, every wider population's feedback, and
ordered option-off fingerprints captured from caf043e for all eight shapes.

`tests/test_config_f_kernels.py` uses float64 **RefSim with actual primitives**.
Mix B is independently drawn log-weight sigma .04, threshold/bias sigma .2 mV,
Bernoulli 5 Hz × 150 q background per neuron/step, and 0–100-step host-load
jitter. These are the campaign's laws, not its exact topology-dependent draw
sequence. Counts are finite exposures, not confidence bounds over game kernels.

Measured checks:

* All eight 2-bit nominal kernels (ROM, RAM, shared RAM, MUL, MULP, streams,
  each-phase and wrap-phase) computed the expected repeated values, with zero
  faults, timeouts, bad decodes, refusals or retries. Each/wrap tests additionally
  assert the order of phase outputs over three frames.
* RAM's fast survey performed 32 writes and 128 reads per condition (nominal
  and mix B), visiting all four addresses, overwriting with complemented rails,
  and reading every peer after each write. No wrong, double or missing rails,
  surviving COPY/read-register latches, or missing write/read completions.
  Word overwrite uses its real reset → READY → COPY path, not a host reload.
  Read-register and port clears are inspected after 200 ms, then reused.
  The survey also reports the interval from the word's last inhibitory spike
  to COPY ignition; this is a measured sample margin, not a universal minimum.
  A separate 8-bit, seed-6110 mix-B margin probe measured **58.4–67.0 ms**
  from the last clear spike to COPY (two writes, eight reads, no errors).
  Expanded RAM surveys passed **256 writes / 1,024 reads** in each of the
  4-bit nominal, 4-bit mix-B and 8-bit mix-B conditions, with all five failure
  counters zero. The 4-bit nominal sample covers every address/value pair.
* Two-bit MUL tests all 16 operand pairs nominally, then complements both operands
  on the same physical copies. The noisy survey uses 16 random pairs; each condition
  has 32 transactions and no failures.
  Every actual multiplier latch is inspected after the ordinary clear and
  200 ms recovery; spent-edge counts detect repeated one-shot firing.
* **Four-bit array MUL fails:** exhaustive nominal 256 pairs plus their
  complemented reloads gives **16 missing high rails / 512 transactions**,
  zero wrong/double rails, repeated one-shots or reset survivors. The 128-draw
  reported mix-B measurement gives **4 missing high rails / 256 transactions**. The smallest
  retained witness is 7 × 1, with the ordinary kernel B/SUB XOR and A/B delays.
  The same two-transaction witness passes with autapse/rate readers but without
  sum one-shots: this exposes a first-edge timing issue in the retimed array,
  not a failed latch clear. Repeated legacy XOR pulses can recover that case.
  A full kernel's particular nominal 7 × 1 schedule does compute 7; it does not
  invalidate the primitive's allowed-phase witness. The public builder refuses
  wider arrays instead of relying on that one favorable schedule. A repair must
  qualify first-edge timing through every retiming row before lifting this gate.
  **2026-10-12 follow-up:** the extended neuron trace below corrects this
  interpretation: the top rail arrives after the survey's observation deadline;
  the first edge is neither lost nor recovered by a repeated legacy pulse.
* MULP exhausts 2-bit operand pairs (32 transactions including complemented
  reloads); noisy 4-bit and 8-bit pipelines test 16 and 8 transactions,
  respectively. All complete correctly. This tests the real expanded rows and
  their full handshakes, including both input streams and input jitter.
* Both generic and specialized two-copy mix-B versions of all eight small
  kernels pass, including host-bit jitter, repeated writes through shared RAM
  ports, and three frames of each/wrap pacing. Ring primitive surveys include
  K=1, 2 and 3, and the larger **64-copy, K=8, three-source join**: **4,608
  advances**, no wrong/double/dark line rails or missing/extra wraps. These use
  the kernel's four-tap kill trains, not the control machine's three-tap drive.
* The real 8-bit render column kernel completes two passes / 16 outputs
  nominally in 22,068.4 ms neural, with all values correct and no faults,
  timeouts or bad decodes. This is a short composition check, not the proposed
  1,000-pass campaign.

Widths above eight remain unqualified for the ordered one-shot. In particular,
**doom2/doom4 are not unlocked** by these small-width results. Alternative
physics, timing or reset policies retain the existing rejection. Autapse-alone
wider kernels also remain rejected: these measurements concern F.

Build records retain all effective flags and the existing circuit-version keys.
`kernel_campaign.rebuild_record(record)` reconstructs current render/perspective
campaign blocks and strictly checks every recorded build option and version,
plus neuron/edge counts if present. Missing versions are rejected there;
historical tick captures still use `stall_diag`'s existing compatibility rules.
With configuration F plus `zero_once`, only tick2.c-shaped kernels are allowed because wider-shape combinations remain unsurveyed.

### Costs and the next campaign

Costs below are exact construction counts, with default timing and `zero_once`
off. Render is `kernel_campaign.block("render")`: the 8-bit column kernel,
heading 3. Doom2 is the **full source**, compiled with `compile_program`, array
MUL, neural pacing and counts `{"input": 160, "p": 16000}` (not the resized
8×5 performance fixture).

| Build | Rate readers only, neurons / synapses | F, neurons / synapses | Added by F |
|---|---:|---:|---:|
| render, generic | 11,448 / 20,942 | **11,768 / 24,228** | +320 / +3,286 |
| render, specialized | 8,238 / 14,792 | **8,318 / 16,730** | +80 / +1,938 |
| doom2, generic, **cost-only candidate** | 1,037,804 / 2,041,991 | **1,050,604 / 2,298,941** | +12,800 / +256,950 |
| doom2, specialized, **cost-only candidate** | 955,644 / 1,884,471 | **962,044 / 2,107,085** | +6,400 / +222,614 |

The doom2 F counts were obtained by constructing the rate-reader netlist with
`Gates(..., ordered_xor_oneshot=True)` and applying the v1 autapse finalizer.
That offline cost calculation bypasses the gate **only for counting**; it is
not a runnable/qualified public F build. Both 16-bit width and its wider array
MUL remain blockers. Default render counts are 11,018 / 19,942 generic and
7,808 / 13,792 specialized.

See [stage_d.md](stage_d.md#render-campaign-after-the-wider-shape-surveys) for the
render campaign commands, exit checks and one-H200 planning estimate. Required
fast regression:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_config_f_kernels.py tests/test_experimental_autapse.py tests/test_ordered_xor.py tests/test_build_options.py -m 'not slow'
```

Expanded surveys use `uv run pytest -s tests/test_config_f_kernels.py -m slow`.
The rejected array surveys assert the rejection and retain the nominal failure
count; they must not be relabeled as successful arithmetic qualification.
The 2026-10-12 follow-up below qualifies a changed wide-array circuit;
its version-0 witness controls retain the historical deadline failure.

## Configuration F: array MUL deadline diagnosis and prefix bypass (2026-10-12)

The preceding survey counts and their original interpretation are retained as
history. The new `test_retimed_array_neuron_trace_disproves_lost_first_edge`
extends the **same 7 × 1 primitive**, without a clear at 900 ms, to 1,000 ms.
It uses unmodified float64 RefSim integration, actual operand latches, six-hop
B/SUB XOR staging, seventeen-hop A staging, completion retiming, autapse v1,
sum one-shot v1 and ordinary reset circuitry. ZERO is loaded at step 0 and
the operands at step 200 (20 ms). Times below are absolute, with dt = 0.1 ms.

| Original full array neuron | First spike, step | Time, ms |
|---|---:|---:|
| `mul.row3.fa3.x.y0.u` (top operand XOR = 0) | 7,929 | 792.9 |
| `mul.row3.fa2.c0.u` (carry into top stage = 0) | 8,711 | 871.1 |
| `mul.row3.fa3.cinr0.d4` (five-hop delayed carry) | 8,976 | 897.6 |
| `mul.row3.fa3.s.once0.edge` (transaction one-shot) | 9,018 | 901.8 |
| `mul.row3.fa3.s.once0.edge_inh` (source inhibitor) | 9,029 | 902.9 |
| `mul.row3.fa3.s.once0.spent.u` | 9,060 | 906.0 |
| `mul.row3.fa3.s.y0.b.edge` (correct sum ignition) | 9,060 | 906.0 |
| `mul.row3.fa3.s.y0.b.edge_inh` (sum relay inhibitor) | 9,071 | 907.1 |
| `mul.row3.fa3.s.y0.u` (product bit 3, zero rail) | 9,102 | 910.2 |

The EARLY XOR is already valid **104.7 ms before** the delayed carry. The
transaction one-shot fires exactly once and the downstream sum relay also
fires exactly once, 4.2 ms later. Their feed-forward inhibition arrives at
904.7 and 908.9 ms respectively (18-step synaptic delay), after each first
edge has fired. There is no missed one-shot window or late
veto: the old survey's `duration = 1000 + width*width*500` ends at step 9,000
for width four, **before the one-shot even fires**. It scores the top rail as
missing and starts an ordinary clear. In this witness the missing rail is
**r0 of the top bit**, not logical-one r1; the other three bits are correct.
With sum one-shots disabled, the top rail's *first* spike is step 8,976, inside
the same window. The three rows' additional one-shot relays add 12.6 ms to
the full array's critical path. Thus the witness does not substantiate the
earlier repeated-pulse recovery explanation. A longer observation reveals a
correct result, but does not qualify completion before the original clear.

Wide-array circuit **`ordered_xor_oneshot_mul_version = 1`** keeps that
deadline and bypasses the zero-addend prefix instead. In shifted row j, bits
0 through j−1 add zero with Cin=0, so they cannot generate a carry. Those
accumulator rails pass through unchanged; the ordered adder starts at bit j
with the existing ZERO carry-in. Completion-tree retiming still covers the
whole accumulator and aligns all remaining LATE bits. Sum one-shots, veto
weights, latch constants, reader policy, kill trains and ordinary clears
are unchanged. In a shortened row `fa0` names product bit j.

The bypass applies only to one-shot arrays wider than two bits. Every
non-F circuit, both small F array widths, all other F units, tick2.c generic
and specialized, and render retain their original ordered netlist bytes.
Frozen SHA-256 tests include roles, ordered edges/weights/delays, biases,
reader/reset-mirror registries and image latches. Non-F wide arrays also
compare against the retained version-0 construction. The new version key
is recorded **only when a wide F array is present**; existing campaign replay
checks every effective build-option key, rejecting missing or mismatched
versions. Version 0 is retained solely for primitive historical controls;
the public builder selects the qualified current version.

There is **no added latency**. For 7 × 1, all four correct product rails
are live by **699.5 ms**, a **210.7 ms reduction** from the original F array.
The qualification fixture, including ordinary reset fan-out, local autapses
and downstream retiming readers, has the following exact savings:

| Width | Removed neurons / synapses in successive rows j=1..n−1 | Total before → after |
|---|---|---|
| 4 | 70 / 148; 140 / 296; 204 / 432 | 1,565 / 3,292 → **1,151 / 2,416** (−414 / −876) |
| 8 | 70 / 148; 140 / 296; 210 / 444; 280 / 592; 350 / 740; 420 / 888; 476 / 1,008 | 6,497 / 14,028 → **4,551 / 9,912** (−1,946 / −4,116) |

Each non-final row saves 70*j neurons and 148*j synapses; the final row
saves 68*j and 144*j because this primitive fixture has no downstream
output rate readers. These are fixture counts, not a whole-kernel count.
Partial products and retiming trees retain their original construction.

Qualification retains the original deadline (550 ms at width three, 900 ms
at width four), complemented reloads on the same physical copies, all-latch
survivor checks after the ordinary clear and 200 ms recovery, and spent-edge
counts throughout each transaction. Mix B retains independent log-weight
sigma .04, threshold/bias sigma .2 mV, 5 Hz × 150 q background and 0–100-step
host-load jitter. Every counter below is zero: wrong/double/missing rails,
repeated one-shots and reset survivors.

| Qualified width | Nominal operand coverage / transactions including complements | Mix-B transactions including complements |
|---|---|---:|
| 3 | All 64 pairs / **128** | **256** |
| 4 | All 256 pairs / **512** | **256** |

The public builder currently allows array MUL widths **1, 2, 4**. The
3-bit result qualifies the multiplier primitive only: the generic public ALU
requires a power-of-two result subtree for its zero flag, so width three is
still refused. Unsurveyed widths also remain refused; MULP's separate
qualification is unchanged.

## Configuration F: prefix-bypass review corrections (2026-10-12)

The earlier bypass counts are retained above as construction history. Wide-array
v1 now omits the unused retime output latches and veto relays for bits below
each row's shift. `retime(..., output_start=j)` still constructs bit-valid
latches for every accumulator bit, the same whole-word completion tree and
the same completion delay; only output bits j through n−1 are copied. Other
retime callers retain all outputs. The wide-array version remains **1**.
All pre-existing frozen digests remain unchanged; there was no frozen digest
for the version-1 wide-array circuit.

Corrected exact primitive fixture counts, including ordinary reset fan-out,
local autapses and downstream readers, supersede the preceding cost table:

| Width | Removed neurons / synapses versus version 0 in successive rows j=1..n−1 | Version 0 → reviewed version 1 |
|---|---|---|
| 4 | 80 / 172; 160 / 344; 234 / 504 | 1,565 / 3,292 → **1,091 / 2,272** (−474 / −1,020) |
| 8 (construction only) | 80 / 172; 160 / 344; 240 / 516; 320 / 688; 400 / 860; 480 / 1,032; 546 / 1,176 | 6,497 / 14,028 → **4,271 / 9,240** (−2,226 / −4,788) |

The unused output removal saves a further 10*j neurons and 24*j synapses per
row: 60 / 144 at width four and 280 / 672 at width eight. Combined savings
versus version 0 are 80*j / 172*j for non-final rows and 78*j / 168*j for
the final row. Completion circuitry and partial products retain their original
construction.

Width eight is **not qualified**: its mix-B survey was never completed, and
the public builder continues to refuse it. Its nominal 256-pair survey
(512 transactions including complemented reloads) and mix-B 128-pair survey
(256 transactions including reloads) now live in the separate opt-in slow
`test_multiplier_width8_cluster_survey`. It skips unless `DROSO_WIDE_MUL=1`:

```sh
DROSO_WIDE_MUL=1 uv run pytest tests/test_config_f_kernels.py -q -s -k test_multiplier_width8_cluster_survey
```

Those zero-failure assertions are future cluster qualification criteria, not
claims of completed validation. The ordinary large-survey parametrization now
contains only widths three and four; the separate width-four nominal kernel
test still covers output mux, completion, commit and reset ownership.

Foreground review validation:

* `uv run pytest tests/test_build_options.py tests/test_config_f_kernels.py tests/test_ordered_xor.py -q -m 'not slow'`
  exited **0**, with **199 passed** and all frozen digests unchanged. The new
  completion-tree structural test was added after this run collected its tests
  and passed separately: **1 passed**, with the two width-eight cluster cases
  **skipped** without the opt-in variable. Together these cover all **200**
  fast tests in the final file state.
* `uv run pytest tests/test_config_f_kernels.py -q -k 'test_multiplier_large_survey or test_qualified_array_kernel_nominal'`
  exited **0**: **5 passed, 95 deselected in 186.47s (0:03:06)**. Width-three
  nominal / mix-B covered **128 / 256** transactions; width-four nominal /
  mix-B covered **512 / 256** transactions. All wrong/double/missing,
  repeated-one-shot and reset-survivor counters were **zero**. The full
  width-four nominal kernel also passed. The unused-output removal is retained.

These runs used the existing project environment without dependency syncing,
with imports directed at this worktree and writable temporary/cache locations.
The slow run overrode the configured extra `-q` solely to show pytest's exact
result summary. Width-eight cluster qualification was not run.
