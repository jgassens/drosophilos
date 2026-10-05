# Stable storage with rate-conditioned readers: measured frontier

**None of the evaluated combinations qualifies. The reserved `stable_latch`
option has been removed from Drive, builders, records, CLIs, diagnostics and
Stage D rechecks.** Production storage and the control machine are unchanged.
Candidate circuits exist only in `tests/test_stable_latch.py`.

The earlier claim that the autapse fails with `rate_robust` readers was a **probe
artifact**. `_modify` passed local inhibitory feedback to `net.synapse`, which
also copied it at gain 16 into existing readers and into required-rail qualifiers.
A latch spike consequently suppressed its own consumers. The fixed transform
builds all consumers first, suppresses mirroring only while adding local feedback,
and preserves genuine reset/kill mirrors. A regression checks pre-existing resets
and resets added after the transform. This construction order is a test-fixture
constraint, not a new production Netlist API.

With that correction, autapse + readers computes **0, 3, 1 with zero faults** in
the multi-cell kernel, including with the other options enabled. It remains
unqualified because storage admits a 36-step independent-corner interval and
reload margins are worse than legacy. High-gain storage fixes the rate floor,
but ordinary clears leave live storage, the slow corner still misses gate
inputs, and the kernel stops after its first output.

## The six requirements

All six must pass before an opt-in implementation is justified:

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

| Requirement, with readers | Autapse | High gain | More rate margin |
|---|---|---|---|
| 1. Storage interval ≥44 | fails: 36 | measured floor 46 | measured floor 54 |
| 2. Holding | no observed dropout | no observed dropout | no observed dropout |
| 3. Ordinary clear | zero survivors in all eight surveys | fails | fails |
| 4. Recovery versus legacy | worse in all eight comparisons | undefined after failed clears in seven | undefined after failed clears in seven |
| 5. Gate inputs | passes tested cases | misses slow-corner conjunctions | misses slow-corner conjunctions |
| 6. Multi-cell kernel | correct with options off/on | only first output | only first output |

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
copy 24 and stage Z0 base copy 18. Old edges, thresholds and biases retain their
reconstructed seed-108, B=100 draws. Added feedback/reader edges receive new
independent mix-B draws; new reader threshold/bias draws are independent too.
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
all trials, but the longer-delay candidate leaves 6/40,000 survivors; that rare
failure also escapes the smoke test. Captured and mix-B scenarios refute both
high-gain designs as well. The exceptions and their matching reader-survivor
counts are asserted. Autapse's clear success cannot remove its separate
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
before reset fan-out. The whole kernel adds **0 neurons / 498 synapses**, with
or without readers. The earlier +630 count included 132 erroneous feedback
mirrors and is withdrawn. Existing reader/qualifier reset fan-out stays intact.
High gain also multiplies both excitatory loop weights by 8 and adds two 6-loop
inhibitory weights; its charge cost is much larger despite the same edge count.

Since no combination qualifies, no stable-latch flag, Drive field, build-option
record key or rebuild plumbing is retained. The previous source and
`test_build_options.py` additions are reverted; only this document and the
qualification probes extend the pre-experiment code. Existing ordered
kernel/machine fingerprints check the unchanged production circuits.

Required command:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_stable_latch.py tests/test_build_options.py -m 'not slow'
```

The sandbox blocks uv's default cache (`~/.cache/uv/sdists-v9/.git`). Validation
uses `uv run --no-sync`, the existing repository virtualenv, this worktree on
`PYTHONPATH`, and task-local cache/temp directories. Slow tests retain every
screen-passer refutation, 40,000-trial scenarios, holding/distribution exposures,
gate controls, recovery comparisons and the corrected multi-cell results.

The final fast selection passes **131 tests**, with 89 slow tests deselected.
All **88 stable-latch slow cases** pass across scenario partitions and targeted
reruns: 32 clear cases (1,280,000 trials), 21 search/recovery cases and 35
gate/holding/distribution/kernel cases. The request-draw preservation and
recovery checks were rerun after correcting its new-mirror weights. Production
source and `test_build_options.py` match `9f2190c^` exactly; relative to that
pre-experiment commit, only this document and `test_stable_latch.py` differ.
No commit is attempted; the orchestrator owns integration and commits.
