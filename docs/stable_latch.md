# Rate-stable latch: qualification failed

**No evaluated design meets all six requirements. `stable_latch=True` is
reserved and rejected, including campaign CLIs, diagnostic rebuilds and Stage D
`--recheck`.** This is a measured design frontier, not a released fix. The
legacy default, ordered netlist bytes and control machine remain unchanged.

The most useful partial result is a 0.2-loop inhibitory autapse on each member:
it holds, improves the random period distribution, and clears the captured
entrained cases with the ordinary train. It nevertheless admits a 36-step
interval at an independent corner, loses reload margin, misses legitimate gate
inputs and fails in combination with rate-conditioned readers. Shipping it as
“stable” would be misleading.

## Method and circuits

All probes are retained in `tests/test_stable_latch.py`. They use the real
`add_latch`, `add_reset`, `add_kill_train`, AND/OR/veto primitives and unchanged
float64 RefSim. Candidate changes are test-only netlist transformations.
There is no spike clipping, simulator refractory change, prescribed latch
train or substituted clear pulse sequence.

Mix B means independent `round(q * exp(N(0, .04)))` weights, independent
threshold and tonic-bias sigma .2 mV, and independent Bernoulli 5 Hz × 150 q
strays on **every** neuron. `MixProbe` uses the existing sparse-binomial sampler
from `test_fast_latch.py`. This is the campaign probability law, not the CUDA
random bitstream. Gaussian tails are unbounded; finite samples cannot establish
indefinite holding or an absolute rate bound.

Each independent corner enumerates both signs of every edge's .12 log-weight
offset, each neuron's .6 mV threshold offset and each .6 mV bias offset. Both
single weak ignition and the captured four-ignition schedule (steps 10, 11, 14,
23) are exercised; a 300-q kick at step 2520 challenges coincidence. The single
ignition is `round(drive.ignite * exp(-.12))`. The four-pulse cases add three
ordinary ignition pulses to that first pulse. Measurements include the kick's
transient. Independent corners are essential: symmetric fast/slow corners
missed severe failures.

Candidate parameters below are multiples of the original 3,621-q loop; delays
are steps of .1 ms. External ignition and clear amplitudes are unchanged.

| Candidate | Loop gain; u→v / v→u delay | Negative feedback | Neurons / synapses per isolated latch |
|---|---|---|---:|
| Legacy | 1; 18 / 18 | none | 2 / 2 |
| Autapse | 1; 18 / 18 | −.2 on each member, delay 0 | 2 / 4 |
| Zero-delay | .95; 0 / 0 | −.1 on each member, delay 8 | 2 / 4 |
| Shared inhibitor | 1.2; 18 / 18 | u and v drive one interneuron at pulse strength/delay 0; it inhibits both at −.1/delay 18 | 3 / 6 |
| Asymmetric | .95; 24 / 60 | none | 2 / 2 |
| High gain | 8; 22 / 22 | −6 on each member, delay 22 | 2 / 4 |

The broad screen covers 3,227 configurations; the complete grids are in
`candidate_grid`. Each runs 5,000 steps, with five correlated electrical
corners and two ignition modes. A live train requires a spike in the last
200 steps. Screen intervals start after step 3,000. This is a search filter,
not qualification.

| Grid | Configurations | Hold all screen cases | All screen intervals ≥44 |
|---|---:|---:|---:|
| Lower gain and asymmetric delays | 1,089 | 847 | 0 |
| Direct/shared delayed inhibition | 704 | 415 | 5 |
| Tonic bias and loop/delay changes | 384 | 208 | 0 |
| High gain with inhibition | 600 | 242 | 39 |
| Refined low-gain/autapse delays | 450 | 295 | 35 |

Twenty refined settings also have a nominal screen period in 44–56 steps.
**All twenty lose holding and admit intervals below 44 when their draws are
independent.** Each additionally received a 2,000-copy, 10,000-step random
mix-B probe. For example, the zero-delay representative has 658 silent copies
and a 35-step minimum in that random probe. Its 512 independent initial-state /
corner cases include 173 silent copies and a 34-step minimum. The 44–56 window
was only a shortlist criterion; slower candidates were also measured and
rejected on actual gates, clear behavior or recovery.

## Period and holding distributions

`period_survey`: seed 109, 10,000 independently perturbed latches per circuit,
6,000 steps, one ignition at step 10, strays everywhere. Intervals have their
first endpoint at or after step 2,000, matching the legacy survey. Added edges
change RNG assignment; these are comparable samples, not paired draw replays.
Period tables report the output tap `u`; clear probes observe both members.

| Individual intervals | min | p1 | p5 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 32 | 43 | 44 | 46 | 49 | 50 | 55 |
| Autapse | 47 | 51 | 52 | 57 | 62 | 65 | 73 |
| Zero-delay | 36 | 38 | 39 | 44 | 65 | 79 | 113 |
| Shared inhibitor | 31 | 42 | 43 | 45 | 48 | 49 | 53 |
| Asymmetric | 34 | 46 | 47 | 48 | 88 | 92 | 102 |
| High gain | 54 | 60 | 63 | 75 | 89 | 96 | 116 |

| Per-latch mean intervals | min | p1 | p5 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 38.202 | 43 | 44 | 46.035 | 49 | 50.167 | 54.944 |
| Autapse | 48 | 51 | 52.513 | 56.971 | 62.629 | 65.3 | 71.836 |
| Zero-delay | 37 | 38.063 | 39.188 | 46.506 | 71.392 | 89.733 | 105 |
| Shared inhibitor | 35.658 | 41.989 | 42.957 | 45 | 47.964 | 49 | 52.32 |
| Asymmetric | 35 | 46 | 46.976 | 48.506 | 89.683 | 92.905 | 101.895 |
| High gain | 54 | 60 | 64 | 75.308 | 90 | 97 | 115.879 |

Interval counts are respectively 853,014; 692,798; 526,973; 880,170; 762,041;
522,018. Zero-delay has 3,370 copies without measured intervals and 3,968
without a spike in the final 200 steps. Every other representative has zero
in both categories. Omitting silent copies from the percentiles does **not**
count them as successful storage.

| Independent corner result | Initial-state / corner cases | Minimum interval | No holding at end |
|---|---:|---:|---:|
| Legacy | 128 | 32 | 0 |
| Autapse | 512 | 36 | 0 |
| Zero-delay | 512 | 34 | 175 |
| Shared inhibitor | 8,192 | 30 | 0 |
| Asymmetric | 128 | 34 | 0 |
| High gain | 512 | 46 | 0 |

The autapse's correlated slow corner approaches a 105-step period, about
95 Hz, versus its 57-step nominal period, about 175 Hz. Legacy nominal is
about 213 Hz. Separate 100,000-step (10 s) random exposures of 1,000 latches
each gave zero silent or terminally dark latches for legacy and autapse: 10,000
latch-seconds per design, minimum intervals 34 and 47 respectively. That is
zero observed terminal holding failures, not an indefinite-holding proof.
The independent corner exposures were also extended to 100,000 steps: zero
terminal holding failures in all 128 legacy and 512 autapse cases, with minimum
intervals still 32 / 36 and maximum intervals 62 / 105 respectively. Those
deterministic corner runs include the ignition bursts and kick, but no strays.

## Ordinary clears and reload frontier

`clear_survey` uses the real four-tap, .75-loop controllers, not four prescribed
arrivals. Four configured taps can emit five spikes. All trials include the
four close ignitions, the 300-q kick, random clears at steps 3,000–4,499 and
strays on the entire circuit. A survivor fires more than 1,500 steps after
clear; a separate count verifies that the storage was live before clear.

Captured primitives retain reconstructed seed-108, B=100 campaign draws for
all old edges, thresholds and biases: request RR copy 28, master base copies
74/17, master RR copy 24 and stage Z0 base copy 18. These reuse the checked
role mapping from `test_completion_stall` and the request constants in
`test_request_clear_entrainment`. New autapse edges receive independent .04
log-normal draws; they have no historical capture draws. Captured loop weights
are preserved, including the request fixture whose weights are stored directly
in its netlist. This is a conditional mechanism probe, not a replay of the
captured CUDA trajectory or an estimate of Stage D stall frequency.

The autapse has **0/40,000 survivors and 0/40,000 unlit-before-clear trials in
each of eight scenarios**: all five captured primitives, correlated fast and
slow three-sigma storage, and independent mix-B static draws. Total: 0/320,000
survivors. This does not qualify independent controller corners or the entire
kernel reset domain. The high-gain representative, despite
its 46-step independent-corner floor, leaves **256/256 survivors** in the
fast-three-sigma clear smoke test. Its large circulating charge invalidates
the premise that a bounded period alone makes ordinary inhibition sufficient.

`reload_survey` compares identical ordinary controllers on a 25-step grid at
15 clear phases (0…56 by 4). Recovery is the earliest offset for which every
phase, and every later tested offset, sustains a train 1,500 steps after a
single reload. No-reload controls have zero survivors. Offsets are from the
controller input, not from an assumed last pulse. Slow-three-sigma storage
has threshold +.6, bias −.6, weak excitation/ignition `exp(-.12)` and strong
inhibition `exp(.12)`. Controller and synthetic 15-hop READY retain nominal
physics; the captured old controller keeps its reconstructed draws. The
request fixture's READY is a comparison clock, **not** its actual START path.

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

The nominal comparison READY is at +872; the slow-corner margins are already
−203 legacy and −228 autapse. No actual-path reload superiority is claimed:
the isolated comparison already falsifies “at least as good as legacy.”

## Gates, costs and kernel latency

The gate fixture has actual live/dark latches, a .55 fault AND, .65 completion
AND, completion OR and pulse-driven veto. Slow storage is perturbed separately
from the nominal gates. At the autapse slow corner both rails keep firing,
but neither AND fires, and one rail cannot fire the OR. Two rails fire the OR;
the veto still responds. At nominal autapse parameters, even two rails fail
to fire the .55 fault gate. Zero false faults alone would hide this loss of
true fault detection.

With independently noisy gates and strays, 2,000 copies per dark/one-live/
two-live condition give 0 false fault spikes, but only 964/2,000 true faults
and 1,986/2,000 true completion conjunctions. OR and veto respond to every live
condition. These counts include the entire 10,000-step exposure, including
startup; measuring only after step 2,000 gives different miss counts.

The `false_fault_probe` uses the .55 gate, −.8 mV gate threshold, +.5 mV gate
bias and 150-q gate stray from the synthetic copy-73 probe in
`test_fast_latch.py`. Here the source is an actual fast30 latch: loop ×1.3,
threshold −1.2 mV, bias +.6 mV. This deliberately exceeds three sigma and is
not a reconstructed capture. Across 47 stray offsets, legacy has 47 false
faults (33-step train), autapse 0 (38-step train), and high gain 47 (24-step
train). The opposite rail stays dark. Passing this one false-fault probe does
not excuse the autapse's missed true inputs or sub-44 corner intervals.

The test-only whole-kernel transform covers all 249 actual latch pairs in a
2-bit ADD → XOR kernel, including constants, requests, ALU, registers and
image state. Inputs 0, 1, 3 should produce 0, 3, 1. “Other options” means
`zero_once`, `robust_request_clear` and `experimental_register_reset` together.

| Rate readers | Other options | Legacy final completion step | Autapse final step / result |
|---|---|---:|---|
| off | off | 47,304 | 55,616; correct 0, 3, 1 |
| off | on | 50,728 | 59,215; correct 0, 3, 1 |
| on | off | 45,499 | no outputs by 60,000 |
| on | on | 48,825 | no outputs by 60,000 |

Successful candidate runs add 831.2 / 848.7 ms to the final completion, with
zero faults, timeouts or bad decodes. An empty output sequence is a failure,
even when those counters stay zero. Costs are +0 neurons / +498 synapses
without rate readers and +0 / +630 with them. The extra 132 edges are
existing inhibition mirrors: 98 onto rate readers and 34 onto required-rail
qualifiers. Adding an inhibitory autapse also sends that feedback to these
targets. This interaction is exercised and remains
unqualified. Isolated per-latch costs exclude such mirrors and reset fan-out.

## Integration and validation

`Drive` and `build_pipeline` expose `stable_latch: bool = False`. Every successful
build records false. Both campaign parsers expose `--stable-latch`, whose help
states that it is unqualified. True is rejected before storage allocation,
including a caller-supplied Drive, every other opt-in combination, recorded
Stage D rechecks and diagnostic rebuilds. Old records default false; nested
build options remain authoritative. No experimental circuit is silently
substituted and no machine latch policy is changed.

Fast tests assert representative failures, reconstruction, clears, recovery,
CLI/rebuild rejection and existing ordered kernel/machine hashes. Slow tests
retain the full search, distributions, 40,000-clear surveys and multi-cell
frontier. Failing qualification is asserted explicitly rather than hidden
behind skipped or expected-failure tests.

The requested command is:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_stable_latch.py tests/test_build_options.py -m 'not slow'
```

The sandbox rejects uv's default cache at `~/.cache/uv/sdists-v9/.git`.
Validation therefore uses the existing repository virtualenv, `uv run
--no-sync`, this worktree on `PYTHONPATH` and task-local cache/temp paths.
The required fast selection passes: **118 passed, 27 deselected** (429 s),
including all existing ordered kernel and machine fingerprints.
The complete new slow selection (`pytest -q tests/test_stable_latch.py -m slow`,
with the same environment overrides) also passes: **26 passed, 19 deselected**
(403 s). That run asserts the 320,000 clears, full period distributions,
long holding, gate failures, multi-cell frontier and complete design screen.
No commit is attempted; the orchestrator owns integration and commits.

Unresolved: a circuit satisfying the independent rate floor, holding, ordinary
clear charge, unchanged reload margin and all gate semantics; qualification of
its actual stage/COPY/request reload paths and combinations; and a successful
enabled production opt-in. The search is finite and does not establish that
such a circuit is impossible.
