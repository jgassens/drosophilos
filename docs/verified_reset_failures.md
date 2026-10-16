# Verified-reset slow-test failures

## Status and caveats

- Verified reset version 1 is the **REJECTED** direct-experiment prototype
  (`docs/stage_d_completion_stall.md`; `docs/stable_latch.md:35–41`). These two
  slow tests asserted guarantees it never had.
- Version 2 is an opt-in correction for this prototype only. No default,
  recorded campaign, Stage D build, or public kernel flag uses it. A coordinator
  check of the 77 Stage D records found 66 with
  `verified_register_reset=False` and 11 that predate the flag (default off),
  so no Stage D result used verified reset.
- Version 2's immediate local feedback is an inhibitory SELF-edge (4.4 x loop,
  zero delay) on neurons whose other outputs are excitatory. This violates
  Dale's law (one neuron, one transmitter sign), just as configuration F's
  latch autapse does (`docs/stage_h/autapse_census.md`,
  `docs/stage_h/dale_feedback.md`). It could not be carried on real wiring
  without a separate inhibitory interneuron.
- Full-size cluster results (verified reset v2, commit ba3fa27): test_10000_verified_actual_noisy_reload_paths passed in both parametrizations (G2 job 4020998, 59 min). test_40000_verified_reset_survey passed in all 16 parametrizations: 13 in G2 job 4020997 before it reached its 4-hour limit, and the remaining three, fast30_bias-nominal, fast3sigma-fast_bias (the case that failed under v1 with 443 false_ready) and fast30_bias-slow_bias, in Juno jobs 459426, 459427 and 459428 (one passed each, 13-15 min).

## Scope and historical evidence

The introducing commit is `c8e94c604efdccbf2ee2468f71317a8dc2499d0e`
(2026-10-04), whose message starts “Completed fallback B after closed-loop
evaluation exposed full-domain and sequencing failures.” Its diff introduces
both tests with their present zero-error assertions. Neither assertion is a
recorded full-size passing result or an expectation of exactly six events.

Relevant evidence in the existing documentation (line numbers at `826b9fd`):

- `docs/stage_d_completion_stall.md:896–904`: verified reset is rejected;
  the separate timed 1.75/eight-link fallback is an unqualified campaign opt-in.
- Lines 912–939: monitor the complete negative inhibitor fan-out, preserve the
  ordinary four-tap 0.75-loop reset, propagate a silence certificate, allow three
  retries, then fail-stop through sticky exhaustion.
- Lines 941–945: this is a candidate certificate, not a proof for arbitrary
  schedules; exhaustion and missing READY are not successful emptying.
- Lines 970–979: extra-fast terminal survivors are deliberately fail-stop;
  the combined fast-three-sigma/fast-biased-controller screen passed only
  **64** noisy trials. That result does not establish a 40,000-trial guarantee.
- Lines 996–1004: the actual reload screen passed only **ten** noisy trials per
  build, using a 32,000-step observation horizon to include retries.
- Lines 1025–1031 explicitly say that the large tests were **not run**, claim no
  verified-reset 40,000/10,000 success, and distinguish the timed fallback's
  earlier successes.
- `docs/stable_latch.md:35–41` retains rejection of verified reset. Its autapse
  campaign results do not qualify this separate circuit.

`verify_register_resets` finalizes the direct-experiment prototype after all
wiring. `add_reset_verification` captures every negative reset target, drives
BUSY from those targets, and inserts four quiet links after recovery. A
seven-link parallel deadline either loses to the certificate or starts the next
of four statically unrolled attempts (`VERIFY_MAX_RETRIES = 3`). Exhaustion is
outside the reset domain. None of this enables the public verified kernel flag.

## A: primitive safety survey

`test_40000_verified_reset_survey[fast3sigma-fast_bias]` checks **zero
`false_ready`** and at most one spike at each attempt entry. It does not check
the reload harness's `failures`/`ready_missing`/`faults`/`survivors` fields.
`false_ready` counts a trial with either multiple READY spikes or a storage
spike at/after the first READY minus 100 steps. Thus the reported **443 is
`false_ready`**, not reset exhaustion or failure to reach READY. All 40,000
trials reaching READY does not imply that those READY outputs are safe.

The corner uses the existing `CORNERS['fast3sigma']` storage draws (loop
`exp(.12)`, threshold -.6 mV, bias +.6 mV, clear `exp(-.12)`). `fast_bias`
lowers control thresholds by .6 mV, adds .6 mV bias, strengthens excitation by
1.12, and weakens new verification inhibition by .88. The nominal controller
instead retains captured old draws and gives new detector edges/neurons
independent static perturbations in noisy surveys. Both receive seed-108
5-Hz/150-q strays and random reset phases.

Reduced reproduction on the unchanged circuit, with 250-trial batches:

| Trials | READY | false_ready | exhausted / undecided / terminal survivors | max_attempt_spikes |
|---:|---:|---:|---|---:|
| 250 | 250 | 1 | 0 / 0 / 0 | 2 |
| 1,000 | 1,000 | 6 | 0 / 0 / 0 | 2 |
| 4,000 | 4,000 | 45 | 0 / 0 / 0 | 2 |

The 4,000-trial rate (1.125%) agrees with the reported full-size rate
(443/40,000 = 1.1075%); the shorter prefix happens to have fewer events.

Trial 121 (zero-based) resets at step 3950. Its original attempt entry spikes
at 3966 and 4071; READY spikes **25 times**, first at 5152 and last at 6074.
Storage's last observed spike is 4109. The failure here is repeated READY,
rather than a surviving storage train. An entry doublet multiplies through the
excitable recovery chain. This is **a real circuit defect (case 2)**, also
violating the test's independent bounded-entry check. Replacing the assertion
with a nonzero allowance would hide the defect. The recorded finite screen
and the intentional fail-stop corners do not justify doing so.

## B: actual noisy staged reload

`test_10000_verified_actual_noisy_reload_paths[True]` runs the actual one-bit
Q/M staged cell with rate-conditioned readers, adverse slow-three-sigma rails
and independently fast recovery/quiet/READY nodes. It observes two commits,
loads the same Q rail at its observed READY + 1 step, and counts DONEs, missing
Q READY, faults, and master spikes within 100 steps of M READY. Its assertion
requires **all four error fields to equal zero**. The reported `0 == 6` compares
zero `faults` with **`survivors = 6`**; six is an observed error count, never an
expected event count. This counts READY observations with recent master
activity, not six distinct trials.

The six observations all come from **trial 9,163** (zero-based). The seed-108
prefix through trial 8,999 is clean. Replaying just trials **9,000–9,249**
reproduces `failures=ready_missing=faults=0`, **survivors=6**. The helper's new
`trial_offset=9000` consumes the earlier random draws without simulating their
trials; it preserves the original 250-trial batching and exact stray stream.
No seed or corner was changed to manufacture the failure.

Trace of trial 9,163:

| Event | Steps |
|---|---|
| M reset entry | 6779, once |
| First recovery relay, `M.ready_delay0` | 6815, 6937 |
| Next recovery relay | 6855, 6960, 7033 |
| First quiet certificate output | 7915, then a burst through 8290 |
| M READY | 7955, 7993, 8029, 8064, 8098, 8132, 8166, 8200, 8234, 8267, 8311, 8357 |
| Selected COPY arm | 8083 |
| First newly copied M rail spike | 8139 |
| READYs counted as `survivors` | 8166, 8200, 8234, 8267, 8311, 8357 |

The old M rail's last spike was 6841: the **first** READY really followed a
clear. A doublet originating inside recovery, rather than at the reset entry,
multiplied along the chain. Later READYs overlap the intentionally reloaded
master. Thus the six counts are not six failed clears or six failed
transactions; they expose repeated, unsafe READY outputs. The helper's minimum
COPY-after-READY metric also falls to 37 steps because each repeated READY
overwrites its timestamp. This is **a real circuit defect (case 2)**. Ignoring
subsequent READYs in the observer would hide the same defect that A explicitly
checks. The zero-error assertion is retained.

## Compatibility and correction

The recorded prototype remains the default **version 1** of both
`add_reset_verification` and `verify_register_resets`. Callers must explicitly
request `version=2` to select the correction. The slow primitive gate selects
that revision, as does the slow actual-reload gate; their zero-error assertions
remain intact. Existing version-1 screens remain tested, including their known
doublet counterexamples.

Version 2 adds immediate local 4.4-loop inhibitory feedback to the pulse/timer
relays, using the same mechanism as the existing transaction edge primitive.
Feedback removes the residual excitation that lets a stray or another input
produce an extra spike. Entry feedback alone fixes A's observed trigger
doublets but leaves B's six counts unchanged, since its doublet starts inside
recovery. Every equivalent timer link therefore receives the same correction:
four attempt entries, 60 recovery relays, 16 quiet links, 28 deadline links,
and READY. This is **109 local edges, zero new neurons**; no extra controller,
state latch, reset-strength change or longer timing window is introduced.
The ordinary reset taps/inhibitor, BUSY, exhaustion latch, fan-out, mirrored
reset gains, delays and retry bound are preserved. The returned verification
metadata records the chosen version. Cost is 104 neurons and D + 261 + F
synapses for v2, versus D + 152 + F for v1.
As required for local feedback by the mirroring artifact documented at
`docs/stable_latch.md:43–50`, only these local edges suppress mirroring; the
registry is restored immediately, and genuine reset inhibition still mirrors.
A regression checks both properties.

In the same 250-trial reload window, final version 2 gives all four error
counts **zero**, with minimum COPY-after-READY **138 steps**. Both full-size
slow tests explicitly select version 2 and retain their original zero-error
assertions. Short regression probes inject a doublet at either an attempt
entry or inside recovery; they retain the v1 counterexamples and require a
single, safe READY from v2. The captured/corner, forced-survivor exhaustion and
ten-trial actual reload tests exercise both versions.

Final v2 primitive measurement at 4,000 trials: READY=4,000, false_ready=0,
max_attempt_spikes=1, no exhaustion/undecided/terminal survivors, and 3,703
trials needing a retry. The minimum observed certificate window is 48.0 ms
versus a maximum observed storage period of 3.9 ms. Thus the original safety
and attempt-bound assertions pass without requiring every trial to succeed
without retries or conflating fail-stop with successful clearing.

No default, timed fallback, recorded autapse campaign, or machine netlist is
changed. The default/explicit-version-1 ordered one-bit image retains its
pre-change SHA-256 fingerprint
`b9d14d6373ff23444032aa0f4a78600858fb5a2f7b5c7ea6f115a4ea0e4159d5`.
The public verified option remains rejected: these corrections do not address
the documented full-domain datapath or stage READY/IDLE sequencing blockers.

## Reproduction and cluster confirmation

Always set `PYTHONPATH="$PWD"`: the shared virtualenv's editable install points
at the main checkout. This reduced primitive command reproduces the old errors
and measures the corrected revision with the same seed/corner:

```sh
PYTHONPATH="$PWD" python -u -c '
import sys
sys.path.insert(0, "tests")
from test_verified_reset import _verification_survey
for version in (1, 2):
    print(version, _verification_survey("fast3sigma", 1000,
          controller="fast_bias", noisy=True, seed=108, version=version))
'
```

The reduced reload reproduction runs only the failing 250-trial window. It
consumes the skipped prefix's random draws without scheduling events or running
RefSim steps for those trials:

```sh
PYTHONPATH="$PWD" python -u -c '
import sys
sys.path.insert(0, "tests")
from test_robust_reset import _path_trials
for version in (1, 2):
    print(version, _path_trials(250, rate=True, verified=True, seed=108,
          verification_version=version, trial_offset=9000))
'
```

Full-size confirmation for the conductor (quote the node IDs):

```sh
PYTHONPATH="$PWD" python -m pytest -q -o addopts='' \
  'tests/test_verified_reset.py::test_40000_verified_reset_survey[fast3sigma-fast_bias]' \
  'tests/test_verified_reset.py::test_10000_verified_actual_noisy_reload_paths[True]'
```

The full slow suite is a separate cluster workload:

```sh
PYTHONPATH="$PWD" uv run pytest -q tests/test_verified_reset.py tests/test_robust_reset.py
```

The literal `uv run` command initially failed before collection because the
sandbox denies access to `~/.cache/uv/sdists-v9/.git`. Setting `UV_NO_SYNC=1`,
`UV_PROJECT_ENVIRONMENT` to the existing shared virtualenv and `UV_CACHE_DIR`
to task-local scratch makes `uv run` work. Runs also disable bytecode, use a
task-local TMPDIR/basetemp, and disable pytest's cache provider.
Before correction, the two files' non-slow tests passed (126 passed,
50 deselected). With the final relay correction, the combined non-slow run
passes **149 tests, 50 deselected** (249.17 s). The subsequently added local
mirroring regression and eight related tests pass in a focused run (**9
passed**, 1.76 s). The final non-slow collection therefore contains 150 tests.
All **186 build-option tests** pass, including the thirteen existing
default/machine ordered fingerprints. `git diff --check` passes. Full-size
cluster confirmation was obtained in G2 jobs 4020997 and 4020998 and Juno jobs
459426, 459427 and 459428; neither this fix nor a finite survey releases
verified reset.
