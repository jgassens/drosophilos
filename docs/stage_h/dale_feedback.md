# Private Dale-compatible latch feedback, version 1

**Verdict: useful local slowing, but not a replacement for configuration F's
autapse.** The private circuit matches the nominal 58-step storage period and
passes the bounded mix-B ROM kernel and isolated H1 additions below. It fails
holding at sampled independent three-sigma corners where the autapse holds:
41 non-ignitions and 43 losses of established hold in 512 cases. Noisy reload
at +700 steps also succeeds less often (102/128 versus 114/128). Both circuits
fail the 44-step floor. This is an opt-in experimental circuit and a negative
result for reproducing all of the autapse's benefit; defaults remain unchanged.

## Circuit and API

For each excitatory latch member `m`, install one dedicated inhibitory neuron
`<member role>.fb`, e.g. `R.b0r0.u.fb` and `R.b0r0.v.fb`:

```
m -- +3621 q, delay 0 --> m.fb -- -724 q, delay 0 --> m
```

The original two excitatory loop edges and all readers/controllers retain their
weights, delays and order. Each feedback neuron has one excitatory input and
one inhibitory output, aimed only at its own member. The construction adds no
cross-member or cross-register inhibition. It adds two neurons/four edges per
latch, replacing two self-edges. The role names expose the inhibitory neurons
to placement and sign audits. This fixes the added latch feedback's transmitter
sign conflict, not every mixed-sign source elsewhere in the machine.

`add_dale_latch_feedback(net, drive, variant='private')` recognizes exactly the
same role-named, mutual `drive.loop` pairs as the existing autapse transform.
It rejects an unsupported variant or an already modified latch before mutation.
Builders call it last, after consumers, controllers and compact resets. Local
inhibition temporarily bypasses `inhibition_mirrors`, so it never suppresses a
reader or required-rail qualifier. The mirror registry is restored for genuine
later clears. As with autapses, installing new readers afterward is invalid.
Feedback neurons receive no separate reset: stopping their sole driving member
stops them after their remaining current drains; the clear probes exercise this.

The versioned constants in `protocol/latch.py` are:

| Constant | Value |
|---|---:|
| `DALE_FEEDBACK_VERSION` | 1 |
| `DALE_FEEDBACK_DRIVE_GAIN` | 1.0 × loop |
| `DALE_FEEDBACK_GAIN` | 0.2 × loop, inhibitory |
| `DALE_FEEDBACK_INPUT_DELAY_STEPS` | 0 |
| `DALE_FEEDBACK_DELAY_STEPS` | 0 |

Both `build_pipeline` and `build_adder_channel` accept
`latch_feedback='autapse'` (historical default) or `'dale_private'`. Private
feedback requires `experimental_autapse=True`, including its existing
`rate_robust=True` and per-shape/F restrictions. The flag name remains for
compatibility; private feedback replaces the latch autapses rather than adding
both. Both builders record `latch_feedback='dale_private'` and
`dale_feedback_version=1` in `build_options`. The kernel's historical autapse
records retain exactly their old fields; absent policy means autapse, absent
private version means version 1. An explicitly supplied private version must
be an integer equal to 1; `None`, strings, booleans and other versions reject.
Versions irrelevant to the selected autapse circuit are ignored.

Stage D's **nested `build_options`** already forward these options to the kernel:
diagnostic rebuilding reproduces the private topology, and rebuilding/recheck
reject a changed private version. Regression tests exercise both. Historical
records still rebuild unchanged, as covered by the existing autapse tests.

**Remaining integration gap:** `bench/stall_diag.py`'s legacy top-level
`kernel_campaign` paths enumerate options explicitly and do not forward the
new keys. In particular, a private top-level tick record can silently rebuild
the autapse circuit, and its private version is not checked. Do not use those
paths for private records. Forwarding/validating both keys there, and adding
CLI exposure if wanted, requires edits outside this task's five owned paths.
The conductor must integrate that before claiming all campaign replay paths
support private feedback. Direct builders and nested Stage D records work.

No shared variant was added. A group inhibitor would let one active latch
inhibit other stored bits, violating the private circuit's independence; it
does not address the measured private circuit's corner failures cheaply.

## Selecting constants and measuring latency

The bounded screen used the unchanged float64 RefSim, 64 reader-equipped
latches, mix B, seed 109, 6,000 steps. It tried relay drives
`(1, 1.8/1.4, 1.6) × loop` crossed with inhibitory gains `(.15, .2, .25)`.
Intervals are measured after step 2,000. Counts include every copy.

| Relay drive / loop | Inhibitory gain | Median interval | Non-ignitions | Lost established hold |
|---|---:|---:|---:|---:|
| 1 | .15 | 55 | 0 | 0 |
| 1 | .20 | 59 | 0 | 0 |
| 1 | .25 | 66 | 0 | 0 |
| 1.8/1.4 | .15 | 61 | 0 | 0 |
| 1.8/1.4 | .20 | 74 | 0 | 1 |
| 1.8/1.4 | .25 | 100 | 9 | 9 |
| 1.6 | .15 | 66 | 0 | 0 |
| 1.6 | .20 | 108 | 9 | 20 |
| 1.6 | .25 | 162.5 | 54 | 10 |

The same 64-copy autapse control has median interval 57. Auxiliary `.1`-gain
checks gave medians 52 (drive 1) and 54 (drive 1.8/1.4). At drive 1, nominal
intervals were 52/.1, 55/.15 and 58/.2; the autapse is 58. We selected drive 1,
gain .2 for the exact nominal period match, unchanged inhibitory quanta, and
absence of holding losses in the screen. This is the best screened setting by
those criteria, not an optimized corner solution or a global search result.
The independent-corner failure below ended tuning.

Zero synaptic delays do **not** make this an instantaneous autapse. From rest,
the first member spike is at step 34 and its feedback neuron spikes at 69:
inhibition is delivered 35 steps (3.5 ms) later than the autapse. In RefSim both
deliveries affect integration on the following step. At settled nominal hold
(after step 2,000), feedback spikes lag `u` by 2 steps and `v` by 1 step; each
feedback neuron emits one spike per member spike. Residual synaptic current
means this settled phase lag is not the isolated-pulse latency. Initial `u`
spikes are `[34,118,189,255,317,377]` for autapse and
`[34,120,197,269,336,400]` for private feedback; the startup transient is slower.
Both eventually have 58-step intervals (5.8 ms, about 172.4 Hz). These latency
and nominal-period measurements are asserted in the tests. No artificial
inhibitory arrival sequence, altered threshold/refractory law, or host feedback
is used.

## Bounded surveys

All results below use RefSim and the same named seeds for both circuits. Mix B
means independent log-weight sigma .04, threshold/bias sigma .2 mV, and 5 Hz
Bernoulli 150-q strays on every neuron, including the added inhibitory neurons.
Different topology sizes shift random draw assignment: these compare aggregate
results, not paired trajectories or the CUDA random bitstream.

`period_probe` reuses the stable-latch holding and interval estimators. Holding
requires two spikes in [500,1000); a subsequent gap over 200 steps, including
terminal silence, counts as loss. Rate intervals start at 2,000. Clear/recovery
probes reuse `test_stable_latch.py`'s actual four-tap .75-loop controller, rate
reader and nominal 15-phase/25-step recovery grid. Clear trials have four close
ignitions, the 2,520-step kick, trigger at 3,000–4,499, and require live storage
in the preceding 200 steps. Survivors fire more than 1,500 steps after trigger.
The master74 fixture keeps captured old parameters and samples new feedback
edge weights/reader parameters; the new relay's membrane parameters are nominal
in that conditional capture probe. Mix B samples all neurons independently.

| Probe | Autapse | dale_private |
|---|---|---|
| Nominal settled interval | 58 | 58 |
| Mix B, seed 109, 128 × 6,000: interval min / median / max | 50 / 57 / 69 | 52 / 59 / 94 |
| Same: non-ignition / holding loss / terminal dark | 0 / 0 / 0 | 0 / 0 / 0 |
| Additional 2,048 × 6,000, seed 109: interval min / median / max | 49 / 57 / 72 | 49 / 59 / 98 |
| Same: non-ignition / holding loss / terminal dark | 0 / 0 / 0 | 0 / 0 / 0 |
| Sampled independent three-sigma signs, seed 109, 512 cases: minimum interval | **36, fails ≥44** | **37, fails ≥44** |
| Same: non-ignition / holding loss / terminal dark | 0 / 0 / 0 | **41 / 43 / 84** |
| Ordinary clear, mix B seed 108, 128: storage / reader survivors | 0 / 0 | 0 / 0 |
| Ordinary clear, master74 seed 108, 128: storage / reader survivors | 0 / 0 | 0 / 0 |
| Ordinary clear, fast three-sigma, seed 108, 128: storage / reader survivors | 0 / 0 | 0 / 0 |
| Non-live before clear, each of those three scenarios | 0 | 0 |
| Nominal recovery / READY / margin, steps | 700 / 872 / 172 | 700 / 872 / 172 |
| Nominal no-reload control survivors, 15 phases | 0 | 0 |
| Noisy reload +700, seed 108, 128: storage **and** reader holding at end | 114/128 | **102/128** |
| Noisy reload +1000, same seed/copies | 128/128 | 128/128 |
| Noisy no-reload controls: any storage/reader survivor at end | 0/128 | 0/128 |
| Noisy F 2-bit ROM kernel, seed 110, one copy | 5/5 outputs | 5/5 outputs |
| Same: faults / timeouts / bad decodes | 0 / 0 / 0 | 0 / 0 / 0 |
| H1 4-bit F flip-flop adder, seed 110, fresh | 8/8 correct | 8/8 correct |
| Same words, chained | 8/8 correct | 8/8 correct |
| Adder fresh/chained wrong / faults / timeouts / missing completion | all 0 | all 0 |

The corner probe samples 256 independent sign vectors, repeats each with weak
single ignition and with additional ignitions at 11/14/23, and applies the
300-q kick at 2,520. It includes independent feedback-edge and inhibitory-neuron
threshold/bias corners. It is **not** an exhaustive corner census. Failed
establishment remains in the denominator; 81 private copies have no measured
intervals, which is distinct from the 43 established holding losses.

The noisy reload probe uses full mix B, four ignitions and the kick, clear at
`3000 + 4*(copy % 15)`, and one reload at +700 or +1000. It observes all storage
members and the reader in the final 200 steps of a 6,200-step exposure. Within
each policy, controls use the same noise draws with no reload and prove the old
train stopped. The deterministic recovery threshold cannot guarantee recovery
under noise. Private feedback has 26 failed +700 reloads versus 14 for autapse;
no significance claim or universal recovery bound follows from this sample.

The kernel uses the `rom` shape from `test_config_f_kernels.small_shape`, tokens
`[0,1,3,2,0]`, expected outputs `[3,0,1,2,3]`, max 6,000 ms and full mix B. This
exercises repeated loads and clears, not a Stage D campaign. Adder measurements
use exactly `build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100,
storage='flipflop', **F, latch_feedback=...)` and H1 `simulate_channel` on its
isolated designed topology; fresh uses a new simulator/power-on per addition,
chained uses one simulator. Operands come from `adder_cases(4,8,seed=110)`.
Adder runs are nominal, not noisy anatomical placement measurements.

Fresh accept times in ms, autapse/private respectively:
`[412,379,378,354,413,372,355,372]` /
`[413,379,379,356,414,373,356,373]`.
Chained: `[412,380,381,358,414,375,357,376]` /
`[413,384,383,358,417,377,361,379]`.
The private path adds 0–2 ms fresh and 0–4 ms chained in these cases. It also
increases adder spikes: 141,855 → 176,162 fresh and 141,955 → 177,139 chained.

## Cost and unchanged baselines

| Build | Autapse neurons / edges | Private neurons / edges | Added neurons / edges |
|---|---:|---:|---:|
| Two-cell, 2-bit ADD → XOR F kernel | 2,478 / 4,918 | 2,992 / 5,432 | 514 / 514 |
| 2-bit ROM F kernel | 1,435 / 2,848 | 1,731 / 3,144 | 296 / 296 |
| 4-bit ordered F flip-flop adder, watchdog 100 | 740 / 1,518 | 880 / 1,658 | 140 / 140 |

Frozen SHA256 tests serialize ordered arrays `(src,dst,quanta,delay,roles)` with
compact JSON. Baselines were captured before production edits; the fully
default adder also uses the existing pre-edit frozen digest from
`test_adder_config_f.py`, independently verified by executing HEAD's unedited
builder. Both omitted policy and explicit `'autapse'` must match:

| Build | SHA256 |
|---|---|
| Default two-cell 2-bit ADD → XOR kernel | `a4a12041ed3f14368d21c2b13ccd753a4b1416e49a4d70cc05ffea22efaf81fe` |
| Default 4-bit adder, all builder defaults | `eed4a4a2b40375ff802d81eebd041d4151354e9557ac01776cface380fea6c5c` |
| Today's F two-cell kernel | `6fcd4004647d069f6c19b9a4a90c906bb9d2870646820ef4200b3154f67774f2` |
| Today's F ordered flip-flop adder, watchdog 100 | `9d9ab9eb0113ce036b28e7bbacd67fc29156dccae07a7b766dd055e398705488` |

## Reproduction and scope

Run from the worktree with `PYTHONPATH=$PWD` (the shared editable installation
otherwise imports the main checkout):

```
PYTHONPATH=$PWD uv run pytest -q -o addopts='' tests/test_dale_feedback.py
PYTHONPATH=$PWD uv run pytest -q tests/test_experimental_autapse.py tests/test_adder_config_f.py
PYTHONPATH=$PWD DROSO_DALE_SURVEY=1 uv run pytest -s -m slow tests/test_dale_feedback.py
```

The first command includes the bounded surveys and asserts the measured corner
and reload failures as well as the successful small computations. The last is
opt-in: 2,000-copy stable probes, ROM kernels at seeds 108/109/110 with two
copies, and 50 fresh plus 50 chained additions per policy/seed. It was not run
for this task. The extra 2,048-copy period measurement above was run separately.
The nine-setting screen is reproducible by importing `period_probe` from
`tests/test_dale_feedback.py` with `PYTHONPATH=$PWD:$PWD/tests` and temporarily
setting the two gain constants to the tabulated values; no simulator changes
are needed. Local runs selected the shared venv with `UV_PROJECT_ENVIRONMENT`
and disabled synchronization with `UV_NO_SYNC=1`.

These are bounded mechanism measurements. They do not repeat the 40,000-trial
clear surveys per scenario, exhaustive independent corners, 10-second holding,
all rate gates/required-rail/multi-cell options, or 100-copy Stage D exits in
`stable_latch.md` and `stage_d.md`. The private path's observed corner and
reload failures already rule out claiming the same qualification as F; larger
passing arithmetic surveys would not erase those failures.

`add_transaction_edge` still has its original `-4.4*loop`, zero-delay self-edge.
The same sign construction would give that edge neuron a private inhibitory
feedback neuron, with an excitatory feed and inhibitory return. It would also
add integration latency where the current self-edge suppresses an initial
doublet before ordinary inhibitors arrive. The latch results do not establish
that such a delayed path suppresses that doublet. That circuit is unchanged
and would require its own timing/corner qualification.
