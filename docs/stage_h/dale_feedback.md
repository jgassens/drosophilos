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

At `stall_diag`'s top-level `CAMPAIGN` level, a `dale_private` record without
`dale_feedback_version` defaults to v1; an explicitly supplied unsupported
version is refused. A `dale_biased` record must include
`dale_biased_version`; a missing or unsupported version is refused.
`h1_fullgraph` mapping metadata requires the version for either non-default
feedback policy. `kernel_campaign` CLI and `rebuild_record`, `stall_diag`,
`h1_fullgraph` mapping metadata, and `h_stage_place` forward and version-check
`dale_biased_version`. The plumbing regression tests cover these paths.
Historical autapse records remain unchanged.

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

## Bounded fast-feedback round, 2026-10-15

**Verdict: the numerical pass bar is met, but the timing goal is not achieved.**
Three fixed constructions were measured, with no parameter grid or subsequent
retuning. The score winner is exposed as `latch_feedback='dale_biased'`, version
1. It has zero sampled corner failures and 118/128 noisy +700 reloads, exceeding
the requested 114/128. Its inhibition is sparse and asynchronous, however, not
one inhibitory arrival per member spike at autapse time. Calling this circuit
`dale_fast` would misrepresent the result. None of the three constructions both
reproduced autapse timing and passed the numerical bar. This round stops here;
the new opt-in is an experimental holding/reload result, not a replacement for F.

### Three constructions and the shared scoring fixture

1. **Biased private relay (selected).** Each member drives its own inhibitory
   neuron with `+round(.2*loop)` q, delay 0; the return is
   `-round(.2*loop)` q, delay 0. The relay has a tonic bias of +5.6 mV,
   leaving a nominal 1.4-mV resting gap. Even the opposing threshold/bias
   three-sigma corners leave a positive .2-mV gap. Scaling input with the
   reduced gap was intended to limit bursting. The fixed reset voltage and
   refractory law do not scale with that gap: measured relay firing is instead
   slower than member firing. Bias also does not initialize the membrane near
   threshold; RefSim still starts at E_L. This is why the proposed same-step
   pre-bias mechanism did not materialize in this construction.
2. **Partner-driven relay.** The relay that inhibits `u` receives `+1*loop`
   from `v`, and the relay that inhibits `v` receives it from `u`; both input
   and return delays are zero, and return gain is .2. Anticipation sometimes
   lands before the target spike, but the independent relay dynamics also
   suppress weak handoffs. It performs much worse on both scores.
3. **Inhibitory twin.** For each member, clone all its incoming netlist edges
   and bias onto an inhibitory neuron, then give that neuron a -.2-loop
   inhibitory self-edge and an identical return to the member, both delay 0.
   Incoming ordinary reset edges are cloned too; feedback mirroring is bypassed.
   The twin's own self-edge is Dale-consistent because all its outputs are
   negative. Equal inputs and state can synchronize twin and member so the
   twin delivers inhibition at the member's spike time. Host ignition events
   into the member are not netlist edges and cannot be cloned by this transform;
   startup differs. Independent membrane and edge variation breaks the nominal
   match and causes corner and reload failures. This is retained only as a
   test-local rejected construction.

All new feedback sources have a single transmitter sign. The two rejected
constructions remain in `tests/test_dale_fast.py`, not in the builder API.
Existing mixed-sign control elements elsewhere in F are unchanged.

The full-size command below reruns the **same** `corner_probe` and
`noisy_reload_probe` from `tests/test_dale_feedback.py` for both baselines and
all three candidates. Only feedback installation is dispatched differently;
the old baseline fixtures, assertions and frozen digests are unchanged. Corners
are the original 256 sign patterns, seed 109, each repeated with the two ignition
regimes (512 cases), 300-q kick, 6,000 steps, and the original hold estimator.
Reload is the original mix B, seed 108, 128 copies, four ignitions, kick,
phase-dependent clear and 6,200 steps. Success requires **both members and the
reader** to spike in the final 200 steps. All added neurons receive the same
independent noise law, including strays; topology sizes change RNG assignment,
as in the original comparison. No artificial inhibitory events or simulator
changes are used.

| Circuit | Corner non-ignitions / holding losses / terminal dark (512) | Corner minimum interval | +700 reload (128) | +1000 reload (128) | No-reload survivors (128) | Numerical pass bar |
|---|---:|---:|---:|---:|---:|---|
| Autapse control | 0 / 0 / 0 | 36 | 114 | 128 | 0 | Pass |
| dale_private v1 control | 41 / 43 / 84 | 37 | 102 | 128 | 0 | Fail |
| **dale_biased v1** | **0 / 0 / 0** | **33** | **118** | **128** | **0** | **Pass** |
| Partner relay | 112 / 60 / 172 | 37 | 3 | 96 | 0 | Fail |
| Inhibitory twin | 67 / 34 / 101 | 36 | 71 | 118 | 0 | Fail |

The numerical bar is zero failures in all three corner columns and >=114/128
at +700. These are bounded aggregate counts, not a significance claim. The
selected circuit still fails the separate 44-step rate floor, more severely
than the autapse; passing the requested score does not qualify rate readers or
all configuration F workloads.

### Measured timing, including the negative result

Nominal measurements use the unchanged RefSim, ignition at step 10, 6,000
steps, and inspect actual member and inhibitory spikes. Zero-delay deliveries
affect membrane integration on the following step for all constructions.

| Construction | First u spike | First inhibition delivery to u | Difference |
|---|---:|---:|---:|
| Autapse | 34 | 34 | 0 |
| dale_private | 34 | 69 | +35 |
| dale_biased | 34 | 177 | **+143** |
| Partner relay | 34 | 134 | +100 |
| Inhibitory twin | 34 | 140 | +106 |

For the selected circuit, u starts `[34,114,175,235]` and its relay starts
`[177,311,438,562]`. After step 2,000 the median member interval is **52**
steps and relay interval **122** steps, about one inhibitory spike per 2.4 member
spikes. This is neither same-step feedback nor a fixed phase relationship.
The nominal dark-latch control, including its biased relays, remains silent.
The twin does achieve same-step member/relay spikes after settling (58-step
intervals on both), but its initial u spike is unmatched and its noisy/corner
results reject it. No measured construction satisfies the full timing goal.

### API, replay and preserved circuits

The new production constants in `protocol/latch.py` are
`DALE_BIASED_VERSION=1`, `DALE_BIASED_BIAS_MV=5.6`,
`DALE_BIASED_DRIVE_GAIN=.2`, `DALE_BIASED_GAIN=.2`, and
`DALE_BIASED_INPUT_DELAY_STEPS=DALE_BIASED_DELAY_STEPS=0`.
`add_dale_latch_feedback(net, drive, variant='biased')` installs this circuit.
The existing default `variant='private'` and all its v1 constants retain their
original behavior, roles, insertion order, weights, delays and zero biases.
Both builders accept `latch_feedback='dale_biased'` with
`experimental_autapse=True` and its existing restrictions. The kernel rejects
`master_ignition`, `operand_xor_oneshot`, and `f_width16_experimental` with
either Dale policy, before building the netlist.

New builds record `latch_feedback='dale_biased'` and `dale_biased_version=1`;
the independent `dale_feedback_version` still belongs to private v1. An explicit
biased version must be integer 1 (not bool); missing version is refused, and an
irrelevant policy's version is ignored. Tests rebuild nested Stage D records,
compare ordered netlist arrays **and biases**, and check rejection of changed,
null, string and boolean versions in both diagnostic rebuild and record recheck.
Use nested `build_options` for this variant. Legacy top-level campaign replay
routes do not consistently forward its new version key; full support there
requires edits outside the owned paths of this round.

The four pre-existing frozen default/autapse digests remain unchanged and pass.
Private v1 F digests were captured before this round's production edits and
are now also frozen in `tests/test_dale_fast.py`:

| Private v1 build | SHA256 |
|---|---|
| Two-cell 2-bit ADD → XOR kernel | `894d6d70ec86691e1c215a2f3826e9e4d1c6e003a860cc0c211a58067b8189c4` |
| 4-bit ordered flip-flop adder, watchdog 100 | `133acfac07d2c027890b8b8024991fba615b4fdafb1fc98b05f7a24c1b6fe8f1` |

### Isolated F adder and costs

The exact requested builder is used with `latch_feedback='dale_biased'`:
`build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100,
storage='flipflop', rate_robust=True, experimental_autapse=True,
ordered_xor_oneshot=True, latch_feedback='dale_biased')`.
All policies reran H1 `simulate_channel` in isolation using
`adder_cases(4,8,seed=110)`, once fresh per addition and once chained. This is
nominal arithmetic, not anatomical/noisy qualification.

| Policy | Fresh correct / completed (8 attempted) | Chained correct / completed (8 attempted) | Wrong / faults / timeouts / no accept / missing completion (each mode) | Fresh / chained spikes |
|---|---|---|---|---:|
| Autapse | 8 / 8 | 8 / 8 | 0 / 0 / 0 / 0 / 0 | 141,855 / 141,955 |
| dale_private | 8 / 8 | 8 / 8 | 0 / 0 / 0 / 0 / 0 | 176,162 / 177,139 |
| dale_biased | 8 / 8 | 8 / 8 | 0 / 0 / 0 / 0 / 0 | 169,476 / 169,779 |

Biased accept times in ms are `[412,377,377,354,413,370,354,370]` fresh and
`[412,380,382,358,413,374,357,373]` chained. Relative to the autapse these are
-2..0 ms fresh and -3..+1 ms chained. Spikes increase by 27,621 fresh and
27,824 chained versus autapse, and decrease by 6,686 and 7,360 versus private.

| Build | Autapse neurons / edges | dale_private neurons / edges | dale_biased neurons / edges | Biased extra vs autapse | Biased extra vs private |
|---|---:|---:|---:|---:|---:|
| 4-bit ordered F flip-flop adder, watchdog 100 | 740 / 1,518 | 880 / 1,658 | 880 / 1,658 | +140 / +140 | 0 / 0 |
| Two-cell 2-bit ADD → XOR F kernel | 2,478 / 4,918 | 2,992 / 5,432 | 2,992 / 5,432 | +514 / +514 | 0 / 0 |

There is one extra neuron and one extra edge per member versus autapse, plus
one biased membrane parameter per relay. Private and biased have equal counts.

### Reproduction for this round

From the worktree, explicitly select its Python sources:

```
PYTHONPATH=$PWD uv run pytest -q -o addopts='' tests/test_dale_fast.py tests/test_dale_feedback.py
PYTHONPATH=$PWD DROSO_DALE_FAST_SURVEY=1 uv run pytest -s -q -o addopts='' -m slow tests/test_dale_fast.py
```

The second command is the **full-size 512-corner / 128-copy reload comparison**
for both baselines and all three candidates, behind the repository's slow marker
and an environment flag. It was run; the first version of the screen took 11.2
seconds on this laptop. The default new tests use only 32 corner cases, nominal
timing, replay/topology checks and eight fresh/eight chained additions; existing
private surveys remain unchanged. Local invocation also sets
`UV_PROJECT_ENVIRONMENT=/Users/jeremiahgassensmith/programming/drosophilos/.venv`,
`UV_NO_SYNC=1`, and a writable `UV_CACHE_DIR` under the task's temporary directory.
The original 2,000-copy / multi-seed slow survey was not rerun. No larger
qualification or further tuning is claimed by this bounded round.

## Downstream fanout evidence and bounded bias round, 2026-10-20

**Verdict: no setting meets all three bars. Stop tuning after these three
settings.** The least-failing candidate is available only for reproduction as
`latch_feedback='dale_refractory'`, version 1. It has **4 non-ignitions / 11
holding losses / 15 terminal-dark copies** on the same 512 corners, **114/128**
noisy +700 reloads, and a **33-step** corner minimum interval. It fails both
the zero-corner-failure bar and the >=44-step rate floor. It is a rejected
experimental circuit, not a qualified replacement for F or an improvement on
dale_biased v1. No fourth setting or further tuning was attempted.

### G2 fanout result supplied by the conductor

The downstream G2 run on **2026-10-16** used the 8-bit `kernel_campaign` block
`fanout`, `--copies 100 --mix B --seed 110 --repeats 40 --max-ms 500000
--rate-robust --experimental-autapse --ordered-xor-oneshot`. The two builds
produced:

| Build | G2 job | OK outputs / expected | Wrong | Missing | Faults | Timeouts | Copies stopped |
|---|---:|---:|---:|---:|---:|---:|---:|
| Autapse | 4023612 | 32,000 / 32,000 | 0 | 0 | 0 | 0 | 0 / 100 |
| dale_biased v1 | 4023613 | 27,336 / 32,000 | 0 | 4,664 | 1 | 7 | 8 / 100 |

Source files (reported evidence, not rerun in this laptop round):

```
/Users/jeremiahgassensmith/programming/drosophilos/data/stage_h/dale/fanout_autapse_s110_r40.json
/Users/jeremiahgassensmith/programming/drosophilos/data/stage_h/dale/fanout_dale_biased_s110_r40.json
```

The local holding/reload score did not predict downstream liveness. In this
run the problem manifests as stalled copies and missing outputs, with no wrong
answers among delivered outputs. This is consistent with a rate/timing
problem; the aggregate counts alone do not isolate a causal circuit path.

### Exactly three measured settings

The owned netlist API carries per-neuron bias but no per-neuron refractory
override. All three settings therefore retain the unchanged simulator's
ordinary **22-step refractory period**, with one private inhibitory relay per
member, zero input/return delays, a lower **5.0-mV bias**, and an increased
**0.4-loop excitatory drive**. Only the inhibitory gain differs. The three
settings were fixed before measurement. The aim was to make inhibition more
frequent and increase slowing without adding a simulator feature or changing
any member, loop edge, reader, controller, or other policy.

Measurements call the original `corner_probe` and `noisy_reload_probe` from
`tests/test_dale_feedback.py`: seed 109, 256 independent sign vectors repeated
under two ignition regimes (512 cases), 6,000 steps and the same kick/holding
estimator; full mix B, seed 108, 128 copies, original clear phases, 6,200 steps
for reload. Added relays receive independent membrane/edge corners and the
same noise law, including strays. Different topology sizes change RNG draw
assignment. Intervals use the original settled estimator starting at step
2,000, including the step-2,520 kick. These are sampled, not exhaustive corners.

| Circuit / inhibitory gain | Corner non-ignition / holding loss / terminal dark (512) | Corner interval min / median / max | +700 reload (128) | +1000 reload (128) | No-reload survivors (128) | All three bars |
|---|---:|---:|---:|---:|---:|---|
| Autapse control | 0 / 0 / 0 | 36 / 54 / 98 | 114 | 128 | 0 | Fail rate floor |
| dale_biased v1 control | 0 / 0 / 0 | 33 / 50 / 91 | 118 | 128 | 0 | Fail rate floor |
| **Candidate 1, .2 (dale_refractory v1)** | **4 / 11 / 15** | **33 / 54 / 143** | **114** | **128** | **0** | **Fail corners and rate floor** |
| Candidate 2, .3 | 76 / 83 / 159 | 35 / 58 / 139 | 108 | 128 | 0 | Fail all three |
| Candidate 3, .4 | 217 / 70 / 287 | 37 / 62 / 176 | 37 | 65 | 0 | Fail all three |

The selection prioritizes corner survival and then reload, with interval as
the remaining criterion: candidate 1 loses the fewest copies and alone meets
the reload bar. It is the least-failing of these three, not a passing design.
Increasing inhibition raised the minimum interval only modestly while greatly
worsening holding and reload. Silent copies remain in the holding denominator;
15, 152 and 277 copies respectively have no settled interval samples. Silence
must not be counted as satisfying a rate floor.

### Opt-in circuit, replay and frozen baselines

The new constants are `DALE_REFRACTORY_VERSION=1`,
`DALE_REFRACTORY_BIAS_MV=5.0`, `DALE_REFRACTORY_DRIVE_GAIN=.4`,
`DALE_REFRACTORY_GAIN=.2`, and
`DALE_REFRACTORY_INPUT_DELAY_STEPS=DALE_REFRACTORY_DELAY_STEPS=0`.
The name identifies this bounded refractory/bias investigation; it does **not**
claim a longer refractory period. `add_dale_latch_feedback(...,
variant='refractory')` retains the private Dale-compatible structure and bypasses
feedback mirroring while preserving genuine later clears. Unignited nominal
latches and relays remain dark for the tested 6,000 steps.

Both builders record `latch_feedback='dale_refractory'` and
`dale_refractory_version=1`. Explicit versions must be integer 1, excluding
booleans. Other policies ignore this version. New builds may omit the version;
campaign and diagnostic replay and saved mapping contracts require it. The
kernel refuses `master_ignition`, `operand_xor_oneshot`, and
`f_width16_experimental` with this policy, as with dale_biased. Existing
experimental-autapse/rate-robust/F shape restrictions still apply.

The policy/version are forwarded through `kernel_campaign` CLI, records and
`rebuild_record`, `stall_diag` (tick, non-tick and nested Stage D),
`h_stage_place` CLI/metadata/building, `h1_fullgraph` builder metadata and
`embed_image.MAPPING_META_KEYS`. CLI help states the measured rejection.
Tests save and reload a complete mapping, then compare ordered roles, edges
and biases with the original channel. Nested Stage D diagnostics reject a
missing version, and both diagnostics and `stage_d.recheck_record` reject
explicit malformed/changed versions. As with dale_biased, direct
`stage_d.recheck_record` still defaults a missing nested version to v1 through
the builder; that entry point is outside this round's owned files.

Default, autapse, private v1 and biased v1 circuits remain unchanged. Existing
frozen autapse/default/private digests are retained verbatim. New biased v1
digests were independently captured by executing HEAD's unedited latch
transform; these include ordered `(src,dst,quanta,delay,roles,bias)`:

| Biased v1 build | SHA256 including bias |
|---|---|
| Two-cell 2-bit ADD → XOR F kernel | `88eab29d4e817312ead92ede0560c6cb4103d312bc6d22be04b17eb922cf1078` |
| 4-bit ordered F flip-flop adder, watchdog 100 | `847e799023d3ba993a8c36cd21d048bf978ba56886e83c09473fdc5b9479e5fe` |

### Isolated arithmetic and fanout smoke

The original H1 isolated F flip-flop adder fixture, `adder_cases(4,8,seed=110)`,
was run fresh and chained. The new variant completed **8/8 correct** in each
mode, with zero wrong answers, faults, timeouts, no-accept or missing completion
events. Fresh accept times were `[413,379,378,355,414,372,355,372]` ms; chained
times were `[413,384,383,357,414,376,358,375]` ms. Total spikes were **171,307**
fresh and **171,491** chained. These nominal isolated successes do not qualify
noisy kernels or erase the failed holding/rate bars.

The 8-bit `kernel_campaign.block('fanout')` smoke used full mix B with
`_NoisyRefSim`, seed 110, **2 copies × 2 repeats**, the same F flags and
`max_ms=50000`. Both copies completed: **32/32 outputs correct, 0 wrong,
0 missing, 0 bad decodes, 0 faults and 0 timeouts**. A preliminary 15,000-ms
observation ended with nine correct outputs per copy; increasing only the
observation ceiling allowed completion. This uses the same noise law as the
campaign, not its CUDA random stream. Passing this tiny smoke does not predict
the 100-copy × 40-repeat liveness result or reverse the failed corner/rate bars.

### Reproduction and full cluster command

Run from this worktree, selecting its sources and the shared environment:

```sh
export PYTHONPATH=$PWD
export UV_NO_SYNC=1
export UV_PROJECT_ENVIRONMENT=/Users/jeremiahgassensmith/programming/drosophilos/.venv
export UV_CACHE_DIR=/Users/jeremiahgassensmith/.agentctl/runs/dale-refr-20261020/tasks/dale-refractory/tmp/uv-cache
uv run pytest -q -o addopts='' tests/test_dale_refractory.py tests/test_dale_plumbing.py tests/test_dale_fast.py tests/test_dale_feedback.py
DROSO_DALE_REFRACTORY_SURVEY=1 uv run pytest -s -q -o addopts='' -m slow tests/test_dale_refractory.py
```

The default new suite measures all 512 corners and 128 reloads side by side
for autapse, dale_biased and the selected variant. The gated slow test reruns
the complete three-setting screen and both controls; it was run in this round.
The existing larger private and fast-feedback surveys were not rerun.
Validation: the combined required suite passed **96 tests, 3 skipped** in
187.29 seconds; the gated three-setting screen passed in 8.88 seconds.
All existing frozen default/autapse/private digests and both new frozen
biased-v1 digests passed. `git diff --check` was clean.

For the conductor on G2, the exact full fanout command from the repository
root is below. It is supplied for reproduction of this rejected candidate,
not as a recommendation to promote it. **The 100-copy × 40-repeat run was
not run on this laptop.**

```sh
PYTHONPATH=$PWD UV_NO_SYNC=1 uv run python -m drosophilos.bench.kernel_campaign fanout --copies 100 --mix B --seed 110 --repeats 40 --max-ms 500000 --rate-robust --experimental-autapse --ordered-xor-oneshot --latch-feedback dale_refractory --device cuda --out data/stage_h/dale/fanout_dale_refractory_s110_r40.json
```
