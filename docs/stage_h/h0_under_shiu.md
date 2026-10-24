# H0 under the released Shiu rules, after retuning

**Tier A passes. The same 15 anatomical H0 neurons compute all four AND
transactions under M5 after retuning within the fixed 4× anatomical weight bound.**
Completion latencies are **15.8, 29.8, 29.8 and 33.2 ms**; every transaction decodes
correctly and passes the existing reset check. The largest effective physical-edge
multiplier is **3.6502016129×**. Tier B was not run, as required when Tier A passes.

The [unretuned failure](model_comparison.md) was therefore a failure of stale
tuning, not a demonstration that this H0 embedding cannot work under the corrected
model. This recertifies an externally designed, isolated circuit. It does not show
that the connectome naturally computes AND, or establish performance in a live
surround, across arbitrary input timing, or under biological variation.

The machine-readable inventory is [h0_under_shiu.json](h0_under_shiu.json). It
contains every transaction's role spike times, trace hash, arrival-loss counts,
per-physical-edge anatomical audit, original and retuned policies, and diagnostic
measurements. The real MCNS cache was read successfully through `--mcns-dir`.

## Circuit and protocol

The body IDs, roles, anatomical connections and counts come from the versioned
[H0 result](../h0/results.json), resolved by body ID against the real graph, with
every designed count checked against that graph. No embedding search selects or
replaces the Tier A neurons:

```text
10643 10864 10881 11127 11429 14160 14210 26764
32461 37111 512079 520816 523040 523983 555296
```

Each trial starts at rest. At 5 ms a single external event ignites each selected
operand rail's driver. Both operands arrive together. Observation is 150 ms.
Passing uses `h0_run.run_transaction`'s existing correct dual-rail decode,
completion and `reset_ok`, plus its existing test's completion latency **<60 ms**.
`reset_ok` requires the last loop spike before the final 30-ms quiet tail; this is
the historical check, not a new reset criterion. No external reset pulse is used:
completion drives the existing inhibitory return path.

All biological constants stay at `Params()` defaults: dt 0.1 ms, E_L/reset −52 mV,
threshold −45 mV, tau_m 20 ms, tau_s 5 ms, refractory 2.2 ms, w_syn 0.275 mV and
edge delay 1.8 ms. M5 is exactly `ModelRules(True, True, "v", True)`. Scheduling an
operand event exempts that selected driver from refractory before the run, just as
the released rules specify; its partner, gates and reset neurons retain refractory.

## What changed

Targets now accept `ModelRules`, and the latch probe runs those rules before
deriving rate-mode targets. The single-pulse response remains a synaptic **g**
response: below threshold there is no spike reset or refractory hold, so neither
clearing/freezing g nor the external-stimulus target changes that response. The
single-pulse need stays exactly **2586.0434707264194 quanta**. External V ignition is
a separate path and was retuned separately.

| target | original M0 | M5, original policy margins | selected M5 retuning |
| --- | ---: | ---: | ---: |
| single-pulse need, quanta | 2586.0434707264194 | 2586.0434707264194 | 2586.0434707264194 |
| measured loop period, steps | 47 | 106 | 106 |
| measured loop period, ms | 4.7 | 10.6 | 10.6 |
| mean-drive rate need, quanta/pulse | 382.8363636363636 | 863.418181818182 | 863.418181818182 |
| loop edge, quanta | 3621 | 3621 | 3621 |
| each AND input, quanta | 249 | 562 | 562 |
| each OR input, quanta | 766 | 1727 | 1296 |
| completion / each physical reset-drive edge, quanta | 3621 | 3621 | 3621 |
| reset per loop member, quanta | −5432 | −5432 | −5432 |
| external ignition per selected driver, quanta | 3621 | — | 410 |

Only the OR policy margin changes, **2.0 → 1.5**. `loop_margin=1.4`,
`and_fraction=0.65` and `reset_factor=1.5` remain fixed. The original OR margin
would request 1727 quanta on body **37111 → 512079**, which has 24 synapses and a
capacity of 1536 quanta. That target is **rejected**, requiring **4.4973958333×**
anatomical weight. It is not counted as a successful policy-compliant tuning.
The selected 1296-quanta OR target requires 3.375× on that edge.

External ignition changes from 3621 to **410 quanta = 7.046875 mV**. It enters V
and produces one initial spike. It does not create the old g reservoir. Timing,
observation window and all correctness thresholds remain unchanged. The selected
policy was first checked on `synthetic_mcns()`; the anatomical run uses that same
choice without a real-data parameter sweep.

**Shared reset-drive edges are counted physically.** The historical emitter lists
each completion→inhibitory edge once per rail served. In this H0 circuit, four
entries are repeats: 30 listed designed entries describe 26 physical connections.
The old simulator sums those entries, giving an effective 7242 quanta on each
shared drive, up to **7.3004032258×** anatomical weight. The original results are
reproduced unchanged as historical controls, but that aggregate exceeds the
present task's bound. The retuned circuit emits each shared physical drive once,
at 3621 quanta. This reduces actual reset drive and is part of the reported
retuning. The largest resulting multiplier is 3621/(62×16) = **3.6502016129×** on
11429 → 523983. The default historical emitter remains unchanged.

Held fixed: **k_max=4.0**, all 15 neurons and their roles, every physical designed
connection and its sign, anatomical synapse counts, both-member reset, and the
existing zeroing of **107 parasitic edges**. No synthetic edge or neuron enters
the tested anatomical circuit. The JSON audit checks the bound after summing any
parallel entries, rather than allowing repeated list entries to evade it.

## Transactions on the real circuit

The M0 and stale M5 controls reproduce the comparison report's outcomes and
latencies. Each cell below reports **correct decode + completion / reset_ok /
completion latency**. Passing additionally requires latency <60 ms.

| a,b | M0 original | M5 stale | M5 retuned | retuned pass |
| --- | --- | --- | --- | --- |
| 0,0 | yes / yes / 22.8 ms | yes / yes / 21.9 ms | yes / yes / 15.8 ms | **pass** |
| 0,1 | yes / yes / 36.9 ms | no / no / absent | yes / yes / 29.8 ms | **pass** |
| 1,0 | yes / yes / 36.9 ms | no / no / absent | yes / yes / 29.8 ms | **pass** |
| 1,1 | yes / yes / 56.3 ms | no / no / absent | yes / yes / 33.2 ms | **pass** |

| a,b | y1 fired | y0 fired | completion spikes | last loop spike relative to DATA | loop arrivals discarded | reset arrivals discarded |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0,0 | no | yes | 2 | 21.3 ms | 0/10 | 0/16 |
| 0,1 | no | yes | 1 | 31.9 ms | 0/14 | 0/8 |
| 1,0 | no | yes | 1 | 31.9 ms | 0/14 | 0/8 |
| 1,1 | yes | no | 1 | 37.2 ms | 0/16 | 0/8 |

Multiple spikes on the correct rail are allowed by the unchanged H0 checks. No
transaction fires the wrong output rail. The four trials use the circuit's own
completion and inhibition to end activity, with no host intervention after DATA.

## Diagnosis

**The stale circuit breaks at gate integration, before completion.** The latch
does regenerate under clear-g. Clearing g alone changes its measured steady period
from 47 to 106 steps; full M5 also gives 106 steps. The rate is therefore about
94.34 Hz, versus M0's 212.77 Hz. In a separate one-second probe without the
completion/reset return, the M5 driver emits **94 spikes**, including four in the
last 50 ms; its last spike is at 990.9 ms. This establishes persistence over the
measured window, not an infinite-duration claim.

The diagnostic motif consists of two independent latch pairs feeding uncoupled
AND and OR receivers; it measures gate dynamics without reset terminating the
input. These small diagnostic motifs are separate from the anatomical transaction
circuit. All voltages below are recorded after each simulator step, and a spiking
receiver has already been reset at its spike steps. The below-threshold peaks are
therefore the useful voltage-margin measurements; spike counts and first-spike
latencies measure suprathreshold behavior directly.

| one-second M5 gate probe | stale weights | retuned weights |
| --- | --- | --- |
| AND, one latch | 0 spikes; peak −49.8891 mV | 0 spikes; peak −47.2356 mV |
| AND, two simultaneous latches | 0 spikes; peak −47.7781 mV | 31 spikes; first at 27.9 ms after DATA |
| OR, one latch | 0 spikes; peak −45.5061 mV | 31 spikes; first at 24.5 ms after DATA |
| OR, two simultaneous latches | 47 spikes; first at 16.6 ms | 47 spikes; first at 10.5 ms |

Threshold is −45 mV. With stale weights, even two true inputs leave AND 2.7781 mV
short, while a single false rail leaves OR 0.5061 mV short. Thus (0,1), (1,0) and
(1,1) never reach completion. Two false rails can still fire OR, explaining the
surviving stale (0,0) transaction. Raising gate weights restores discrimination:
one retuned AND input remains **2.2356 mV below threshold**, while two fire.

Refractory loss is real but is not the loop failure mechanism here. Arrival counts
are reconstructed from the source spikes, 18-step delays and each postsynaptic
spike's refractory interval, including rejection on its own spike step and the
selected drivers' refractory exemptions. **0/54** loop arrivals are discarded in
each stale failing transaction; **0/54 across all four retuned transactions** are
discarded. A 1.8-ms edge delay does not by itself imply that a reciprocal loop's
return arrives within the same neuron's 2.2-ms hold: the return also waits for the
partner's integration and spike. The stale (0,0) transaction loses 2/8 inhibitory
reset arrivals and still resets. The retuned transactions lose no inhibitory
reset arrivals. Completion and reset therefore work with their original target
values once sufficient gate drive is restored, including the reduced physical
reset-drive weights described above.

## Tier B and scope

**Not run: Tier A passed.** The implemented fallback passes M5 rules into
`find_embeddings`, uses the same selected targets and k_max=4.0, and would cap the
search at **50 solutions / 300 seconds**, with the existing **400 assignments per
gate pair** cutoff, then test the best **10** returned embeddings by the existing
weight-margin/parasitic ranking. Those are prospective fallback caps, not a claim
that a Tier B search was performed. Its rule propagation, caps, physical-edge
audit and transaction execution are unit-tested on the synthetic fixture.

Only small circuits were simulated: 15 neurons for the anatomical trials and six
for the diagnostic motif. Reading MCNS's 166,700-neuron, 25,582,938-edge graph did
not construct or simulate a whole-brain topology. The source data were not copied,
linked, downloaded or modified. The graph digest is recorded in the JSON.

## Regression evidence and reproduction

No simulator or default model setting changed. Before editing, the unchanged
implementation at `927d6b4` supplied exact target values and four complete
transaction trace hashes for the synthetic fixture. New tests reproduce all those
values exactly for implicit and explicit default rules. Additional historical
target snapshots cover dt=0.02 ms, tau_m=25 ms and tau_s=4 ms. A direct comparison
with the unchanged module also matched every target field across 12 combinations
of those parameter sets and policy margins. The existing tests were not edited.

From the worktree root, using the required project environment:

```bash
PYTHONPATH="$PWD" /Users/jeremiahgassensmith/programming/drosophilos/.venv/bin/python \
  -m drosophilos.bench.h0_retune \
  --mcns-dir /Users/jeremiahgassensmith/programming/drosophilos/data/mcns \
  --out docs/stage_h/h0_under_shiu.json

# Development only; no real dataset is needed.
PYTHONPATH="$PWD" /Users/jeremiahgassensmith/programming/drosophilos/.venv/bin/python \
  -m drosophilos.bench.h0_retune --synthetic --out /path/to/writable/synthetic_h0.json

PYTHONPATH="$PWD" /Users/jeremiahgassensmith/programming/drosophilos/.venv/bin/python \
  -m pytest tests/test_h0_retune.py tests/test_embed_h0.py tests/test_rules_cli.py -q
```

The required targeted suite passes **38 tests**. An additional **48 existing
tests** pass across `test_model_compare.py`, `test_shiu_parity.py`,
`test_brian2_parity.py`, `test_dale_refractory.py`, `test_netlist_determinism.py`
and `test_netlist_split.py`, excluding tests marked `slow`. Brian2 reports only
the existing Pyparsing deprecation warnings. An unreadable MCNS directory is
recorded as `not_run`, with the error, without a replacement dataset or access
workaround. For this run, the requested checkpoint commit was attempted immediately
after the targeted suite passed, before the anatomical run; the sandbox denied
creating the Git index lock in the repository metadata outside the worktree.
Integration and commit must be completed by the orchestrator.
