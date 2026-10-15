# Perspective F16 copy 85: an operand-XOR veto escape leaves R6 missing

**Seed 110, copy 85 stalls because an internal operand XOR becomes double-railed,
not because a stage or master latch fails to establish its partner.** In
`c4_mulp.r4`, `fa6.x.y1.a.edge` (**81107**) incorrectly fires at **198,621**,
lighting the wrong XOR rail alongside the correct rail. Both rails then veto
the two possible sum outputs. Consequently **stage bit 6 (R6)** has neither
rail, its validity latch stays dark, and stage completion never rises.

A single receiving-neuron RefSim experiment with reconstructed copy-85 static
draws and captured input spikes reproduces **81107 at exactly 198,621** from a
cold receiver. Its second excitatory arrival escapes the weaker veto before
the strong source inhibition arrives. This is the **ordered-XOR double-rail
class**, now in the **operand XOR**, upstream of the sum XOR protected by
configuration F's transaction one-shot. It is not the reset-recovery partner
failure established for [copy 75](stage_d_f16_stall.md).

**A master ignition fix driven by COPY would not cover this failure. Neither
would merely applying the same ignition fix to Q's stage-register members:**
their output drivers never fire. Repair coverage must include the upstream
ordered operand-XOR evaluation, while preserving reset/reload margins.

## Artifacts and read-only method

Inputs under `/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:

- `perspective_f16_cal_s110_9206797.json`;
- `perspective_f16_s110_copy85_dump.npz`;
- `reports/perspective_f16_s110_copy85.md`.

`build_tick_pipeline(record)` reconstructed **123,323 neurons / 249,940
synapses**, with rate readers, experimental autapses, ordered sum-XOR one-shots,
and the experimental 16-bit option. `load_dump` loaded the full dump **once**:
**219,212,686 spikes**, 90,018 firing IDs, steps **23–449,999**, explicit capture
of all **123,323 IDs**, and **zero role-ID drift**. One step is **0.1 ms**.
`datapath_view(..., 'c4_mulp.r4', 196730)` and
`completion_view(..., 'c4_mulp.r4', 196730)` supplied the wiring and signal
checks; relevant spike arrays were grouped and retained in task scratch space
for subsequent reads. `arrival_alignment` checked the preceding reset arrivals.
The initial load/extraction took about **103 seconds**; reduced simulations
took seconds. No campaign artifact or code was changed.

All silence claims below concern captured IDs. Unless otherwise stated, the
failure window is **196,730–449,999**, inclusive. Names in tables omit the
`c4_mulp.r4.` prefix except for the generic `out.b6*` drivers, whose IDs are
resolved by wiring, not by their repeated role names.

## From the operand XOR to missing stage completion

At ACT^d (**78843**, **197,318**), the preceding r3 master supplies exclusive
operand rails encoding **acc=172, A=43, B=100**. Row 4 computes
`acc + ((A * B4) << 4)` modulo 2^16. Since **B4=0**, the expected accumulator
remains **172** and **R6=0**. The bit-6 operand XOR compares delayed acc6=0
with partial product **A2 AND B4=0** (`pp2`, shifted to position 6).

| Signal | ID(s) | Transaction-7 evidence |
|---|---:|---|
| `start` / final `actd.d10` | 18924 / 78843 | 196,730 / 197,318 |
| `pp2r0.u/v` | 80841 / 80842 | First 198,404 / 198,459; sustained |
| Delayed acc6=0, `s6dr0.d22` | 80371 | First 198,548; next 198,589, 198,627 |
| Correct operand-XOR ignition, `fa6.x.y0.b.edge` | 81116 | **198,587**, once |
| **Incorrect** operand-XOR ignition, `fa6.x.y1.a.edge` | **81107** | **198,621**, once |
| Correct XOR rail, `fa6.x.y0.u/v` | 81105 / 81106 | First **198,631 / 198,681**; last 449,960 / 449,988 |
| Incorrect XOR rail, `fa6.x.y1.u/v` | 81103 / 81104 | First **198,662 / 198,712**; last 449,957 / 449,982 |
| Delayed carry-in 0, `fa6.cinr0.d4` | 81579 | First 199,313; sustained; carry-in 1 (**81584**) silent |
| Sum transaction edge, `fa6.s.once0.edge` | 81593 | **199,354**, exactly once |
| Its spent latch, `fa6.s.once0.spent.u/v` | 81591 / 81592 | First 199,395 / 199,444; sustained |
| Sum-1 candidate, `fa6.s.y1.a.edge` | 81605 | Silent; veto **81607** driven by x0 starts 198,682 |
| Sum-0 candidate, `fa6.s.y0.b.edge` | 81614 | Silent; veto **81616** driven by erroneous x1 starts 198,713 |
| Both sum latches, `fa6.s.y1.u/v`, `fa6.s.y0.u/v` | 81601/81602, 81603/81604 | All silent |
| Final stage drivers, `out.b6r0.edge`, `out.b6r1.edge` | 82121 / 82123 | Both silent |
| Q bit-6 rail 0 / rail 1, u/v | 16288/16289 / 16290/16291 | **All four silent** |
| `Q.valid6.or`, `Q.valid6.L.u/v` | 16522, 16523/16524 | All silent |
| `Q.valid6.ign.edge/edge_inh/hold_inh` | 16527/16528/16529 | All silent |

The source one-shot delivers its sole pulse to both sum candidates at
**199,372** (18-step delay). Both veto trains are already established. The
other sum candidates, **81608** and **81611**, depend on the absent carry-in-1
one-shot (**81598**) and are also silent. The sum one-shot's spent latch
establishes normally; it neither double-fires nor loses a partner.

Exactly **50 of 51 Q validity latches** and **44 of 50 Q completion latches**
ignite. The other 50 stage bits establish one selected rail with both members;
R6 alone is missing. The six dark completion joins are the actual ancestors
of valid6, with both members silent:

| Q completion join | u/v IDs |
|---|---:|
| `c0_3.L` (valid6 AND valid7) | 16952 / 16953 |
| `c1_1.L` | 17136 / 17137 |
| `c2_0.L` | 17232 / 17233 |
| `c3_0.L` | 17280 / 17281 |
| `c4_0.L` | 17304 / 17305 |
| `c5_0.L` (stage root) | 17320 / 17321 |

The root's other branch, `Q.comp.c4_1.L.u/v` (**17312/17313**), starts at
**200,753/200,806** and persists. Thus completion correctly waits for a missing
leaf; there is no evidence of a separate validity or completion failure.
Commit command **122411**, master reset **18432**, COPY **18474/18475**, and
DONE **18884** never fire in this window. All Q fault gates and FAULT
**17367/17368** are silent: the double rail is **inside the arithmetic**, and
never reaches the stage register's double-rail detector. Six outputs had
completed, with no wrong outputs or refusals. The **252,681 steps** since
ACT^d greatly exceed the report's **26,386-step** stall threshold.

## The first wrong pulse: a second input beats source inhibition

Receiver **81107** has exactly three wired inputs, all with **18-step delays**:

| Source | Nominal q | Reconstructed copy-85 q | First emissions → arrivals |
|---|---:|---:|---|
| Delayed acc6=0 **80371** | +4655 | **+4919** | 198,548→198,566; **198,589→198,607**; 198,627→198,645 |
| Source inhibitor **81108** | −7966 | **−7663** | **198,606→198,624**; 198,653→198,671 |
| Partial-product-0 veto **81109** | −1810 | **−1792** | 198,462→198,480; 198,530→198,548; **198,596→198,614** |

Its threshold is **−45.17450109365213 mV**, bias
**+0.022141032716653797 mV**. The initial veto is present **86 steps before**
the first excitatory arrival. This is not missing EARLY data. The first
excitation alone is suppressed, but the second arrives just **41 steps**
later, while the source inhibitor's first arrival is still pending. The
receiver fires at **198,621**, **14 steps after the second excitation**, seven
steps after the latest weak veto arrival, and **three steps before** strong
source inhibition. Its output reaches wrong-rail u **81103** at **198,639**;
u and v then establish normally at **198,662/198,712**.

The +4919 input is about **5.67% stronger** than nominal; the magnitude of
source inhibition is about **3.81% weaker**. These static draws and the actual
arrival spacing are sufficient for the failure. Storage autapses do not exist
on this relay and cannot prevent its initial escape.

### Reduced reproduction and controls

Static draws were regenerated in `make_perturbed_sim` order:
`default_rng(110)`, B=100; log-normal weight factors with sigma 0.04 over
`(100, 249940)` **sorted-topology edges**, followed by threshold and bias
normals with sigma 0.2 mV over `(100, 123323)`; select zero-based copy **85**.
The isolated receiver starts at its resting equilibrium, **E_L + bias**, with
zero current. Captured presynaptic spikes are prescribed as external arrivals.
There is no supplied stray input or preceding reset.

| Receiving-neuron experiment | Result |
|---|---|
| Captured transaction-7 inputs and reconstructed parameters | **81107 fires at 198,621**, matching the dump |
| Only the first two excitatory pulses, same inhibitory trains | Same **198,621** spike |
| Only the first excitatory pulse | No spike; maximum voltage **−49.7878800009 mV** |
| Nominal three input weights, same noisy threshold/bias and spike times | No spike; maximum **−45.5319066719 mV** |
| Delay only the second excitatory arrival by 3 steps | No spike; maximum **−45.3047385854 mV** |
| Captured transaction-5 input trains, same static draws | No spike; maximum **−45.2870575598 mV** |
| Captured transaction-6 input trains, same static draws | No spike; maximum **−45.2787721554 mV** |

Separate receiving-neuron checks of the two sum candidates, using their own
reconstructed draws and captured inputs, produce no spikes at **81605** or
**81614**. Removing only erroneous-x1 veto **81616 → 81614** permits the
correct sum-0 relay to fire at **199,399**. This isolates the downstream
suppression; it is a prescribed-input counterfactual, not a repaired pipeline.

For a short self-contained reproduction of the first escape, use the existing
RefSim with these **arrival** times (origin 198,000). The first two excitatory
arrivals suffice; subsequent captured arrivals occur after the observed escape.

```python
from drosophilos.sim.model import Params, Topology
from drosophilos.sim.ref64 import RefSim

p = Params()
s = RefSim(Topology.empty(1), p,
           V_th=[-45.17450109365213], bias=[0.022141032716653797])
s.V[:] = p.E_L + 0.022141032716653797
s.add_events(0, [566, 607], [0, 0], [4919, 4919])
s.add_events(0, [480, 548, 614], [0, 0, 0], [-1792] * 3)
s.add_events(0, [624], [0], [-7663])
s.run(650)
assert s.trace.events['step'].tolist() == [621]
```

This establishes a sufficient local mechanism, not an exact full-copy GPU
replay. The dump has spikes, not membrane voltages or the device's background
events. The original batch-dependent stray stream is unavailable. The cause
of the small transaction-to-transaction input timing differences is therefore
not reconstructed; attributing those differences specifically to strays is
a **hypothesis**. The receiver's wrong spike itself needs no extra stray in
the reduced experiment.

## Reset history and the two successful preceding transactions

| Transaction | START / ACT^d | Correct x0 u/v first | Wrong x1 | Q R6=0 u first | Stage completion / DONE |
|---|---|---|---|---:|---|
| 5 | 166,100 / 166,685 | 168,003 / 168,053 | Silent | 168,899 | 171,708 / 177,521 |
| 6 | 181,450 / 182,035 | 183,351 / 183,401 | Silent | 184,248 | 187,055 / 192,870 |
| 7 | 196,730 / 197,318 | 198,631 / 198,681 | u/v 198,662 / 198,712 | Silent | Neither |

All three transactions have the same captured operands **acc=172, A=43,
B=100**, and the same static parameters. The first two delayed acc6=0 spikes
are separated by **44 steps** in transactions 5 and 6, versus **41** in
transaction 7. Relative to that source's first spike, the first source-inhibitor
arrival is +76 steps in transactions 5 and 7, and +74 in transaction 6.
The successful cases are close to the same receiver threshold in the reduced
simulation. This is a timing-sensitive corner, not a new operand value first
selecting an unused weak latch.

Transaction 6's DONE at **192,870** causes Q reset **17327** at **192,913**.
Q reset inhibitor **17328** emits at **192,964, 193,007, 193,050, 193,099,
193,172**. The nominal −2716-q reset edges reach both members of the relevant
internal XOR latches and stage rails at:

**192,982; 193,025; 193,068; 193,117; 193,190.**

- Old x0 u/v **81105/81106** last fire at **193,009/192,964**, then stay
  silent until the new load at **198,631/198,681**. The second reset arrival
  is 16/61 steps after their respective last spikes.
- Wrong x1 u/v **81103/81104** last fired in transaction 4, at
  **162,552/162,519**. They remain dark throughout transactions 5 and 6,
  until **198,662/198,712**. This is fresh incorrect ignition, not survival.
- Old Q R6=0 u/v **16288/16289** last fire at **192,968/193,022**; neither
  ever fires again. At the second reset arrival their spike ages are 57 and
  three steps respectively. A pulse close to a refractory interval does not
  establish survival: the subsequent complete silence disproves it here.
- The old delayed acc6 source **80371** drains through **194,653**, its source
  inhibitor **81108** through **194,748**, and veto **81109** through
  **193,077**. All stop well before START. Receiver **81107** has no reset
  input: its incoming edges are exactly the three listed above.

Q READY **17348** occurs at **193,742**. From the final reset arrival to the
next ACT^d is **4,128 steps (412.8 ms)**, to the first delayed acc6 input at
the erroneous relay **5,376 steps**, and to wrong x1's ignition arrival
**5,449 steps**. These are not copy 75's short master-reset-to-COPY budget.
Moreover, both internal XOR latch partners establish, and the cold-receiver
experiment reproduces the initiating error without any reset history.

## Why r4 again?

**There is no evidence that r4 is uniquely fragile as a register or has a
special reset fan-out.** Every one of the 16 MULP rows has a **51-bit** packed
stage/master word: R[15:0], C/Z/V, A[15:0], B[15:0]. Every row carries **all**
A and B bits; copy 75's packed bit 40 is B5 in each row, not an r4-specific
operand. Every row has **711 negative master-reset edges** and **1,347
negative stage-reset edges**, including the arithmetic reset domain and
reader mirrors. Counts are edges, not necessarily distinct targets.

The row index changes the selected multiplier bit and shift: row j consumes
Bj and adds partial product A[i−j] AND Bj at result position i≥j, with zero
below j. Thus this r4 failure concerns **A2/B4 at R6**, whereas copy 75 lost
the **carried B5 master rail**. B=100 has ones at 2, 5 and 6; r4 is one of
many zero-addend rows, rather than a uniquely active multiplier row.

The pipeline builder uses the same B/A/acc token delays (**6/17/23 hops**),
five-hop carry delays, ordered operand-XOR primitive, sum one-shot and register
control policy in these rows. The failing delayed source **80371** has six
outgoing edges; x1 u/v **81103/81104** have four/two, and x0 u/v
**81105/81106** six/two. These are ordinary roles in the common bit-slice
wiring, not a special r4 load. Synapses deliver independent weighted events;
fan-out does not divide the source's output weight or consume its voltage.

Measured intervals also do not single out r4 as unusually short. Final Q
reset arrival → ACT^d for transaction 7 is **3976 / 3948 / 4128 steps** in
r2/r3/r4. For r5/r6's last completed transaction (6), it is **3883 / 4009**.
These are different cells' observed schedules, not a controlled timing survey,
but r4 has no uniquely tight reset-to-load interval in this comparison.

The supported interpretation is **two different vulnerable draws/timing
combinations happened to localize in r4** in copies 75 and 85. The same seed
does not give the two copies the same per-neuron weights, thresholds or biases.
The present evidence cannot prove a statistical coincidence or rule out
workload-dependent row exposure. Any claim of a campaign-wide r4 hotspot is
a **hypothesis** requiring the other frozen copies and cross-row qualification.

## Comparison and repair implications

| Established class | Relation to copy 85 |
|---|---|
| Copy 75: master u 17531 fires once after COPY; partner v 17532 never fires under residual reset inhibition | Different. Copy 85 never reaches COPY; neither a lone-u stage load nor a missing partner initiates it. |
| Tick2 master/stage survivor through clear, followed by inhibited validity re-ignition | Different. Old relevant rails extinguish; the wrong internal rail is freshly ignited. R6's validity OR and ignition inhibitor are both silent. |
| Tick2 seed-110 copy 82: ordered sum-XOR relay escapes veto on a second delayed carry pulse | Same local hazard: a second excitatory pulse beats source inhibition despite the weak veto. Copy 85's affected primitive is the **operand XOR**, driven by delayed acc6, rather than the sum XOR driven by carry. |

See [tick2 datapath analysis](stage_d_datapath_stall.md) and
[completion survivors](stage_d_completion_stall.md). Configuration F's
`Gates.ripple_adder_ordered` applies `ordered_xor_oneshot` to **sum** XORs;
the preceding `fa*.x` operand XORs retain ordinary veto relays. The captured
sum one-shot here works as designed, but assumes an exclusive EARLY operand.
With x0 and x1 both live, its single carry pulse has no permitted output.
This explains why the symptom is a missing stage bit and no FAULT, unlike
tick2 copy 82's double rail reaching the stage detector.

The proposed opt-in COPY ignition change—drive both master members, or drive
the partner after a delay—targets copy 75's failed master establishment.
It has **no direct mechanism-based coverage for copy 85**. A Q-register
equivalent is also insufficient: **82121/82123 never supply an ignition**.
Forcing validity or completion would conceal the missing computation.

A candidate for this failure must make the **operand XOR** exclusive under
the observed delayed-acc burst/veto timing. Consuming that source once,
improving veto/source-inhibitor timing, or positively qualifying the branch
are hypotheses for repair, not qualified fixes in this analysis. In particular,
the prior sum-XOR work documented reload problems when extending one-shots to
operand XORs; simply enabling them everywhere is not established safe. Such
work must qualify first-pulse acceptance, rejection of later pulses, and
repeated reset/load behavior, alongside master-ignition and survivor tests.

Validation: the full-dump role/capture checks and reduced receiving-neuron
experiments above passed. The embedded 650-step reproduction and
`test -f docs/stage_d_f16_stall_copy85.md` passed; `git diff --check` passed.
Only this new document is changed. Integration and committing remain with
the orchestrator.

## 2026-10-14: experimental operand-XOR transaction one-shot, version 1

**Historical, rejected candidate (427af21).** The local observations in this
section predate the failed cluster surveys below. V1 is not safe to qualify
or enable by default; current enabled builds use independently identified v2.

`operand_xor_oneshot=True` / `--operand-xor-oneshot` extends the existing
transaction-edge circuit to the ordered adder's **`fa*.x` operand XORs**.
It is independently recorded as `operand_xor_oneshot_version=1`, and defaults
off. Disabled builds omit both new record keys and retain the old ordered
neuron/synapse arrays and image. No frozen digest is updated. Complete
configuration F is required (`rate_robust`, `experimental_autapse`,
`ordered_xor_oneshot`); its ordinary timing/reset restrictions apply. The
experimental F16 flag and master ignition v2 can both be enabled alongside it.
This is an **unqualified candidate**; the smoke results below are not campaign
or cluster qualification.

Each delayed operand rail now drives `add_transaction_edge` before branching
to its two XOR veto relays. The first edge pulse ignites a spent latch and
delivers the existing zero-delay −4.4-loop feedback; a spent-driven inhibitor
holds consumption over source gaps. The ordinary source inhibitor remains.
Both spent members join the owning `Gates.latches` and the stage reset domain.
Source/hold inhibitors drain naturally. The generate/kill and carry relays,
the SUB B-complement XORs (`bx*`), standalone logic, and reset/READY policy
retain their circuits. This is specifically operand protection in ordered
adders, including array and pipelined multiplier rows, rather than a global
change to every XOR.

Both campaign CLIs forward the flag. `kernel_campaign.rebuild_record`, Stage D
`--recheck`, and `stall_diag.build_tick_pipeline` require an integer v1 when
enabled, reject bool/string/unknown versions and non-boolean flags, and default
historical records to off. Stage D's nested compiled options take precedence.
Build tests compare generic and specialized tick/render/mul8/perspective
netlists before and after replay, including both options together. The combined
build differs from the operand-only build by exactly master ignition's existing
partner edges. No controller or reset edge is otherwise added by ignition.

### Copy-85 reconstruction

The test freezes receiver 81107's reconstructed threshold/bias and its three
input weights from the analysis above. Captured arrivals (origin 198000) at
566/607/645, 480/548/614 and 624/671 reproduce **step 621** without strays;
nominal-weight and first-excitation-only controls stay silent.

The circuit probe uses the actual `Gates.ripple_adder_ordered` operand XOR.
It retains that receiver's draws and captured delayed-acc and veto emissions.
The legacy circuit additionally prescribes its captured source-inhibitor
emissions and becomes double-railed. In the candidate, both source inhibitors
and the new spent-latch circuit run as real neurons: the wrong output stays
silent and the correct rail establishes. New components and the other
receivers get independent ±3-sigma endpoint draws; subsequent synthetic source
pulses and a long gap check retained consumption. The copy-85 survey jitters
the captured excitatory train by ±5 steps and includes its unshifted case in
every batch. It does not regenerate changed-topology random IDs or claim a
full-copy CUDA replay. Only the receiving draws documented above are frozen;
the missing device stray stream is not reconstructed.

### Cost and local latency

Measured generic configuration-F builds, ordinary resets, operand option
off/on; perspective additionally uses `f_width16_experimental=True`:

| Kernel | Protected operand bits | Neurons off / on | Synapses off / on | Added neurons / synapses | With both operand protection and master ignition: neurons / synapses |
|---|---:|---:|---:|---:|---:|
| `tick2.c` | 72 | 31,363 / 32,083 | 64,238 / 66,110 | 720 / 1,872 | 32,083 / 66,412 |
| `render.c` | 32 | 11,768 / 12,088 | 24,228 / 25,060 | 320 / 832 | 12,088 / 25,164 |
| `mul8` | 80 | 16,674 / 17,474 | 35,703 / 37,783 | 800 / 2,080 | 17,474 / 37,865 |
| perspective (`render2.c`, F16) | 320 | 123,323 / 126,523 | 249,940 / 258,260 | 3,200 / 8,320 | 126,523 / 260,114 |

Each protected bit adds **10 neurons / 26 synapses**, including four reset
links and four storage autapses. Added anatomical k4 synapse capacity is
129,744 / 57,664 / 144,160 / 576,640, respectively. Weights and feedback
strengths are inherited from the sum-XOR transaction primitive.

The nominal operand-XOR output latency is **8.4 ms off / 12.6 ms on**, all four
truth-table inputs: **+4.2 ms per operand XOR** in each of these four kernels.
Added controller/READY waiting is **0 ms** for each. The extra edge precedes
the ripple, rather than adding a delay at every carry hop. Dependent multiplier
rows and successive cells may accumulate its delay. These are primitive
latencies; end-to-end kernel output latency and throughput have not been
measured, and cannot be inferred by multiplying the bit count by 4.2 ms.

### Separate cluster scenarios and bounded laptop smoke

`tests/test_operand_xor.py` reuses the unmodified RefSim step with bounded
count/first/last summaries. Each following case is a separate pytest parameter
of **`test_operand_xor_cluster[CASE]`**, marked `slow` and skipped unless
`DROSO_OPXOR_CLUSTER=1`:

| CASE | Cluster exposure |
|---|---|
| `second-pulse` | 40,000 physical copies, four truth-table/reset transactions each; 41-step second arrival, later bursts and long gap; independent endpoint draws |
| `order-15ms` | Same, EARLY leads by at least 15 ms; 30-step burst spacing |
| `order-100ms` | Same, EARLY established 100 ms before LATE |
| `mix-B` | Same four-transaction/reset probe with independent Gaussian/log-normal static mix-B draws |
| `copy85` | 40,000 captured-receiver probes, new-component endpoints and timing jitter |
| `addsub4-mix-B` | All 512 operand/op combinations and swapped second words: 1,024 noisy transactions |
| `addsub8-mix-B` | 2,048 physical copies / 4,096 noisy ADD/SUB transactions |
| `addsub16-nominal` | 2,048 physical copies / 4,096 nominal ADD/SUB transactions |

Endpoint draws are independent log-weight ±0.12 and threshold/bias ±0.6 mV.
The local XOR probes test every truth-table value and polarity change, later
burst rejection, gaps exceeding source-inhibitor recovery, actual reset and
reload. They assume exclusive EARLY data before LATE; reversed arrival order
is outside this ordered primitive's contract. Local probes omit target strays.
Arithmetic uses the kernel's 6/17-hop B/A and five-hop carry delays and actual
SUB image rail, with a 200-ms reset recovery. Noisy arithmetic includes 4%
log-normal weights, 0.2-mV threshold/bias sigmas, 5-Hz / 150-q independent
background input, and 0–100-step per-bit load jitter. Directed carry/borrow,
zero, overflow and alternating patterns lead the eight- and sixteen-bit
surveys. It checks every sum and final carry/no-borrow rail, consumed-edge
acceptance/repetition and post-reset latch silence. A 100-ms recovery is not
qualified by this candidate.

Run **one cluster scenario per job**, replacing CASE with one row above:

```sh
DROSO_OPXOR_CLUSTER=1 uv run pytest -q -s -m slow -p no:warnings \
  'tests/test_operand_xor.py::test_operand_xor_cluster[CASE]'
```

The independent laptop parameters are `test_operand_xor_smoke[CASE]`, skipped
unless `DROSO_OPXOR_SMOKE=1`. They use eight local physical copies (32 reset/
truth-table transactions, or eight copy-85 probes) and twelve arithmetic
copies / 24 transactions per case. Commands:

```sh
DROSO_OPXOR_SMOKE=1 uv run pytest tests/test_operand_xor.py \
  -q -s -m 'not slow' -p no:warnings
uv run pytest tests/test_build_options.py tests/test_ordered_xor.py \
  tests/test_operand_xor.py tests/test_config_f_kernels.py \
  -q -m 'not slow' -p no:warnings
```

The existing project virtualenv is read-only in this isolated worker, so the
local runs use `UV_PROJECT_ENVIRONMENT` pointing to it, `UV_NO_SYNC=1`,
`PYTHONPATH` pointing to this worktree and writable `UV_CACHE_DIR=$TMPDIR/uv-cache`.
All work is foreground; no long operand cluster survey is run locally.

### Observed noisy-16 frontier: carry propagation remains vulnerable

An additional directed **noisy** 16-bit probe is deliberately retained as
`test_noisy16_carry_escape_frontier`, gated by `DROSO_OPXOR_SMOKE=1`. This is
a regression of a rejected envelope, not a passing qualification case or an
expected-failure marker. In 24 transactions it observes **one double/wrong
sum** (copy 9, second word, `0xaaaa - 0x5555`, R12).

`fa11.x.once1.edge` fires once, x0 establishes and x1 stays silent. But the
ordinary carry-propagation relay **`fa11.p0.edge` fires at 11527**, despite
the established x0 veto, lighting carry-0 alongside the correct carry-1.
The next sum's two one-shots then each consume their respective input rail;
the result double-rails. The test asserts this adverse behavior explicitly,
including the exclusive operand XOR and wrong carry pulse. It does not weaken
the arithmetic acceptance checks, change the random seed to avoid the draw,
or silently discard the transaction.

The requested nominal 16-bit transactions and noisy four-/eight-bit smoke
cases pass. **No noisy-16 qualification is claimed**, and this additional
failure prevents claiming a clean broader 16-bit noisy envelope. Carry
generate/propagate protection needs separate investigation; it is outside
this operand-XOR-only circuit extension. The long surveys above are prepared
for separate cluster runs and have not been executed here.

### Foreground smoke results

The final operand test command above passes **14 tests** with the eight slow
cluster parameters deselected. This includes the explicit noisy-16 rejection
regression and two real noisy ADD → SUB → XOR pipelines (generic/specialized)
with operand protection, master ignition and `zero_once` enabled together.

All eight acceptance smoke scenarios report zero wrong/double/missing rails,
repeated consumed edges or post-reset survivors:

| Smoke case | Transactions | First output latency, min–max |
|---|---:|---:|
| `second-pulse` | 32 | 10.6–15.4 ms |
| `order-15ms` | 32 | 10.6–14.9 ms |
| `order-100ms` | 32 | 10.6–14.9 ms |
| `mix-B` | 32 | 12.1–13.1 ms |
| `copy85` | 8 | 11.4–14.6 ms |
| `addsub4-mix-B` | 24 | 104.4–226.4 ms |
| `addsub8-mix-B` | 24 | 98.4–374.0 ms |
| `addsub16-nominal` | 24 | 100.9–641.2 ms |

Arithmetic timings include operand/ripple staging and are measured from the
nominal load origin; noisy per-bit load jitter is included. Local XOR timing
starts at its first LATE emission. The eight scenario bodies take about
14 seconds combined; the whole smoke file also exercises the controller
pipelines and the rejected noisy-16 frontier and finishes in under a minute.
The nominal four-row latency probe measures 84 steps off / 126 steps on.
**These are smoke observations only; no cluster or kernel campaign
qualification is claimed.**

The required foreground command covering `test_build_options.py`,
`test_ordered_xor.py`, `test_operand_xor.py` and `test_config_f_kernels.py`
with `-q -m 'not slow' -p no:warnings` exits **0**. Existing default and F
frozen digest checks pass unchanged. The mandatory suite takes approximately
11 minutes on this laptop, including the existing large Doom4 fingerprint
builds; the new operand smoke stays under a minute. `git diff --check` passes.
No cluster survey, full-kernel latency benchmark, GPU campaign or commit was
performed. Only the nine owned files are changed.

## 2026-10-14: failed v1 cluster surveys and opt-in v2 redesign

**V1 protects the operand XOR but exposes the unprotected carry path, and its
operand output storage can survive a weak clear.** The named wrong sums do
not originate in a wrong first operand-XOR edge. V2 shares the transaction
pulses with carry branches and increases clear strength only for operand-XOR
output/spent storage. It remains **unqualified and default off**.

### Failed results retained as history

Read-only logs in
`/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/opxor_cluster/`:

| Log / v1 scenario | Transactions | Wrong | Double | Missing | Survivors |
|---|---:|---:|---:|---:|---:|
| `slurm-457245.out`, `copy85` | 40,000 | 0 | 0 | 0 | **32 latch members** |
| `slurm-457246.out`, `addsub8-mix-B` | 4,096 | **19** | **19** | 0 | 0 |
| `slurm-4018656.out`, `addsub4-mix-B` | 1,024 | **4** | **4** | 0 | 0 |

Consumed-edge repetitions are zero. The arithmetic failure counts are wrong
**bit rails**, not necessarily that many distinct transactions; e.g. the
four-bit copy 75 has both R3 and carry/no-borrow wrong. The earlier sum-only
one-shot survey's zero failures in 1,024 four-bit and 4,096 eight-bit noisy
transactions did not justify extending its circuit without checking carry
timing and the extra reset state.

### First incorrect events in the named arithmetic cases

All steps below match the log and the local v1 RefSim replay exactly.

| Evidence | Eight-bit copy 175, tx 0: 14 − 161 | Four-bit copy 75, tx 1: 5 − 2 |
|---|---:|---:|
| Correct operand XOR | `fa1.x.y0.u/v`: 1347 / 1406 | `fa2.x.y0.u/v`: 6361 / 6416 |
| Wrong operand rail `x.y1.u/v` | Both silent | Both silent |
| Selected operand one-shot | `fa1.x.once1.edge`: once at 1258 | `fa2.x.once1.edge`: once at 6272 |
| Selected EARLY operand veto | `fa1.x.y1.b.veto`: 726 | `fa2.x.y1.b.veto`: 5720 |
| Correct generate carry-1 | `fa1.c1.u`: 1301 | `fa2.c1.u`: 6312 |
| Carry-0 delayed driver | `fa1.cinr0.d4`: 1460 | `fa2.cinr0.d4`: 6460 |
| Carry-propagate x0 veto | `fa1.p0.veto`: 1400 | `fa2.p0.veto`: 6417 |
| **Wrong first carry-propagate ignition** | **`fa1.p0.edge`: 1536** | **`fa2.p0.edge`: 6530** |
| Strong source inhibitor emission → arrival | 1518 → **1536** | 6514 → **6532** |
| Incorrect carry-0 establishes | `fa1.c0.u`: 1577 | `fa2.c0.u`: 6571 |
| Next sum accepts both carries | `fa2.s.once1/once0.edge`: 1604 / 1876 | `fa3.s.once1/once0.edge`: 6608 / 6886 |

V1 delays x by a nominal **42 steps** while generate/kill still take raw Ad
and propagate still takes the raw delayed carry train. The x0 veto therefore
has less time to settle before carry propagation. The wrong propagate relay
escapes before its strong source inhibition takes effect. Once it establishes
the second carry rail, two independent sum one-shots each correctly consume
one of the two asserted rails; they do not arbitrate exclusive carry data.
The sums then become double-railed, producing silent wrong-value observations.

Captured carry excitations arrive at **1478/1522** and **6478/6522**,
respectively: 44-step doublets. Cold single-neuron controls using those first
two pulses, the actual veto pulses and reconstructed receiving draws reproduce
the wrong spikes at **1536 / 6530**; a first-excitation-only control stays
silent in both cases. Copy 175 uses q **+4491 / −7631 / −1674** (driver /
source inhibitor / veto). Copy 75 uses **+4489 / −7403 / −1622**, plus its
captured **150-q background arrival at 6371**. Removing that background pulse
in the reduced copy-75 experiment delays the spike to 6531; it still escapes
before strong source inhibition arrives at 6532. These cold controls establish
a sufficient second-pulse mechanism; the full two-word batch replay preserves
actual history and background rather than assuming a cold transaction start.
Advancing only the two captured veto arrivals by 42 steps also prevents both
reduced escapes, with excitation and receiving draws held fixed. This supports
the lost-veto-margin explanation directly. The earlier sum-only surveys used
a different topology/draw assignment; their clean counts alone cannot isolate
the latency change. The v1/v2 comparisons below preserve v1's physical draws.

The operand's EARLY veto precedes its consumed edge by 532/552 steps in these
two slices, and the correct x0 alone establishes. This rejects a premature
operand edge or a SUB rail-swap error for these failures. The logged ADD cases
(four-bit copies 140/338 and eight-bit copies 497/536) also rule out a
SUB-specific explanation. This does not establish safety under reversed
EARLY/LATE ordering, which remains outside the ordered primitive's contract.

### A frozen survivor: output storage, not a spent latch

The original copy85 log reports only an aggregate survivor count, so it cannot
identify which latch or ignition survived. A bounded local diagnostic prefix
of 4,096 single-bit probes localized **copy 2174**; the permanent regression
reconstructs its draws and simulates only that one copy.

With v1, `add.fa0.x.y0.u/v` continue through steps **6179 / 6156** after the
clear at 4200. Their corresponding spent latch does not survive. The noisy
ordinary reset controller emits **three** inhibitor spikes, the last at
**4461**, rather than the nominal five. All delayed prescribed excitation
ends by 2705, well before clear. This is held output storage surviving the
available clear charge, not a late source pulse re-igniting a cleared output.
Two continuing members account for the two failures in this frozen case;
the aggregate 32 need not all have the same cause.

The shared-pulse redesign alone still leaves both members alive under the
same draws. Raising only operand-XOR output/spent clear links from **0.75 to
1.5 loop** extinguishes both. No controller or READY delay is changed, and
source/hold inhibitors remain outside reset so draining trains remain blocked.
The existing four-transaction polarity-change probes and two-word arithmetic
replays exercise reloading after these stronger clears. Copy 2174 additionally
reloads the opposite EARLY polarity after its 2000-step recovery, checks one
exclusive output and one consumed edge, and clears again. Tighter recovery
times and all 40,000 original clear draws still require cluster evaluation.

### V2 circuit and replay policy

`OPERAND_XOR_ONESHOT_VERSION = 2`. The flag and complete-F prerequisites stay
the same; absent/false flags retain the old arrays, image and record keys.
Enabled records with missing, v1, string, boolean or unknown versions are
rejected by campaign rebuild, Stage D recheck and diagnostic reconstruction;
v1 is reproduced only by a test fixture, never silently upgraded on replay.

The existing transaction edges are reused:

- `fa*.x.once{0,1}.edge` drive both operand XOR candidates and the matching
  generate/kill branches. Generate/kill now share the operand's delay and
  receive only the consumed pulse.
- `fa*.s.once{0,1}.edge` drive both sum candidates and matching carry-propagate
  branches. Bit zero uses the existing `s.once1` pulse for both Cin-qualified
  propagation candidates; later bits use their respective carry pulses.
- `Gates.finalize_operand_reset` strengthens the completed stage-reset fan-out
  to operand XOR output/spent members and their registered reader mirrors.
  Sum spent storage, other state and reset generators keep their policy.

This adds **no neurons or synapses beyond v1**: the neuron/synapse counts in
the v1 table still apply (10 neurons / 26 synapses per protected bit). Eight
existing storage clear weights per bit become 1.5 loop; the v1 weight-dependent
anatomical-capacity estimates therefore do not carry over. Some node IDs
change because sum pulses are allocated before their carry consumers. Standalone and SUB
B-complement XORs remain unchanged. The ordinary sum-only option remains v1.

Nominal operand primitive latency remains **84 steps off / 126 on**. Carry
generation and propagation now also include the shared consumed edge, so
end-to-end latency must be measured afresh; the v1 arithmetic latency table
does not describe v2. No kernel throughput or F16 campaign claim follows.

### Explicit frozen regressions and bounded local execution

The existing eight cluster scenarios remain behind `DROSO_OPXOR_CLUSTER=1`.
Ten additional scenario IDs reproduce every distinct copy in the logs'
capped `failure_details`, plus the localized clear survivor:

| Scenario | Frozen failing transaction: tx, bit, a, b, sub |
|---|---|
| `replay8-copy175` | 0, 2, 14, 161, 1 |
| `replay8-copy391` | 0, 2, 238, 1, 1 |
| `replay8-copy418` | 1, 4 (also 5), 103, 10, 1 |
| `replay8-copy497` | 1, 8, 131, 137, 0 |
| `replay8-copy536` | 0, 7 (also 8), 72, 218, 0 |
| `replay8-copy699` | 0, 2, 233, 198, 1 |
| `replay4-copy75` | 1, 3 (also 4), 5, 2, 1 |
| `replay4-copy140` | 0, 4, 4, 6, 0 |
| `replay4-copy338` | 1, 4, 9, 10, 0 |
| `replay85-clear2174` | x0 output survives three emitted clear pulses |

These also run as independent smoke parameters behind `DROSO_OPXOR_SMOKE=1`.
Each arithmetic regression first asserts the adverse v1 result, including
the reported operands/tx/bit, then requires zero wrong/double/missing/repeated
rails or surviving state under v2. Both named cases additionally assert the
exact wrong propagate pulse above. The existing noisy-16 frontier is retained
as a v1-negative/v2-clean frozen 24-transaction probe, not a qualification.

Copy indices are **not seeds**. Arithmetic reconstruction preserves RNG 718,
64-copy batch sizes, all preceding static draws, operand choices, load jitter
and 5-Hz/150-q background draws; preceding batches consume RNG without neural
simulation. Only the batch containing the requested copy is simulated, and
only that copy's two words are scored. V2 maps original perturbations by role
and source/destination/delay; a rewired positive driver retains the original
receiving edge's draw. Background events map to the same neuron roles. The
copy85 regression preserves RNG 8585 and 256-copy draw blocks but simulates
only selected rows. Assertions require complete bijections of nodes/edges.
No changed-topology reseeding can make a failing physical draw disappear.

Commands remain the two foreground commands in the v1 section above, using
the existing environment and writable task temp/cache directories. Long
cluster surveys and GPU/kernel campaigns have **not** been rerun locally.

### V2 foreground validation results

The required command passed with exit **0**:

```sh
uv run pytest tests/test_build_options.py tests/test_ordered_xor.py \
  tests/test_operand_xor.py tests/test_config_f_kernels.py \
  -q -m 'not slow' -p no:warnings
```

This includes the existing frozen default/F digests, explicit-off byte
comparisons, v2 option replay and v1 rejection, both datapaths, F16 builds,
master-ignition compatibility, and actual small pipeline reset/rearm cycles.
No frozen digest was updated.

The final operand smoke run passed **25 tests**, with **18 slow cluster
parameters deselected**, in **269.99 seconds (4 minutes 30 seconds)**:

```sh
DROSO_OPXOR_SMOKE=1 uv run pytest tests/test_operand_xor.py \
  -q -s -m 'not slow' -p no:warnings -o addopts=''
```

All eight original v2 smoke scenarios report zero wrong/double/missing rails,
repeated consumed edges and post-clear survivors. All nine arithmetic frozen
copies reproduce their logged v1 failure before passing with v2. The frozen
clear draw reproduces two v1 survivors, retains them with shared pulses alone,
and passes both clear and opposite-polarity reload/clear with complete v2.
Both reduced receiver controls and the retained noisy-16 v1-negative/v2-clean
probe pass. These observations remain **smoke, not qualification**.

V2 arithmetic-fragment smoke latencies, measured from nominal load (including
noisy load jitter where enabled), are **108.2–239.7 ms** for four-bit mix B,
**102.6–401.4 ms** for eight-bit mix B, and **105.1–704.2 ms** for nominal
sixteen-bit arithmetic. These are not full-kernel latency measurements.

## Operand-XOR v2 cluster results and v3 reset correction (2026-10-14)

The v2 tip **87319ea** completed its 18 cluster cases on Juno/G2. Fourteen
passed, including all nine frozen v1 arithmetic failures, the frozen v1
operand-clear failure, `mix-B`, `addsub4-mix-B`, `addsub8-mix-B` and
`addsub16-nominal`. Every case had **zero wrong, double and missing rails**.
The four failures were exclusively surviving latch members after reset:

| Case | Transactions | Surviving members | Log |
|---|---:|---:|---|
| `second-pulse` | 160,000 | 120 | `v2-slurm-457312.out` |
| `order-100ms` | 160,000 | 106 | `v2-slurm-457313.out` |
| `order-15ms` | 160,000 | 98 | `v2-slurm-4018753.out` |
| `copy85` | 40,000 | 14 | `v2-slurm-457314.out` |

These logs are in the original checkout's `data/stage_d/opxor_cluster/`.
They name carry `add.fa0.c0`, sum output `add.fa0.s.y{0,1}`, and sum spent
`add.fa0.s.once0.spent` storage, outside v2's strengthened operand members.
This rejects v2's reset result; it is not qualification of v2 or F16.

### Control first: ordinary F sum one-shot, operand option off

Before changing the circuit, bounded RefSim controls ran all four scenarios
with **operand protection off**, sum one-shots and experimental autapses on.
These primitives have no rate-mode consumers requiring an F reader transform.
RNG **8585**, the original **256-copy blocks**, endpoint draws and all timing
jitter were retained. Option-off nodes use the v2 draw for the same role;
rewired positive inputs use the original receiving-edge draw. Removed operand
one-shot components still consume their original random draws. Preceding
blocks consume all four transactions' jitter without neural simulation.
Thus copies 406, 1098, 3725, 4082, 11001 and 16939 are the logged physical
draws, not new seeds or shortened random prefixes.

| Scenario | Small control: copies / transactions / survivors | Frozen copies | Frozen transactions | Option-off survivors | V2 survivors |
|---|---|---|---:|---:|---:|
| `second-pulse` | 8 / 32 / 0 | 406 | 4 | **8** | 8 |
| `order-100ms` | 8 / 32 / 0 | 406, 1098 | 8 | **6** | 8 |
| `order-15ms` | 8 / 32 / 0 | 406 | 4 | **8** | 8 |
| `copy85` | 8 / 8 / 0 | 3725, 4082, 11001, 16939 | 4 | **4** | 8 |

All controls had zero missing rails. The option-off `order-15ms` control also
had 4 wrong/double rails in the small sample and 2 in frozen copy 406;
`copy85` had 5 in the small sample and 4 across the frozen copies. These are
unprotected operand responses, separate from the reset accounting. Repeated
operand candidate ignition is expected with that option off. The first
controls took **5.25 seconds**; checking v2 against the original survivor
details brought the diagnostic to **7.34 seconds**.

**This is a shared ordinary-F reset weakness, not a three-tap harness clear.**
`build_pipeline` constructs each stage via `add_register`, which calls
`add_reset` with its default **four configured taps**. `extend_reset` connects
the arithmetic latch members to that stage's inhibitor at **0.75 loop**.
The operand probe uses the same `add_reset`, three relay neurons plus the
direct trigger tap, pulse weights and delays, and the same ordinary fan-out.
A structural regression compares those controller edges and carry/sum clear
weights with an actual sum-only F kernel stage.

Four configured taps do not guarantee four emitted spikes: the original
controller emits **three** on these endpoint draws, with cumulative counts
3/6/9/12 over four transactions. The ordinary-F weakness is demonstrated on
**3 of the 8 survivor copies**: carry copy 406 and sum copies 3725/4082 remain
active after those clears with the operand option off. V2 changes ignition
timing and which sum rail is active, so the other copies (1098 and copy85
11001/16939) survive only in v2 on these particular draws. Their ordinary sum
storage and clear wiring are still shared with F. Increasing only the storage
clear weights suppresses the survivors while retaining the controller and its
draws. Prescribing an extra inhibitor spike would not reproduce the kernel's
wiring and is not the remedy.

This finding matters for ordinary F builds too. **The default F circuit keeps
this known weakness** to preserve byte-identical defaults; these small controls
do not measure its frequency in a full kernel campaign.

### V3: opt-in carry and sum reset protection

`operand_xor_oneshot=True` now records **version 3**. It retains v2's consumed
operand/sum pulses and extends its **1.5-loop** storage clear to the adder's
carry rails, sum output rails and sum spent latches. Collection now covers the
whole ripple adder instead of stopping after its operand XORs. Each bit changes
**12 additional existing clear weights**, from 0.75 to 1.5 loop, alongside the
8 operand-member weights already strengthened by v2. Registered reader mirrors
receive the corresponding rounded gain. Source/hold inhibitors, controller
timing, SUB B-complement XORs and other stage storage retain their policy.

V3 adds **no nodes or edges over v2**, and does not alter the input image or
feed-forward timing. Stronger clears can change post-reset phase and recovery.
Versionless, v1, v2, boolean/string and unknown enabled
operand records are rejected on replay. Absent/false flags keep the original
netlist and records. Existing default/F frozen digests are not updated.

Both review findings are covered: Stage D diagnostic reconstruction validates
the nested operand flag/version before discarding version metadata, so a valid
top-level version cannot mask a missing nested version. The mirror branch of
`Gates.finalize_operand_reset` is tested with a real `Netlist` reader registered
both before and after ordinary reset wiring, asserting the old and strengthened
weights and their rounded gain.

Eight explicit cluster regressions, also available as smoke parameters, cover
every distinct copy in the four logs' capped `survivor_details`:

- `survivor-second-pulse-copy406`
- `survivor-order-100ms-copy406`
- `survivor-order-100ms-copy1098`
- `survivor-order-15ms-copy406`
- `survivor-copy85-copy3725`
- `survivor-copy85-copy4082`
- `survivor-copy85-copy11001`
- `survivor-copy85-copy16939`

Each runs the option-off control first, reproduces the v2 survivor count,
member roles, transaction and emitted reset counts, then requires zero
wrong/double/missing/repeated/surviving state with v3. Copy85 additionally
reloads the opposite EARLY polarity and clears again after the same recovery
interval. All eight frozen replays passed locally; the v3 copies total
**24 transactions**, including the four reloads. The eight new parameters
bring the cluster list to **26**, gated by `DROSO_OPXOR_CLUSTER=1`.

At the measured **2000-step recovery** after the 1.5-loop clears, post-clear
establishment latency was identical for v2 and v3: **102–154 steps**. Recovery
shorter than 2000 steps after the 1.5-loop clears is unmeasured.

Long cluster surveys remain unrun locally. **V3 is unqualified.** Stronger
clears must still be evaluated for recovery and reload margin in the full
cluster and kernel campaigns; no throughput or F16 claim follows.

### V3 foreground validation

The required command passed with **exit 0**, including the unchanged frozen
default/F digests, explicit-off comparisons, v3 replay/version rejection,
nested Stage D operand validation, controller wiring comparison, both mirror
registration orders and small real pipeline reset/rearm tests:

```sh
uv run pytest tests/test_build_options.py tests/test_ordered_xor.py \
  tests/test_operand_xor.py tests/test_config_f_kernels.py tests/test_stall_diag.py \
  -q -m 'not slow' -p no:warnings
```

The eight new frozen survivor parameters and two mirror tests passed in the
targeted foreground smoke run. The eight original small scenario parameters
also passed, **8 passed / 55 deselected in 13.83 seconds**, with zero
wrong/double/missing/repeated rails or surviving latch members:

```sh
DROSO_OPXOR_SMOKE=1 uv run pytest tests/test_operand_xor.py \
  -q -s -m 'not slow' -k 'survivor or mirror' -p no:warnings
DROSO_OPXOR_SMOKE=1 uv run pytest tests/test_operand_xor.py \
  -q -s -m 'not slow' \
  -k 'test_operand_xor_smoke and not replay and not survivor' \
  -p no:warnings -o addopts=''
```

The existing source-checkout environment was reused read-only with
`UV_PROJECT_ENVIRONMENT` and `UV_NO_SYNC=1`; `PYTHONPATH` selected this worktree,
and `UV_CACHE_DIR` pointed into writable task temp space. The required suite
took **roughly 13 minutes**, exceeding the approximate ten-minute target,
largely in existing full-program default digest builds. The additional
option-off controls, survivor/mirror probes and original small scenarios
together took **under a minute**. No full v1 replay sweep, cluster survey or
GPU/kernel campaign was run locally. `git diff --check` passes.

`git diff --check` passes. Only owned files are changed; no commit was made.
Full cluster requalification, F16 kernel campaigns, tighter recovery margins,
weight-dependent anatomical capacity and end-to-end performance remain to be
evaluated before any qualification claim.
