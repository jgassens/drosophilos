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
