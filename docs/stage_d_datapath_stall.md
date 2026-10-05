# Stage D: bit-level evidence for seed 108, copies 18 and 34

The captures localize two different blockages, but **do not establish the initiating
rail/kill mechanism**. Copy 18 loses the **Z-valid latch's ignition**, even though its
validity OR gate keeps firing. That problem is already visible during recovery from the
preceding transaction. Copy 34 trips the **Z fault gate while SUB is still rippling**;
the fault path discards the partial stage and the cell never retries. Its fault might be
a real double rail or a false rate-mode fault. The omitted rail and reset spikes prevent
choosing between these explanations. No netlist change or fix recommendation is made.

## Inputs, rebuild and interpretation

Read first: [copy 18](stage_d_copy18_stall.md), [copy 34](stage_d_copy34_stall.md),
[tick stall methods](tick_stalls.md), and the
[seed-109 fault and its corrective recapture](a2/datapath_fault_s109_n73.md).

All three input artifacts were read by absolute path under
`/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:

- `mixB_s108_c100_replay40.json`
- `mixB_s108_c100_replay40_copy18.npz`
- `mixB_s108_c100_replay40_copy34.npz`

The record rebuilds **29,375 neurons / 52,808 synapses**, with zero recorded role/ID
drift. It uses the generic datapath, true guards v2, `copy_requires_rail=True`, four-pulse
kernel/request clears at strength 0.75, delayed request repair, and **no `rate_robust`**.
It matches the supplied Juno 429332 / commit 09f2a01 replay description. The two compact
files contain 40,161,515 / 41,628,289 spikes and explicitly select **5,674 neuron IDs**.
Their inclusive windows are 0–2,473,209 / 1,611,372–4,261,465. Steps are 0.1 ms; neuron
IDs below are the recorded IDs. Transaction numbers count STARTs inside each window.

“Valid” below means the captured `Q.valid{i}.L.u` **OR indicator** rose. This does not
prove that exactly one data rail was live or that its value was correct. “Never” means
silent from the failed START through capture end, for a captured role. The data rails,
most ALU gates, and `Q.reset*` are **unavailable**, not silent. A fault-gate spike is not
proof of two live rails: the seed-109 recapture established a false fault from a single
correct fast rail. The 141-step train threshold is appropriate to latch trains; OR/AND
gate spikes can be farther apart and must also be inspected as raw event times.

## Copy 18: c3_xor transaction 2

START **3095** fires at **101,602**; ACT^d **12669** at **102,170**. Completion
**2695**, commit **28081**, DONE **3055**, every c3 stage fault gate **2569 + 7i**
(`i=0..10`), and fault latch **2740** have no spikes after START. No fault latch was
observed anywhere in this copy. The intended result is `30 XOR 200 = 214` (`0xD6`),
with C=Z=V=0, from the program and scripted second input; it is not a rail decode.

| Bit | `Q.valid` u ID | First valid spike | Last valid spike | Stage rail u IDs (r0 / r1), both unavailable |
|---|---:|---:|---:|---|
| R0 | 2564 | 103,742 | 2,473,173 | 2519 / 2521 |
| R1 | 2571 | 103,743 | 2,473,191 | 2523 / 2525 |
| R2 | 2578 | 103,715 | 2,473,208 | 2527 / 2529 |
| R3 | 2585 | 103,712 | 2,473,187 | 2531 / 2533 |
| R4 | 2592 | 103,725 | 2,473,167 | 2535 / 2537 |
| R5 | 2599 | 103,695 | 2,473,164 | 2539 / 2541 |
| R6 | 2606 | 103,724 | 2,473,164 | 2543 / 2545 |
| R7 | 2613 | 103,733 | 2,473,202 | 2547 / 2549 |
| C (8) | 2620 | 102,698 | 2,473,167 | 2551 / 2553 |
| **Z (9)** | **2627** | **never** | **never** | **2555 / 2557** |
| V (10) | 2634 | 102,659 | 2,473,200 | 2559 / 2561 |

Each listed non-Z indicator has one uninterrupted latch train through capture end. The
data completion subtree `c3_xor.Q.comp.c2_0.L.u` **2689** rises at **105,459** and
continues through **2,473,190**. The C/Z join (`c0_4.and` **2664**, latch **2665**)
never fires, nor does its parent **2683** or the final completion **2695**. Thus this
is a Z-valid blockage, not a failure of the data completion subtree.

### The validity gate fires; its ignition relay does not

| Probe | ID | Captured evidence |
|---|---:|---|
| `Q.valid9.or` | 2626 | Transaction 1 first spike 48,377; 16,245 spikes after transaction-2 START, first 101,669, last 2,473,092 |
| `Q.valid9.ign.edge` | 2629 | **Only one spike in the entire dump: 48,420** |
| `Q.valid9.L.u` / `.v` | 2627 / 2628 | Transaction-1 trains 48,463–55,705 / 48,519–55,669; neither member ever fires again |
| `Q.valid9.ign.edge_inh` | 2630 | Resumes at 56,363; 17,686 spikes after START, first 101,689, last 2,473,112 |
| `Q.valid9.ign.hold_inh` | 2631 | Last spike **55,779**, no transaction-2 spikes |
| preceding `done.edge` | 3055 | **55,558** |
| preceding `Q.ready` | 2721 | **56,432**; no further READY |

The OR gate has a **708-step gap**, 55,608→56,316, around the preceding clear, then
fires **310 times between steps 56,000 and 101,602** (all starting at 56,316). Its
inhibitor has a 711-step gap, 55,652→56,363. This is not an uninterrupted OR train
through reset: it **returns during reset recovery**, before READY, and persists long
before transaction 2 supplies operands. The validity latch itself stays dark.
After READY and before START, **only bit 9's validity OR** fires (309 spikes); the
other ten validity ORs are silent throughout that interval.

The wiring identifies the precise blocked ignition path:

```text
Q.b9r0.u 2555 ---- +766 ----+
                           +--> valid9.or 2626 -- +4655 --> ign.edge 2629 -- +4655 --> valid9.L.u 2627
Q.b9r1.u 2557 ---- +766 ----+          |
                                     +-- +3621 --> edge_inh 2630 -- -7966 --> 2629
valid9.L.u 2627 -- +3621 --> hold_inh 2631 ------------------------- -905 --> 2629
Q.reset_inh 2701 ---------------------- -2716 --> 2626, 2627, 2628
```

These are nominal quanta, with 18-step synaptic delays; the copy's noisy weights and
membrane voltages are not recorded. The continuously firing inhibitor is an observed
negative input to the silent ignition relay. The validity-latch hold inhibitor is silent
after 55,779, although its residual inhibition could matter at the first return. The
capture cannot decide whether the first missed ignition was due to recovery, the
relay/inhibitor race, or another perturbation. It also cannot identify which Z rail
supplied the resumed OR activity or whether it survived clear or was re-ignited.

### What the missing-bit datapath capture does and does not show

The XOR's input A is `c2_sel.M.b{i}r{r}`: u IDs **2165 + 4i + 2r**, v=u+1;
B is `K.k200.b{i}r{r}`: u IDs **660 + 4i + 2r**, v=u+1 (`i=0..7`). All are
omitted. Their presence and exclusive validity at ACT^d are unknown. The eight data
valid indicators and their completed subtree are downstream evidence, not substitutes
for operand-rail capture.

Z0 is driven from the eight `Q.b{i}r1.u` rails by `alu.z0.r{i}.edge`
**14505 + 2i**, with inhibitors **14506 + 2i**. Z1 is driven from data completion
**2689**, through `alu.zd.d0..2` **14499–14501**, to `alu.z1.edge` **14502**;
its source inhibitor is **14503**, and its all-data-r1 veto is **14504**.
**All those Z drivers and the four Z-latch members 2555–2558 are omitted.**
Consequently:

- The Z-valid OR gate fired; its ignition edge and both valid-latch members did not.
- Whether either Z **data latch** ignited, survived reset, or was killed/re-lit is unknown.
- `Q.reset` **2700** and `Q.reset_inh` **2701** are omitted. The preceding DONE/READY
  bracket a clear; they do not record its pulse timings or effects on the Z rails.
- START's captured kill train **12655** fires at 101,695–101,915. Its wired targets
  are IDLE/request-true latches, **not the Z rails or valid latch**. It cannot simply be
  named as this bit's killer. No transaction-2 commit kill train fires.

## Copy 34: c5_sub transaction 6 in the window

START **4249** fires at **1,909,328**, ACT^d **15044** at **1,909,917**.
The record's preceding master commits are `c9_sel=64` at 1,890,170 and `c4_sel=66`
at 1,907,355, so the intended subtraction is `64 - 66 = 254` (`0xFE`), C=Z=V=0.
Those prior decoded commits do not establish the rails' state at sampling time.

| Bit | `Q.valid` u ID | First valid spike | Last valid spike (count) | Stage rail u IDs (r0 / r1), unavailable |
|---|---:|---:|---|---|
| R0 | 3718 | 1,911,594 | 1,912,578 (21) | 3673 / 3675 |
| R1 | 3725 | 1,911,937 | 1,912,503 (12) | 3677 / 3679 |
| R2 | 3732 | 1,911,849 | 1,912,503 (14) | 3681 / 3683 |
| R3 | 3739 | 1,912,190 | 1,912,491 (6) | 3685 / 3687 |
| **R4** | **3746** | **never** | **never** | **3689 / 3691** |
| **R5** | **3753** | **never** | **never** | **3693 / 3695** |
| **R6** | **3760** | **never** | **never** | **3697 / 3699** |
| **R7** | **3767** | **never** | **never** | **3701 / 3703** |
| **C (8)** | **3774** | **never** | **never** | **3705 / 3707** |
| Z (9) | 3781 | 1,911,863 | 1,912,583 (15) | 3709 / 3711 |
| V (10) | 3788 | 1,911,535 | 1,912,531 (20) | 3713 / 3715 |

The one captured completion-tree latch is the R0/R1 join **3795**, with spikes
**1,912,340, 1,912,419, 1,912,482**. Root **3849** never fires. Commit **28341**
and DONE **4209** never fire for this transaction.

### Fault, disappearing validity, and late bit 4

| Event | ID / role | Step(s) |
|---|---|---|
| Z-valid OR | 3780, `Q.valid9.or` | 1,911,781; 1,911,919; 1,912,037; 1,912,156; 1,912,277; 1,912,383 |
| Z-valid ignition | 3783, `Q.valid9.ign.edge` | 1,911,823 |
| **Z fault gate** | **3786, `Q.fault9.and`** | **1,912,340**, one spike |
| FAULT latch u | 3894, `faultL.u` | **1,912,381; 1,912,460** |
| FAULT latch v | 3895, `faultL.v` | **1,912,434** |
| R4-valid OR | 3745, `Q.valid4.or` | **1,912,455**, one spike |
| R4-valid ignition | 3748, `Q.valid4.ign.edge` | **1,912,502**, one spike |
| R4-valid source inhibitor | 3749, `Q.valid4.ign.edge_inh` | 1,912,509 |
| R4-valid u / v | 3746 / 3747 | **neither ever fires** |
| post-fault READY | 3875, `Q.ready` | **1,913,270** |

Bits R5, R6, R7 and C have neither a valid-OR nor an ignition-edge spike. Their
OR / ignition-edge IDs are **3752/3755, 3759/3762, 3766/3769, 3773/3776**.
All other captured stage fault gates are silent; no master fault gate fires in this copy.
This is the dump's one FAULT-latch activation, consistent with the record's **one
run-level fault** despite **zero runner-counted refusals**. It is an internal cell's
stage fault, not an observed host-input refusal/retry. The reported completed output
counts remain 3 for copy 18 and 94 for copy 34, with zero wrong values in both.

The Z fault gate reads Z0 **3709** and Z1 **3711**, at **211 nominal quanta each**
(`add_and_gate(fraction=0.55)`). Both rails and their drivers were omitted. Its firing
does **not** distinguish a real second rail, a fast correct rail plus stray drive, or
another analogue disturbance. In particular, the Z-**valid** latch's intervals are not
the Z-**data** latch's firing rate. The earlier seed-109 false fault is a reason to
recapture this distinction, not evidence that it happened again here.

The observed fault→loss of validity→READY sequence supports a **fault-driven stage
discard**. The exact topology is:

1. Gate **3786** ignites FAULT **3894/3895**. FAULT directly inhibits completion
   root members **3849/3850**, its AND **3848**, and ignition edge **3851**.
2. FAULT excites `Q.reset` **3854** and reset-edge inhibitor **3856**. The reset
   controller is wired to drive `Q.reset_inh` **3855** through its four-pulse chain
   and eventually READY **3875**. **3854–3859 were not captured**; exact reset pulse
   steps cannot be reported from these dumps.
3. In the rebuilt netlist, **3855** inhibits **453 neurons**: stage rail members,
   validity latches/OR gates and completion latches/AND gates; FAULT **3894/3895**;
   ACT **3876/3877**; COMMIT **4099/4100**, grant **4114/4115**, COPY **4119/4120**;
   and the resettable SUB/operand/ripple/mux/flag latches installed by `extend_reset`.
   Its edges do not directly clear the source masters, c5's master, request pairs or
   IDLE. Zero/output edge relays and delay chains are not all direct reset targets.
4. FAULT also inhibits the grant latch and vetoes grant/COPY arms. There was **no
   commit in flight here**. The preceding seed-109 empty-stage-copy race is not this
   sequence. The stage's partial computation is lost before autocommit can request
   commit; no DONE/idle recovery or new START follows.

R4's valid ignition attempt occurs after FAULT and amid the disappearance of the other
valid trains. The reset path is the supported explanation for discarding this partial
word. It is **not a directly captured kill train**, and the silent R4 valid latch is not
evidence that its data rail never fired. Whether the data latch or ripple sum ignited,
and which exact pulses killed it, remains unavailable. The six formerly live validity
trains demonstrably stop; their associated data-latch trains are not recorded.

### Missing operand and computation evidence

At the datapath boundary, A is `c9_sel.M.b{i}r{r}` (u **6204 + 4i + 2r**, v=u+1),
and B is `c4_sel.M.b{i}r{r}` (u **3319 + 4i + 2r**, v=u+1), `i=0..7`.
All are omitted, so exclusive validity at ACT^d is unknown. The required local paths
are `c5_sub.a{i}r{r}` / `b{i}r{r}`, their operand-gate edges/vetoes, delayed A/B,
`bx{i}`, `fa{i}.x`, carry and carry-delay chains, `fa{i}.s`, unit-select rails,
`mux{i}r{r}u0`, mux latches, and final output edges. None of these computing gates
is captured. The final R4–R7 output-edge IDs, in r0/r1 order, are
**16853/16855, 16857/16859, 16861/16863, 16865/16867**; C uses **16869/16871**.

The Z data rails **3709–3712** receive Z0 edges **16883 + 2i** (inhibitors
**16884 + 2i**), or Z1 edge **16880** with inhibitor **16881** and veto **16882**,
from delay chain **16877–16879**. These roles are also omitted. Capturing both latch
members, both polarities, every ignition source and their inhibitors is necessary to
distinguish a fast single rail from a transient opposite rail and reset/re-ignition.

## Exact recaptures

Run at the replay's netlist (09f2a01, same netlist as ad4046d), with the same CUDA
backend and default float64. **Keep all 100 copies**; the device stray stream depends
on batch size. These commands shorten only the token/time budget; no `--rate-robust`
or other build change is requested. Their horizons extend beyond the failures while
retaining the relevant preceding clear. With these explicit budgets the default stall
watch will not freeze the selected copy before the horizon; a resulting `max_ms` run
status is expected and does not invalidate its diagnostic window.

The regexes retain the original control filter and add the whole blocked cell (all
ALU/latch/kill/reset/COPY roles), source masters, relevant constants, and **all** `out.*`
and `alu.z*` roles. Those latter names repeat across cells; IDs and incoming edges
disambiguate them. Both regexes were checked against every signal selected by the new
datapath view: no required signal is excluded.

Copy 18 (includes transaction 1 and its reset at ~5.56 s, and failure at ~10.16 s):

```sh
uv run python -m drosophilos.bench.stage_d --ticks 3 --seed 108 \
  --backend torch-fast --device cuda --copies 100 --mix B --max-ms 30000 \
  --dump-copies 18 \
  --dump-roles '\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)|^(?:c3_xor\.|c2_sel\.M\.|K\.k200\.|out\.|alu\.z)' \
  --dump-out data/stage_d/mixB_s108_c100_dp18 \
  --out data/stage_d/mixB_s108_c100_dp18.json
```

Copy 34 (40-token prefix; failure at ~190.93 s, prior 30 s retained):

```sh
uv run python -m drosophilos.bench.stage_d --ticks 40 --seed 108 \
  --backend torch-fast --device cuda --copies 100 --mix B --max-ms 205000 \
  --dump-copies 34 \
  --dump-roles '\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)|^(?:c5_sub\.|(?:c9_sel|c4_sel)\.M\.|out\.|alu\.z)' \
  --dump-out data/stage_d/mixB_s108_c100_dp34 \
  --out data/stage_d/mixB_s108_c100_dp34.json
```

Expected **uncompressed NPZ planning sizes are approximately 0.1–0.2 GB for copy 18
and 0.15–0.3 GB for copy 34**. These are estimates, not measurements of omitted spikes:

| Copy | Selected IDs (additional) | Expected retained steps | Existing-filter bytes in that interval | Added bytes if every additional ID fires once per nominal 47 steps |
|---|---:|---|---:|---:|
| 18 | 8,854 (3,180) | 0–300,000 | 38.46 MB | 162.38 MB |
| 34 | 9,003 (3,329) | 1,611,372–2,050,000 | 55.24 MB | 248.54 MB |

Compact storage is 8 bytes/spike plus the small role table. Many added opposite rails
and deselected gates stay dark, so the all-added-active estimate is conservative at
nominal rate; it is not a hard upper bound under fast firing. Allow headroom for chunk
rounding and noisy rates. The original ~322/334 MB files cover much longer blocked
tails. These recaptures have not been run in this worktree.

## Reusable diagnostic and validation

`stall_diag.analyze_dump` now adds `report['datapath']` when its existing classifier
finds a blocked stage. Existing classification, neighbour timelines and accounting are
retained. The Markdown adds per-bit validity/rail/fault evidence, operand activity at
ACT^d, control/reset probes, validity OR/ignition/inhibitor probes and final rail drivers.

`datapath_view(pl, dump, cell_name, start, stop=None)` is also callable for any selected
transaction. Its structured view contains all upstream positive **and negative** edges,
nominal weights/delays, both latch members, train intervals, spike samples/counts, ISIs,
reset/fault targets and explicit unavailable IDs. Source registers and control/reset
inputs bound the trace, so a feedback loop does not silently mix other transactions.
It follows actual neuron IDs rather than assuming `out.*` or `alu.z*` names are unique.
It records evidence without labelling a stopped train as a proven kill or a fault as a
proven double rail. Example using either new capture:

```sh
uv run python -m drosophilos.bench.stall_diag \
  data/stage_d/mixB_s108_c100_dp18_copy18.npz \
  --campaign data/stage_d/mixB_s108_c100_dp18.json --node 18
```

Synthetic tests cover captured silence versus omitted roles, a live validity gate with
a silent ignition edge, partial words discarded around a fault/reset, an ignition edge
whose target stays dark, pre-existing trains and transaction boundaries, repeated role
names, and specialized/rate-conditioned topology traversal. They describe diagnostic
evidence, not a speculative failure simulation.

The required command is:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_stall_diag.py tests/test_datapath_stall.py
```

The sandbox denies uv's default cache (`~/.cache/uv/sdists-v9/.git`). Validation uses
the existing repository virtualenv without syncing it, a writable task-local uv cache
and pytest basetemp, and this worktree on `PYTHONPATH`: **all 28 tests pass**. The two
real dumps were also passed through the extended analyzer and Markdown renderer, and
both retain their original classifications with the bit sets above. `git diff --check`
passes. No commit is attempted; the orchestrator owns integration and commits.

## Fix decision

**Defer a primitive/netlist fix until recapture.** For copy 18 the unresolved depth is
the Z rail/reset/driver and validity ignition path (`lib/alu.py`,
`protocol/celement.py:_ignite_from`, `protocol/latch.py:add_edge_relay` / reset).
For copy 34 it is the Z rail/driver versus the rate-mode fault detector, followed by
the already identified stage fault-reset path. The evidence does not justify choosing
one of those primitives, changing its weight/timing, or estimating a fix's cost.

The control true-rail guards completed normally before both STARTs. START/request kill
trains do not clear these data latches. `copy_requires_rail` protects COPY from an empty
stage, but neither failed transaction reaches commit; disabling it would not repair the
observed path. `rate_robust` changes fault/completion readers and their reset coupling
and must be tested as a different netlist; its existence is not evidence for a fast-rail
trigger here. Preserve these protections while collecting the missing evidence.

## Recapture (2026-10-04)

This section supersedes the unresolved base-copy mechanisms and deferred fix decision
above. **Base 18 retains Z0 through its preceding stage clear; base 34 falsely faults
on a fast, single-railed Z0.** The two rate-robust datapath stalls are on different
bits and need further capture; neither establishes a Z failure.

### Artifacts and scope

Read-only inputs are under
`/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`. Each NPZ was loaded
with `stall_diag.load_dump` against the pipeline rebuilt from its JSON. All four have
**zero role/ID drift**. The base rebuild is 29,375 neurons / 52,808 synapses, matching
09f2a01/ad4046d; the rate-robust v2 rebuild is 30,643 / 55,936. Both retain true guards
v2, `copy_requires_rail`, and the recorded four-pulse, 0.75-strength kernel policy.

| JSON / NPZ prefix (`mixB_s108_c100_…`) | Copy | Inclusive steps | Spikes | Selected IDs |
|---|---:|---|---:|---:|
| `dp18` / `dp18_copy18.npz` | 18 | 0–299,999 | 9,606,850 | 8,854 |
| `dp34` / `dp34_copy34.npz` | 34 | 1,611,372–2,049,999 | 11,676,009 | 9,003 |
| `rr_replay870` / `rr_replay870_copy34.npz` | 34 | 33,780,223–35,904,403 | 36,339,748 | 6,037 |
| `rr_replay870` / `rr_replay870_copy43.npz` | 43 | 25,302,390–27,426,543 | 43,670,231 | 6,037 |

The base files occupy **77,991,713 / 94,561,641 bytes**, including role tables. Their
shorter budgets end at `max_ms`, before the stall watch freezes them. The START, ACT^d,
fault and preceding-clear times agree with the original replay. This is a truncated
capture of an independently localized persistent blockage, not a stall inferred from
the time limit. The rr windows did freeze, 100,000 steps after detection. All steps
below are 0.1 ms. Synaptic weights below are **nominal quanta** unless explicitly
identified as reconstructed static draws; every listed synaptic delay is **18 steps**.

### Base 18: Z0 survives clear; the returning OR cannot re-arm its ignition

The first c3 result is nonzero. Four independent Z0 drivers fire at **48,207 (14517,
R6), 48,208 (14505, R0), 48,211 (14519, R7), and 48,220 (14513, R4)**. Each has a
`+4655` edge into Z0 u **2555**. This is four ignition pulses into one latch, although
each individual driver's edge relay fires only once. Z0 first fires at **48,233**,
then **48,262**; its partner **2556** first fires at **48,283**. Before clear, both
members reach a **35-step period**, against the nominal 47.

| Event | Captured evidence |
|---|---|
| DONE / stage reset trigger | **3055: 55,558** → `+4655` → **2700: 55,600** |
| Stage reset inhibitor | **2701: 55,661; 55,718; 55,772; 55,834** |
| Inhibition delivered to Z0 u/v | `-2716` on each member, at **55,679; 55,736; 55,790; 55,852** |
| Z0 u through and after clear | **55,615; 55,650; 55,690; 55,731; 55,782; 55,852; 55,961; 56,032; 56,090; 56,142** |
| Z0 v through and after clear | **55,612; 55,647; 55,685; 55,727; 55,780; 55,890; 55,996; 56,062; 56,118; 56,169** |
| Z-valid latch really stops | **2627** last **55,705**; **2628** last **55,669** |
| Z-valid OR / source inhibitor return | **2626: 55,608 → 56,316**; **2630: 55,652 → 56,363** |
| READY | **2721: 56,432**, while Z0 is still live |

The Z0 members have single train intervals **48,233–299,961 (5,762 spikes)** and
**48,283–299,986 (5,758 spikes)**. Their largest gaps across clear are **109 / 110
steps**, followed by recovery to roughly 45-step periods. The spike at 55,852 precedes
that step's synaptic delivery in the simulator; the later spikes prove survival beyond
the final pulse. Neither Z1 member **2557/2558** fires. All eight Z0 ignition edges are
silent between the initial burst and transaction 2's result; the next is **14509 at
103,403**. Thus no captured Z driver re-lights a cleared latch. The latch continues
through the clear, supported by the reciprocal `2555 ↔ 2556` **+3621** loop. Background
input is not recorded, so this does not exclude its contribution to the survival margin.

The reset clears the validity latch and temporarily suppresses its OR, while leaving
the OR's Z0 input alive. That distinction explains the later ignition failure:

```text
surviving Z0 2555 -- +766 --> OR 2626 -- +4655 --> ignition 2629 -- +4655 --> valid u 2627
                                |
                                +-- +3621 --> inhibitor 2630 -- -7966 --> 2629
valid u 2627 -- +3621 --> hold inhibitor 2631 -------------------  -905 --> 2629
reset 2701 -- -2716 --> OR 2626 and valid members 2627/2628
```

**2629 is not a direct reset target.** It retains inhibition accumulated during the
old gate train. At the first returning OR spike, the source inhibitor has been quiet
for only 664 steps (66.4 ms), and the hold inhibitor for 537 steps (53.7 ms). Its last
hold-inhibitor spike is **55,779**. The returning excitation arrives at **56,334**;
fresh source inhibition arrives at **56,381**, before ignition succeeds. Thereafter
the surviving Z0 keeps the OR and source inhibitor firing, so the missed first edge
does not get a fresh recovery interval. The hold inhibitor remains silent.

A small receiving-neuron calculation checks this explanation without claiming a CUDA
replay. Reconstructing `make_perturbed_sim`'s static draws with seed 108, **B=100** and
the full base topology gives 2629 input quanta **+4490 / -8027 / -915**, threshold
**-45.102508 mV**, bias **-0.108463 mV**. Prescribe its captured input spikes, including
the 18-step delays, to RefSim from step 47,000, omitting target strays. It reproduces
the initial ignition at **48,420** and none after clear. At 56,316 its modeled voltage
is **-57.548 mV**; just before the new inhibition takes effect it reaches only
**-45.750 mV**, below threshold. Remove the preceding input history while keeping the
same returning OR/inhibitor spikes, and it fires at **56,359**. These voltages are
model calculations, not recorded measurements; the captured absence of 2629 and the
inhibitory spike arrivals are the direct evidence.

At transaction-2 ACT^d **102,170**, the captured operand rails encode **30 and 200**
exclusively. The data rails subsequently encode **214**, C=V=0; Z0 is still the old
survivor. Z-valid never returns, the C/Z completion join stays dark, and no commit,
DONE or fault follows. The root problem is **failed stage clear followed by one-shot
re-arm failure**, not a missing Z computation at transaction 2 or a request-clear train
targeting Z. Increasing the validity relay's drive alone would conceal surviving state.

### Base 34: a false Z fault discards a correctly forming SUB

The captured source masters at ACT^d **1,909,917** encode A=**64**, B=**66**,
exclusively. The partial result agrees with `64 - 66 = 254`. Z0 u/v **3709/3710**
start at **1,911,587 / 1,911,641**; **both Z1 members 3711/3712 and its sole driver
16880 are silent throughout the failed transaction**. The data completion subtree
has not completed, so there is no valid zero-result ignition waiting behind it.

The Z0 driver path is the same eight-way fan-in as c3's, now fed by a ripple:

| Data-1 rail u | Rail first spike | Z0 driver | Driver spike → arrival at 3709 |
|---|---:|---:|---|
| R2, **3683** | 1,911,507 | **16887** | **1,911,549 → 1,911,567** |
| R1, **3679** | 1,911,607 | **16885** | **1,911,652 → 1,911,670** |
| R3, **3687** | 1,911,853 | **16889** | **1,911,892 → 1,911,910** |
| R4, **3691** | 1,912,200 | **16891** | **1,912,239 → 1,912,257** |

Each rail drives its edge at **+4655**, its source inhibitor at **+3621**, and that
inhibitor returns **-7966** to the edge. Each edge supplies **+4655** to 3709, in
addition to the **+3621** reciprocal Z0 loop. Thus later result bits deliver extra
ignitions to an already-live Z0. Its u spikes before the fault include:

```text
1,912,000  036  072  108  144  180  216  252  283  315
          (same 1,912, prefix; final intervals 31 and 32 steps)
```

Fault gate **3786** receives **+211** from each of **3709 and 3711**, with no other
netlist inputs. Only 3709 supplies spikes. The fault at **1,912,340** occurs seven
integration steps after its **1,912,315 → 1,912,333** arrival. This is a **false
rate-mode AND fault on a single rail**, not a real double rail. At 32-step periods a
single nominal 0.55-rate input contributes roughly `0.55 × 47/32 = 0.81` of the
nominal sustained threshold need, reducing its background-input margin.

The dump does **not** locate a particular stray pulse. A calculation using the static
copy-34 draws (Z0→fault **206 q**, Vth **-45.415768 mV**, bias **+0.646589 mV**) and
captured rail inputs without background reaches only about **-46.540 mV** before
the fault. The fast rail alone is therefore not a demonstrated sufficient trigger.
The regression explicitly adds *synthetic* background arrivals to demonstrate this
false-fault class, rather than inventing observed stray times or a transient Z1 rail.

The discard is now directly captured:

| Path/event | Wiring and step evidence |
|---|---|
| Fault → FAULT | **3786 → 3894**, +4655; FAULT u **1,912,381; 1,912,460**, v **1,912,434** |
| FAULT → stage reset | **3894 → 3854**, +4655; trigger **1,912,422**; **3894 → 3856**, +3621, supplies its edge inhibitor |
| Reset train | **3855: 1,912,476; 1,912,522; 1,912,568; 1,912,614; 1,912,717** |
| First inhibition reaches reset targets | **1,912,494**, -2716 into both stage-rail/valid-latch members and resettable ALU state |
| R4 validity attempt | edge **3748: 1,912,502**, +4655 arrives at **1,912,520** amid the clear; valid members **3746/3747** stay silent |
| R5 result attempt | edge **16859: 1,912,483**, +4655 arrives at **1,912,501**, after the first clear; rail members **3695/3696** stay silent |
| Stage READY | **3875: 1,913,270** |

The reset controller has four wired taps (trigger and three relays, each **+3621**
into 3855); it emits **five observed inhibitor spikes**, so the table reports actual
events rather than imposing the nominal pulse count. R4's data rail did fire before
the fault; its missing validity is not evidence of a missing R4 computation. The
reset's 453 inhibitory targets include the partial ripple, ACT, FAULT and staged
commit state. FAULT also directly suppresses completion and grant at **-5432**.
No commit was in flight, no master COPY occurred, and no new START follows. This is
an internal-cell discard with no retry, distinct from the old empty-stage COPY race.

### Rate-robust 34 and 43: different observed bit failures; initiators unresolved

| Copy / cell | Established from the default-role dump | What it does not establish |
|---|---|---|
| rr34 / `c10_xor`, tx5 | START **8134: 34,029,813**; ACT^d **24651: 34,030,394**. **R3 fault 7553** fires **34,031,982**; FAULT **7737/7738** rises **34,032,023 / 34,032,073**. All eleven validity indicators rise then stop; READY **7718: 34,032,896** follows. Z-valid **7600** fires **34,031,979; 34,032,068; 34,032,135**; Z fault **7607** is silent. | The R3 data rails, rate readouts and reset spikes are omitted. A true double rail, erroneous reader activity and a false conditioned fault remain distinguishable only by recapture. |
| rr43 / `c8_sub`, tx5 | START **6812: 25,583,794**; ACT^d **22180: 25,584,355**. Only **R1** fails validity. OR **6205**, valid **6206/6207**, ignition **6210**, source inhibitor **6211**, hold inhibitor **6212** are all silent after START. All fault gates and FAULT **6415/6416** are silent. Z-valid **6278** holds **25,586,337–27,426,517**. | Both R1 rails, their output relays, rate readers and upstream computation are omitted. Silence of the OR cannot locate the failure in the arithmetic, output ignition, rail storage or readout. |

For rr34, the relevant inputs are **R3 r0 7486/7487** and **r1 7488/7489**,
driven by output edges **26456/26458**. Their rate taps **7548/7549** each receive
**+28968** from the respective u rail and **-43456** from stage reset **7698**.
These taps, not the raw rails, drive fault **7553** at **+264** each and valid OR
**7545** at **+814** each. Both gates have nominal bias **-7 mV** (the doubled-gap
reader). R3's OR accelerates to 52-step intervals while neighbouring ORs are roughly
90–116 steps apart: useful evidence of excess drive, **not proof of two data rails**.
FAULT → reset **7697** is +4655; reset-inhibitor **7698** feeds the stage reset
domain, including the already-present v2 reader-clear mirrors.

For rr43, the analogous chain is mux u **23709/23726** → output edge **23977/23979**
→ R1 u **6156/6158**, with **+4655** on each link. Rail partners are **6157/6159**.
Rate taps **6208/6209** receive +28968 from the rails and -43456 from reset **6376**;
they drive OR **6205** at +814 and fault **6213** at +264. The OR-to-ignition path
has the same +4655 / -7966 / -905 structure as base18, but here **the OR and both
inhibitors are silent**, unlike base18's live OR/source inhibitor. The failure is
upstream of a usable R1 validity signal; it is not evidence for that Z-valid re-arm
mechanism. No primitive fix is recommended for either rr initiator yet. rr34 shares
base34's *fault/discard sequence*, not an established cause of the fault.

### Exact rr recaptures and expected sizes

Use the recorded **rate-robust v2 netlist**, torch-fast/CUDA/float64 and **100 copies**.
These budgets retain the preceding transaction and the failed computation while
shortening the blocked tail. They leave the default 30 s pre-commit / 10 s post-stall
window settings intact; `max_ms` will arrive before the inferred stall watch plus
post-window, so an unfrozen, resource-truncated record is expected. Do not switch to a
single-copy run or compare neuron IDs with the base build.

```sh
python -m drosophilos.bench.stage_d --ticks 610 --seed 108 \
  --backend torch-fast --device cuda --copies 100 --mix B --rate-robust \
  --max-ms 3430000 --dump-copies 34 \
  --dump-roles '\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)|^(?:c10_xor\.|(?:c9_sel|c4_sel)\.M\.|out\.|alu\.z)' \
  --dump-out data/stage_d/mixB_s108_c100_rr_dp34 \
  --out data/stage_d/mixB_s108_c100_rr_dp34.json

python -m drosophilos.bench.stage_d --ticks 460 --seed 108 \
  --backend torch-fast --device cuda --copies 100 --mix B --rate-robust \
  --max-ms 2580000 --dump-copies 43 \
  --dump-roles '\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)|^(?:c8_sub\.|c9_sel\.M\.|K\.k2\.|out\.|alu\.z)' \
  --dump-out data/stage_d/mixB_s108_c100_rr_dp43 \
  --out data/stage_d/mixB_s108_c100_rr_dp43.json
```

The regexes include **every signal in the respective 2,179-neuron datapath view**,
including raw rails and `.rate` readouts. Repeated `out.*` and `alu.z*` roles remain
disambiguated by their incoming edges. Expected **uncompressed** planning sizes are
roughly **0.1–0.4 GB per copy**; reserve at least **0.5 GB each**, with extra memory
for the writer's array copies. The following estimate assumes *every* added ordinary
neuron fires once per 47 steps and every added rate tap once per 25 steps:

| Copy | Selected / added IDs (added rate taps) | Expected retained steps | Measured existing-filter payload in that interval | Added-payload estimate | Total estimate |
|---|---|---|---:|---:|---:|
| rr34 | 9,514 / 3,477 (148) | 33,780,223–34,299,999 | 75.98 MB | 319.14 MB | ~0.40 GB |
| rr43 | 9,326 / 3,289 (106) | 25,302,390–25,799,999 | 79.75 MB | 286.48 MB | ~0.37 GB |

Many opposite rails and inactive ALU arms will be silent. These are estimates, not hard
bounds on fast/noisy firing; int32 steps plus neuron IDs cost eight bytes per spike.
The commands have not been run here. No additional spike-role recapture is needed to
distinguish survival versus re-lighting in base18 or single versus double rail in base34.

### Fix recommendations supported by these captures

**Base18 needs a storage-clear margin fix, not a control-guard or validity-OR repair.**
The responsible primitive is `protocol/latch.py:add_reset`, used by the staged register
in `protocol/handshake.py:add_register`. It already defaults to four taps independently
of `Drive.kill_pulses`: merely changing `kernel_kill_pulses` would not fix this reset.
Qualify the stage clear against the observed 35-step, multiply-ignited Z0 state and
weak/independent clear edges. A **stage-only trailing fifth tap** is a concrete first
candidate: **+1 neuron / +2 synapses per affected reset controller**, using the same
inhibitor fan-out, and about **one pulse-hop (~5.3 ms)** more clear time. If reload
recovery requires one extra READY hop, that adds **+1 neuron / +1 synapse** and ~5.3 ms
to READY. These are design costs, **not evidence that five pulses suffice**; select the
train only after the phase/reload regression below. Do not globally strengthen the
machine's three-pulse clears or shorten READY based on this copy.

`lib/alu.py:add_zero_flag` also supplies a concrete source of extra drive: eight
independent relays can ignite Z0 repeatedly. The implemented optional `zero_once`
circuit in `lib/alu.py:add_zero_flag` uses an OR of the eight data-1 taps feeding one
common ignition/hold circuit into the existing Z0 latch. It needs **4 neurons /
19 synapses**, replacing **16 / 32** for the eight legacy relays: **-12 neurons /
-13 synapses** in base mode per 8-bit Z generator. The nineteen include the OR reset,
three resets for the relay and its two inhibitors, and the hold-inhibitor quench of
the OR needed for re-arming at READY. It adds the OR's rate-integration latency
(order tens of ms for one active input). `tests/test_zero_flag_once.py` measures
latency and checks simultaneous inputs, Q-domain reset and reload at READY.
This removes the observed multiple
ignition paths, but **does not by itself prove the fast-latch clear margin**. A local
calculation with base18's static latch draws and captured ignition times runs at 37
steps after the burst versus 45 after just the first pulse; without the unrecorded
background it clears successfully in both cases, so it is not a replay of the survivor.

**Base34 supports fixing the rate reader at `protocol/celement.py:add_and_gate` /
`protocol/rate.py`, preserving true-double-rail detection.** The existing v2
`rate_robust` design is the concrete candidate: refractory-limited, reset-coupled taps
prevent one fast latch from counting as an extra logical input. For an isolated Z
fault/valid reader pair with two previously unconditioned rails and one reset source,
this costs **+2 neurons / +4 synapses** (two source→tap and two mirrored resets;
existing gate-input edges are rewired), plus an extra 18-step synaptic hop and tap
charging time. End-to-end gate latency changes with integration and must be measured.
Enabling the existing whole Stage-D option costs **+1,268 neurons / +3,128 synapses**;
it is not a claim that the same noisy copy will then pass or that every false fault
is eliminated. rr34 is a reason to inspect the conditioned circuit, not evidence that
its fault had the same fast-single-rail trigger.

These proposals preserve true-rail guards, **4 × 0.75 kernel/request kill trains**,
**3 × 0.75 machine trains**, and `copy_requires_rail`. The optional fifth stage-reset
tap is a separate, explicitly scoped experiment. START/ACT^d succeeded in every case;
the relevant START kills target IDLE/request state. `copy_requires_rail` protects a
later COPY but cannot restore a stage discarded before commit. With `rate_robust`,
additional stage clear pulses must also reach the existing mirrored reader resets;
readout conditioning does not change the underlying storage loop or guarantee its
clear. Any shared Z0 OR must likewise be conditioned/reset consistently in that mode;
the base-mode reduction cost above excludes newly needed rate taps.

### Regression and validation

No diagnostic extension is needed: `datapath_view` already exposes both rail members,
all signed inputs, reset targets, availability and train intervals. Use the preceding
START as its lower bound when inspecting a clear; a view beginning at the failed
START alone cannot explain the old Z survivor.

Two small RefSim regressions were added to `tests/test_datapath_stall.py`:

- Feed the recorded 55,000–57,000 OR/source-inhibitor/hold-inhibitor arrivals into the
  base18 ignition neuron with its reconstructed static parameters. It stays silent;
  the same returning inputs without prior history produce an ignition. This tests
  residual inhibition independently of the missing full-copy background stream.
- Feed only base34's recorded fast Z0 train into its fault neuron. Three **synthetic**
  150-q arrivals at 1,912,225/235/245 make it fire. Removing those arrivals, or using
  a 47-step rail with the same arrivals, leaves it silent. This demonstrates the lost
  single-rail margin; those stray times are not claimed as captured evidence.

A future fix regression must additionally build the actual Z/valid/reset primitives,
exercise the four near-simultaneous c3 ignition pulses and the staggered c5 ripple
pulses, sweep clear phase and independent loop/clear noise, and require **both Z
members dark before READY**, successful validity/completion after reload, and no
false fault. Test one fast rail plus background **and** two slow opposite rails;
silencing the fault detector is not a fix. Retain the true-guard, COPY, machine
three-pulse and kernel four-pulse suites. Full seed-108 verification still requires
the 100-copy CUDA run; a small RefSim corner cannot reproduce its device stray stream.

The required command was attempted:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_stall_diag.py tests/test_datapath_stall.py
```

It is **blocked by the sandbox's denial of uv's default cache** at
`~/.cache/uv/sdists-v9/.git`. The same tests **pass: 30 tests**, using the existing repository
virtualenv, `uv run --no-sync`, a writable task-local `UV_CACHE_DIR` and pytest
`--basetemp`, `PYTHONPATH` pointing at this worktree, and the required `TMPDIR`.
`git diff --check` also passes. The original diagnostic commit changed no netlist-building
file; the subsequent optional `zero_once` implementation is described above. Integration and committing remain with the
orchestrator; no commit is attempted.

## Compact register reset withdrawn after review (2026-10-04)

`robust_register_reset=True` now raises `ValueError`: full mix-B reload
qualification failed. The measured frontier and regression details are in
[the completion-stall report](stage_d_completion_stall.md#register-reset-qualification-and-opt-in-2026-10-04).
The experimental primitive retains four taps with zero-delay links and
**1.75 × loop** inhibition. Eight final 100-step READY links (+65.6 ms) repair
the biased stage-rail reload race; the earlier four-link policy (+32.8 ms)
does not. Neither setting is exposed as a qualified kernel option.

The stage **c3_xor.Q.b9r0** primitive uses copy 18's reconstructed seed-108 static
draws, the four captured ignition offsets and additional synthetic kicks/strays.
It has **0 survivors in 40,000 resets**. The actual capture's background stream is
not replayed; even the original controller need not fail at every reduced phase.
The old **20-step / 2.0-ms** isolated margin omitted tonic bias. At −0.2 mV
storage bias it becomes **−30 steps**, and the actual next stage word can fail
to complete. Tests now assert biased margins and advance actual stage loading
and master READY→COPY by 100 steps when evaluating the eight-link candidate.

Finalizing after all wiring includes `extend_reset` ALU state, FAULT, ACT,
COMMIT/grant/COPY, `zero_once`'s OR/relay/inhibitors, and RR mirrored readers.
Two-word, fault-discard/reload, whole-Q-domain silence, multi-cell arithmetic and
noisy RefSim checks exercise these effects. The combined three-sigma extended
domain still fails: base stops before its first cell reset, and RR never
completes its master. This also happens with the option off; changing reset
timing/strength cannot qualify that full envelope. `zero_once` still prevents
multiple Z0 ignition; it is independently selectable and is not required by this
experimental reset. Neither change establishes a fix for the RR34/RR43 captures or all
false-fault mechanisms. The control machine and default-off netlist fingerprints
are unchanged. The old policy's measured steady tick cost was +149–174 ms
(+2.7–3.1%), with +311 ms first-tick latency in the three-tick probe. Cell-stage
READY is unread and adds no critical-path delay; master COPY and input READY
consume the relevant delays. The eight-link candidate's tick cost is unmeasured.
