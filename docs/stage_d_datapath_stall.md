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
