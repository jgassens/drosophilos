# Perspective F16 seed 114 copy 57: operand-XOR veto escape at R10

**Copy 57 belongs to the operand-XOR double-rail class diagnosed in
[copy 85](stage_d_f16_stall_copy85.md).** In `c4_mulp.r3`, incorrect operand-XOR
relay **77605**, `fa10.x.y1.a.edge`, fires at **94,156** despite an established
zero-operand veto. Both operand-XOR rails establish and persist. They veto
both possible sum outputs, leaving **R10 alone without either stage rail**.
Stage completion correctly waits forever within this capture. This is not
the master-partner establishment failure of [copy 75](stage_d_f16_stall.md).

**The operand-XOR one-shot extension targets this failure mechanism and should
cover this freeze if it successfully accepts the first source pulse and rejects
later pulses.** That is mechanism-based expected coverage, not a demonstrated
fixed-run result. No modified circuit or counterfactual simulation was run.

## Evidence and method

Read-only inputs under
`/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:

- Record: `perspective_f16_cal_s114_fp32_3090_655fda2.json`.
- Dump: `perspective_f16_s114_copy57_dump.npz`.
- Report: `reports/perspective_f16_s114_copy57.md`.

The recorded run used **fp32 on an RTX 3090**, built at **655fda2**, with the
F16 flags. `build_tick_pipeline(record)` rebuilt **123,323 neurons and 249,940
synapses**. `load_dump(path, pl.net.roles)` loaded **94,012,506 spikes**, steps
**22–299,999**, with explicit full capture of all **123,323 IDs** and **zero
role-ID drift**. Both `datapath_view(pl, dump, 'c4_mulp.r3', 92241)` and
`completion_view(pl, dump, 'c4_mulp.r3', 92241)` reported no unavailable probes.
The rebuild, load, both views, and extraction of relevant spike arrays took
about **13 seconds**; subsequent inspections took about one second each.
No run approached ten minutes. Only this document is a repository change.

Unless stated otherwise, silence means no captured spikes during
**92,241–299,999**, inclusive. All role names below omit `c4_mulp.r3.`;
the repeated `out.b10*` names were resolved by wiring, not name alone.
One step is **0.1 ms**. Evidence is spike timing and wiring, not voltage.

## The failing arithmetic path

Transaction 1 START **16263** fires at **92,241** and ACT^d **75277** at
**92,805**. The preceding r2 master supplies exclusive operands
**acc=256, A=64, B=100** at ACT^d. Row 3 adds `(A * B3) << 3`; since B3=0,
the expected result remains 256 and **R10=0**. The bit-10 operand XOR compares
delayed acc10=0 with partial product **A7 AND B3=0** (`pp7`, shifted to bit 10).

| Signal | Neuron IDs | Observed steps |
|---|---:|---|
| `pp7r0.u/v` | 77335 / 77336 | First 93,889 / 93,939; sustained to 299,986 / 299,954 |
| `pp7r1.u/v` | 77333 / 77334 | Silent |
| Delayed acc10=0, `s10dr0.d22` | 76989 | First 94,077; next 94,119, 94,157, 94,193 |
| Correct XOR ignition, `fa10.x.y0.b.edge` | 77614 | **94,118**, exactly once |
| Incorrect XOR ignition, `fa10.x.y1.a.edge` | **77605** | **94,156**, exactly once |
| Correct XOR rail, `fa10.x.y0.u/v` | 77603 / 77604 | First **94,158 / 94,218**; last 299,977 / 299,952 |
| Incorrect XOR rail, `fa10.x.y1.u/v` | 77601 / 77602 | First **94,197 / 94,254**; last 299,948 / 299,980 |
| Delayed carry-in 0, `fa10.cinr0.d4` | 78221 | First 94,419; sustained |
| Delayed carry-in 1, `fa10.cinr1.d4` | 78226 | Silent |
| Sum transaction edge, `fa10.s.once0.edge` | 78235 | **94,462**, exactly once |
| Sum spent latch, `fa10.s.once0.spent.u/v` | 78233 / 78234 | First 94,500 / 94,553; sustained |
| Sum-1 candidate, `fa10.s.y1.a.edge` | 78247 | Silent; x0-driven veto **78249** starts 94,205 |
| Sum-0 candidate, `fa10.s.y0.b.edge` | 78256 | Silent; erroneous-x1-driven veto **78258** starts 94,253 |
| Sum rail 1 / rail 0, u/v | 78243/78244 / 78245/78246 | All silent |
| Final stage drivers, `out.b10r0.edge`, `out.b10r1.edge` | 78571 / 78573 | Both silent |
| Q bit 10 rail 0 / rail 1, u/v | 13643/13644 / 13645/13646 | All silent |
| `Q.valid10.or`, `Q.valid10.L.u/v` | 13897, 13898/13899 | All silent |
| `Q.valid10.ign.edge/edge_inh/hold_inh` | 13902/13903/13904 | All silent |

The sole sum-source pulse arrives at both candidates at **94,480**, after
both veto trains are established. The other sum candidates **78250/78253**
depend on absent carry-in-1 one-shot **78240** and are silent too. The sum
one-shot works: one emitted pulse, both spent-latch members established.
Inspection of all 16 r3 operand and sum XORs finds only **fa10.x** with both
output rails firing; no sum XOR is double-railed.

## Escape timing: two excitations before effective suppression

Receiver **77605** has exactly these three wired inputs; weights below are
**nominal**, not reconstructed copy-57 draws. Every edge has an 18-step delay.

| Source | Nominal q | Emission → arrival steps |
|---|---:|---|
| Delayed acc10=0 **76989** | +4655 | 94,077→**94,095**; 94,119→**94,137**; 94,157→94,175 |
| Source inhibitor **77606** | −7966 | 94,138→**94,156**; 94,185→94,203 |
| Partial-product-0 veto **77607** | −1810 | 93,943→**93,961**; 94,010→94,028; 94,076→**94,094**; 94,141→94,159 |

The first veto arrives **134 steps before** the first excitation; another
veto arrives one step before that excitation. Thus this is not absent or late
EARLY data. No wrong-branch spike follows the first excitation alone in the
observed trace. A second excitation arrives **42 steps** later. The receiver
fires **19 steps after that second arrival**, at **94,156**, which is also the
first strong source-inhibition arrival step. Its output reaches wrong-rail u
**77601** at **94,174**, and both wrong-rail members subsequently establish.

The established diagnosis is a vetoed operand-XOR relay escaping and igniting
the wrong rail. **Hypothesis:** the closely spaced second excitatory input
accumulates enough drive to cross threshold before strong inhibition can
prevent firing, analogous to copy 85. Here the observed escape and first
strong inhibition share a numbered step, unlike copy 85's three-step lead.
Spike timestamps alone do not resolve the membrane/current dynamics within
that step, or prove that removing the second pulse suffices. No specific
perturbed weight, stray event, or fp32 rounding effect is asserted as the cause.

No RefSim reproduction was attempted. The static draws could be regenerated
with the same seed procedure, but the original integration was **fp32**;
an **fp64 RefSim** experiment would not necessarily reproduce the exact escape
step or threshold crossing. A local counterfactual would also not constitute
a replay of the campaign's batch-dependent background events.

## Completion and exclusion of a latch-establishment failure

Exactly **50/51 Q validity latches** rise. All other 50 stage bits establish
one selected rail with both members; **R10 is the sole bit with neither rail**.
Six Q completion joins, precisely the ancestors of valid10, have both members
silent:

| Join | u/v IDs |
|---|---:|
| `Q.comp.c0_5.L` (valid10 AND valid11) | 14307 / 14308 |
| `Q.comp.c1_2.L` | 14483 / 14484 |
| `Q.comp.c2_1.L` | 14579 / 14580 |
| `Q.comp.c3_0.L` | 14619 / 14620 |
| `Q.comp.c4_0.L` | 14643 / 14644 |
| `Q.comp.c5_0.L` (root) | 14659 / 14660 |

The root's other branch **14651/14652** establishes at **96,139/96,193** and
persists. Commit **122347**, master reset **15771/15772**, COPY **15813/15814**,
DONE **16223**, Q reset **14666/14667**, and FAULT **14706/14707** are all
silent, including before START within the capture. There is no preceding r3
transaction/reset to explain this as reset recovery. The internal double rail
does not propagate to Q, so the stage fault detector cannot report it.

In particular, the stage latch is never driven: **78571/78573 never fire**.
Both partners of each operand-XOR rail establish; both partners of the sum
one-shot spent latch establish. No missing latch partner initiates this chain.
A COPY/master ignition fix, or stronger stage-latch ignition, therefore has
no direct coverage. The missing sum output is already upstream of those writes.

## Repair coverage and limits

The proposed operand-XOR one-shot extension addresses the same unprotected
`fa*.x` source-burst hazard as copy 85. The existing sum one-shot cannot repair
a double-railed EARLY operand: both sum branches are vetoed even with a perfect
single carry pulse. Restricting the operand-XOR source to its first pulse is
therefore the relevant candidate repair for copy 57 as well.

Expected coverage remains a **hypothesis pending qualification** of this fp32
case with the new circuit. Qualification must verify correct first-pulse
acceptance, wrong-branch suppression, and repeated reset/reload behavior;
this analysis does not establish that the extension fixes every freeze.

Validation: captured-ID availability and role alignment passed; direct
assertions confirmed R10 as the only missing stage bit, established pairs on
the other 50 bits, and both members silent in all six listed completion joins.
`test -f docs/stage_d_f16_stall_copy57.md` and `git diff --check` passed.
