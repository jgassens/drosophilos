# Datapath fault: seed 109, copy 73, specialized campaign

## Finding

The fault was not in `c9_sel`. The only captured stage fault-gate rise is
`c8_sub.Q.fault9.and` (neuron **5325**) at step **224,804**, in `c8_sub` transaction 4.
Bit 9 is the Z flag (`R[0:8], C[8], Z[9], V[10]`). All eleven `c8_sub.Q.valid*` latches
had risen and were live. No `c9_sel.Q.fault{i}` gate fired.

The later `c9_sel` stall is the downstream fail-stop symptom. The late c8 Z fault reset the
stage after a commit was already in flight. That delayed commit subsequently granted the
now-empty stage. At COPY, neither stage rail was present to veto either copy arm, so both
arms copied into every c8 master bit. All eleven `c8_sub.M.fault{i}` gates then fired and
continued firing. Transaction 4 of `c9_sel` selected this double-railed B input. Its mux is
made of veto relays: for each B bit, live rail 0 vetoed the rail-1 arm and live rail 1 vetoed
the rail-0 arm. Consequently neither result rail was lit for data bits 0--7. Only the
constant C=0 and V=0 outputs became valid; Z could not be formed without data completion.
There was therefore no c9 double rail, no c9 fault gate, and no c9 completion.

The initiating event is less specific than the downstream mechanism because the capture
omits the c8 stage rails and their Z drivers. The best-supported hypothesis is a stray
ignition of the other Z rail, `c8_sub.Q.b9r1` (Z=1, although 82 is nonzero), either directly
on its latch or through its sole driver `alu.z1.edge`. This is a hypothesis, not a captured
path: both roles were filtered out. The fault gate proves that both Z rails were live, but
cannot distinguish those two ignition paths.

## Exact rebuild and role mapping

`fault_diag` rebuilt the pipeline through `stall_diag.build_tick_pipeline` from every option
in `kc_tick_B90s109_n73_spec.json`: specialized requested datapath; true guards v2; ACT 11,
idle 20, watchdog 170 hops; power-up veto, commit reignition, delayed relight repair, and
four-pulse request/kernel clears enabled; kill strength 0.75. The result is exactly **25,113
neurons / 44,644 synapses**, matching the record.

The dump contains 22,157,010 spikes and 6,379 distinct captured neuron IDs. Its per-spike
role strings were checked in vectorized chunks against the rebuilt role at the same ID;
all mapped with zero ID drift. Running the new read-only tool gives:

```text
node 73: 1 stage fault-gate rise(s); 6379 captured neuron ids mapped
step 224804: c8_sub.Q.fault9.and (neuron 5325, bit 9), c8_sub transaction 4 START=219475 ACT^d=220049 completion=224012 commit=224770 DONE=228994; valid active=[0,1,2,3,4,5,6,7,8,9,10] rose=[0,1,2,3,4,5,6,7,8,9,10]; fault latch=224845
```

The campaign requested `datapath="specialized"`, but `build_pipeline` specializes only
AND, OR, XOR, MOV, and LOAD cells. SEL is deliberately excluded. Thus `c9_sel.op == "SEL"`
and `c9_sel.datapath == "generic"`; it uses the four veto-relay mux arms per result bit in
the generic SEL implementation.

## Evidence

### The originating stage fault

| Event | Neuron | Step |
|---|---:|---:|
| `c8_sub` transaction 4 START | 5766 | 219,475 |
| ACT^d (`c8_sub.actd.d10`) | 17694 | 220,049 |
| `c8_sub.Q` completion | 5388 | 224,012 |
| `c8_sub.commit_pulse` | 24265 | 224,770 |
| **`c8_sub.Q.fault9.and`** | **5325** | **224,804** |
| `c8_sub.commit_in` (delayed arrival from commit pulse) | 5640 | 224,814 |
| `c8_sub.faultL.u` | 5433 | 224,845 |
| `c8_sub.M.reset` | 5616 | 225,680 |
| `c8_sub.M.ready` (COPY is enabled from this) | 5637 | 226,497 |
| first c8 master fault (`M.fault10`) | 5555 | 227,120 |
| c8 master completion root re-rise | 5611 | 228,951 |
| `c8_sub.done.edge` | 5726 | 228,994 |

Every stage valid latch was already live when the bit-9 fault gate fired:

| Stage bit | Meaning | valid-latch neuron | transaction-4 rise |
|---:|---|---:|---:|
| 0 | R0 | 5257 | 221,675 |
| 1 | R1 | 5264 | 222,040 |
| 2 | R2 | 5271 | 221,929 |
| 3 | R3 | 5278 | 221,937 |
| 4 | R4 | 5285 | 222,283 |
| 5 | R5 | 5292 | 221,901 |
| 6 | R6 | 5299 | 222,287 |
| 7 | R7 | 5306 | 221,941 |
| 8 | C | 5313 | 221,964 |
| **9** | **Z** | **5320** | **222,107** |
| 10 | V | 5327 | 221,700 |

The intended c8 result is 82, so Z must be 0: `Q.b9r0` was the original valid rail. The
fault at 224,804 therefore means `Q.b9r1` also became live. This happened 792 steps after
stage completion and 34 steps after the commit pulse, not while the result was initially
forming. At the fault step the data valid latches were still firing, which argues against a
late data clear making the zero detector legitimately see an empty word.

### Empty-stage copy and corrupt master

The master fault gates establish that COPY put both rails into every master bit. Their first
captured spikes are:

| Master bit | Gate neuron | First step | Master bit | Gate neuron | First step |
|---:|---:|---:|---:|---:|---:|
| 0 | 5485 | 227,344 | 6 | 5527 | 227,307 |
| 1 | 5492 | 227,469 | 7 | 5534 | 227,188 |
| 2 | 5499 | 227,232 | 8 (C) | 5541 | 227,149 |
| 3 | 5506 | 227,282 | 9 (Z) | 5548 | 227,360 |
| 4 | 5513 | 227,322 | 10 (V) | 5555 | 227,120 |
| 5 | 5520 | 227,231 |  |  |  |

All eleven continued firing through step 899,999. This all-bit pattern is not propagation of
the one bad Z bit. It is the signature of `add_staged_commit`'s two absence-veto copy arms
both seeing a dark stage. The chronology supplies the race: `commit_in` fired at 224,814;
the fault latch did not rise until 224,845; the fault reset cleared the stage and commit
latches but could not retract the pulse already travelling down `guardd`; the later grant
reset the master at 225,680; and `M.ready` enabled COPY at 226,497, after the stage and fault
latch were clear. Since each `cp{i}r{r}` relay is vetoed by only the opposite stage rail and
the fault latch, both relays passed for each empty bit.

### `c9_sel` transaction 4

The source values at c9 ACT^d step 263,035 are partly inferred because the capture omitted
the source data rails. The inference is fixed by the prior captured outputs and tick
semantics: the previous `mx` is 84 and the current `px` is 27, so `c7_add` intends 84+2=86,
`c8_sub` intends 84-2=82, and `c6_and` holds `(84-27)&128 = 0`. Therefore the condition's
Z1 rail selects B (`c8_sub`). The capture directly establishes that c7 and c6 have no master
fault, while all eleven c8 master bits are double-railed; it does not directly expose the
86/82/0 data rails.

| Input/state at ACT^d | Best reconstruction | Capture support |
|---|---|---|
| A: `c7_add.M` | valid 86 | expected from captured prior state 84; no `c7_add.M.fault*` |
| B: `c8_sub.M` | intended 82, actually both rails on all 11 bits | every `c8_sub.M.fault0..10` firing |
| condition: `c6_and.M` | 0, hence Z1 (`c == 0`) selects B | `(84-27)&128`; no `c6_and.M.fault*` |

| c9 event | Neuron | Step / result |
|---|---:|---|
| transaction 4 START | 6321 | 262,456 |
| ACT^d (`c9_sel.actd.d10`) | 19781 | 263,035 |
| V valid (`Q.valid10.L.u`) | 5882 | 263,511 |
| C valid (`Q.valid8.L.u`) | 5868 | 263,524 |
| data valids 0--7 | 5812, 5819, 5826, 5833, 5840, 5847, 5854, 5861 | no transaction-4 rise |
| Z valid (`Q.valid9.L.u`) | 5875 | no transaction-4 rise |
| stage completion | 5943 | no transaction-4 rise |
| `Q.fault0..10` | 5817, 5824, 5831, 5838, 5845, 5852, 5859, 5866, 5873, 5880, 5887 | **none fired** |

## Missing decisive roles and replay filter

The downstream empty-stage-copy mechanism is supported by captured fault and reset timing.
Only the first extra Z-rail ignition remains unresolved. A second capture should include the
c8 stage rails and zero drivers, the grant/copy race, the relevant source-master rails, and
the c9 mux/stage rails. `--dump-roles` uses `re.search`; this anchored regex supplies those
roles without requesting the whole netlist:

```text
^(?:c8_sub\.(?:Q\.(?:b|reset|valid|fault)|faultL|commit(?:_in|_pulse|\.)|guardd|grant|copy|cp|M\.(?:b|reset|ready|comp|fault))|c9_sel\.(?:s[0-7]r[01]|Q\.(?:b|valid|fault)|start$|actd)|c7_add\.M\.b|c6_and\.M\.b9r|alu\.z)
```

`alu.z*` role names are repeated across generic cells; rebuilding the topology and following
the incoming edges to `c8_sub.Q.b9r0/1` disambiguates the captured neuron IDs. In this build,
the sole non-latch input to `c8_sub.Q.b9r1.u` (neuron 5250) is `alu.z1.edge` neuron 19530.

## Fix indication

A kernel fix is indicated in `drosophilos/lib/staged.py:add_staged_commit`, but none is made
here. A stage fault/reset must durably cancel a commit already travelling through `guardd`,
or COPY must require the selected stage rail to be live rather than treating absence of the
opposite rail as sufficient. Wiring the master's existing fault gates into a fail-stop latch
would add defense in depth, but would only detect this race after the invalid COPY. The Stage
D faults cannot be assigned this mechanism until a capture shows the same fault/commit/reset/
all-bit-master sequence; `fault_diag.py` is intended for that check.

## Recapture (Juno 425635, 2026-09-26): the trigger is a false fault, not a double rail

The targeted recapture reproduced copy 73 exactly (7 ok / 9 missing, 1 fault). The Z=1 rail
`c8_sub.Q.b9r1.u` (neuron 5250) and its driver `alu.z1.edge` never fired. Only the correct
Z=0 rail `c8_sub.Q.b9r0.u` (5248) was live, and it was running **fast: one spike every 34
steps** against the nominal 47 (224,539, 224,573, 224,607, ... 224,777). The fault gate
`c8_sub.Q.fault9.and` (5325) is a rate-mode AND with both rails at 211 quanta each (0.55 of
the train need at the *nominal* rate); one rail at 47/34 = 1.38x the rate supplies ~0.76 of
the need alone, and a stray-input spike covers the rest. It fired at 224,804, 27 steps after
the rail's last spike. So the chain is: a fast latch (the mix-B drift class of the kill-margin
item) trips a rate-mode fault gate on a correct word -> the stage resets under an in-flight
commit -> the empty stage is copied (fixed on 2026-09-26: `copy_requires_rail`, now a
fail-stop). The hypothesis above (a stray Z=1 ignition) is **refuted**.
