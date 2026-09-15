# A2/B: data RAM and the control machine

Follows `a2_alu_register.md`. This note records the data memory, the one-hot control
machine that executes a program held in neural memory, what the composition exposed, and
the campaigns. Everything below is Profile 3 (free synthesis), isolated execution, and
hybrid orchestration only in the sense that the host loads the program and data image and
lights the first PC line; after that the host reads spikes.

## 1. Data RAM (`lib/ram.py`)

A bank of W words × n bits. Every word is a master register (rails, bit-valid ORs,
completion tree W_w, its own reset controller and READY chain), the staged register's
master. Decoding is exact and has no threshold margin: a **word-select is a veto relay**
whose vetoes are the address rails that disagree with w, so any one mismatching rail blocks
it. Hierarchical predecoding, the plan's remedy for threshold decoders, would only reduce
fan-in here; the count is 3 neurons per word per port either way, and only allocated words
are instantiated (the address is a namespace).

| port | mechanism |
|---|---|
| write | trigger → WS_w (veto relay, vetoes: addr ≠ w, "op is not WRITE") → the word's edge-detected reset → its READY (80 ms) → COPY_w latch → per-rail veto relays (driver COPY_w, veto = the data source's other rail) → word rails → W_w → "written" pulse |
| read | driver → per-rail veto relays (vetoes: addr ≠ w, "op is not READ", the word's other rail) → destination rails. A word's "not this address" veto neuron is shared by all its relays |

Standalone block (`build_ram_block`, 8 × 4: 1,626 neurons): producer (addr, data, op) →
stage → ports; reads land in an output register → consumer. Clean model: write done 506 ms
after the load, read 417 ms, cycle ~600 ms. A read of a never-written word has no rail to
veto either relay, so both rails of every destination bit ignite: the read consumer's fault
gates refuse it and the refusal clears the stage (tested). An initial image is loaded by
igniting the words' rails; their completions then fire "written" pulses once, which the
runner lets pass.

## 2. Control machine (`lib/control.py`)

| part | form |
|---|---|
| IMEM | 8 words × 17 dual-rail bits, latch-only; the host loads the program image once |
| PC | one-hot ring of 8 lines; a newly lit line's **rise** fires a 3-pulse kill train on the others |
| FSM | one-hot ring FETCH → COMMIT → NEXT → FETCH, same kill-at-rise rule |
| IR | LOAD, STORE, JZ, JNZ, ADDR as **kill pairs**: each bit's two rail latches kill each other at their rise, so a reload needs no reset train |
| ACC | the accumulator (ALU + stage + master), grant as a veto relay (COMMIT is issued after W_S, so the grant is ordered and the last rate-mode AND of the register is gone) |
| DMEM | the RAM above: read port into the ALU's B rails (LOAD), write port from the master (STORE) |

Instruction word: `B[4] | U[5] | SUB | LOAD | STORE | JZ | JNZ | ADDR[3]`. MOV/ADD/SUB/AND/OR/
XOR immediate; LOAD [a] (acc ← DMEM[a], through PASSB); STORE [a] (DMEM[a] ← acc, the ALU
runs OR 0); JZ/JNZ t (OR 0, then PC ← t on the master's Z rail); HALT = JZ self and JNZ
self (a jump to the lit line makes no rise, so nothing re-lights FETCH). Every instruction
runs the ALU and commits the accumulator, so the committed master is the architectural
state after every instruction.

Cycle, times from FETCH's rise (clean model): +82 ms IR loaded; +156 P loaded (B vetoed by
IR.LOAD); +177 P.B from DMEM if LOAD, store-pending latch set if STORE; ~+560 stage complete
→ COMMIT lit → grant → master rewritten → W_M's pulse clears the stage, starts a STORE's
write, and lights NEXT (a STORE lights NEXT from the word's "written" pulse instead); NEXT
decides the jump from IR and Z, and 64 ms later the PC advances or jumps; the new line's
rise lights FETCH. 920–1,130 ms per instruction; 4,230 neurons for 8 program words and 8
data words.

`reference_run` executes the same ISA in Python; `run_machine` loads the image, lights PC 0
and decodes every committed word, PC rise and written word; `random_program` draws programs
the reference halts within 20 instructions with every LOAD reading a written word.

Later additions: memory operands for every ALU op (`ADDM [a]` etc.: the LOAD flag routes B
from the data RAM for any unit) and `JMP t` (JZ and JNZ both set), both encoder-only;
`CLR [a]` (a STORE with the copy vetoed, so the word is emptied; NEXT comes from the word's
READY); **indexed addressing** through an index word X (`LOADI`, `ADDI`…, `STOREI`: a second
read port and write port over the bank whose address taps are X's rails, ~50 neurons per
data word), which is what arrays need; and a **send timer** (§2.2).

### 2.1 Safe-point interrupts

`build_machine(handler_pc=k)` adds: an interrupt-pending kill pair INTP (the host ignites
rail 1; rail 0 = "none pending" is asserted at power-up, not left dark), a MASK pair, a link
ring LR of one line per program word, and two decision latches at NEXT. The safe point is
NEXT: the accumulator is committed and any store has completed. IT = relay(driver NEXT, vetoes
INTP.r0, MASK.r1, IR.IRET.r1) lights when an interrupt is pending; 64 ms later the lit PC
line is copied into LR (one relay per line vetoed by the other lines), INTP.r0 and MASK.r1
are set, and the handler's PC line is lit. The handler ends with `IRET`: IRT = relay(driver
NEXT, veto IR.IRET.r0) copies the lit LR line back into the PC and clears MASK. The ordinary
increment and jump relays are vetoed by IT and IRT. An interrupt arriving within ~55 ms of a
NEXT waits for the next one (the veto-residual rule), which is the intended bounded latency.

Measured: two interrupts during a countdown loop are taken after instructions 2 and 5, the
handler (save acc, increment a counter, restore acc, IRET) runs twice, returns to the
interrupted word each time, and the main program's result is unchanged: the final memory
equals the reference with interrupts at those indices, word for word (`tests/test_machine.py`).
Cost for 16 program words: ~250 neurons.

### 2.2 Send timer and status word

`build_machine(timer_hops=k, status_word=w)`: a delay chain started by the output port
word's completion (a send) and cancelled by the input port word's completion train (the
reply). On expiry it writes the constant 1 into the status word through a write port of its
own (word reset, READY, copy from a latched constant-1 level that the image asserts) and
raises the interrupt, so one handler serves both events and tells them apart by loading the
status word; it then stores 0 back and re-sends (a STORE to the port word re-completes it,
which re-exports its rails and restarts the timer). The status word is loaded with 0 at
power-up and must never be emptied: the exact-channel sender first cleared it with `CLR`,
and the next reply's handler then `LOAD`ed an empty word, which reads as both rails and halts
the machine as a fault (found on the Stage C clean-link test). A word completing without a STORE (the status
word, or the data image at power-up) must not light NEXT: the "store pending" state is a
kill pair now, and every "written" pulse into NEXT is vetoed by "no store pending".
Measured: with no link, two timeouts 1.6 s apart run the handler twice, the counter reads 2,
the port word is re-written once (`tests/test_machine.py`).

## 3. What the composition exposed

Five mechanisms, each found from the spike anatomy of one run and fixed on measurement.

1. **A lit line must not hold its successor down.** The first ring had every lit line
   inhibiting the others continuously; the FSM never left FETCH because FETCH itself held
   COMMIT down. Kills must be edge-triggered: the new line's rise fires a short train on the
   others and nothing inhibits a line while it waits to be lit. The IR bits use the same
   rule as pairs.
2. **A reset train paralyses a latch for ~80 ms.** Clearing the IR with a 3-pulse train at
   FETCH's rise and reloading it 20 ms later loaded nothing (every flag read 0, the data read
   port fired unvetoed and doubled P's B rails). That is the READY-chain rule from M1 seen
   from the other side; the kill pairs remove the train.
3. **A relay must be driven ≥ 55 ms after any of its veto rails dies.** The IR fetch relays
   are vetoed by "not this word" (the other PC lines) and the previous line dies ~8 ms after
   FETCH's rise; at +34 ms its veto's residual (0.5× loop, ~55 ms to decay) still blocked the
   relay, at +63 it did not. The same rule moved the P fetch to +156 (its LOAD veto may have
   just flipped) and the store-pending latch to +177. It is now a stated timing rule beside
   the ≥ 15 ms set-up rule.
4. **A done pulse counts only in COMMIT.** The data image's words complete at power-up and
   their "written" pulses lit NEXT, which advanced the PC under the first fetch. Both
   NEXT-lighting paths are vetoed by FETCH and NEXT, and PC 0 is lit before the image
   completes.
5. **A doublet into a trigger stacks reset trains.** Under mix B one node's word-select relay
   fired twice 3.6 ms apart (the driver train's second spike beat the relay's inhibitor,
   whose pulse lands only ~1.2 ms before that crossing), the word's reset trigger fired three
   times, three trains overlapped, and the copy could not hold. Relays that feed a trigger
   now drive their inhibitor at the head-start drive (`fast_inhibitor`), landing at 3.6 ms;
   relays that feed a latch keep the plain form, where a doublet is harmless.

### 3.1 Independent review (Kimi, 2026-09-14, on `2f0f818`) and what changed

| finding | fix |
|---|---|
| A stage fault caught **after** the grant found the master already reset: "discarded, M untouched" only held before the grant (same hole in the RAM write port) | The grant is guarded: the COMMIT token's pulse (`commit_in`) is delayed ~64 ms before it can grant, so a rail arriving up to ~25 ms after completion is refused with M untouched (the second review measured the window: the grant lands at W_S + 79 ms against a fault gate of 54.5 ms, not the ~60 ms first claimed; see §3.2); the copy relays are vetoed by the fault latch, so a fault caught later leaves M empty or partial, which M's completion and fault gates turn into a fail-stop, never a silently wrong master. Both blocks. |
| `classify_machine_run` skipped undecodable commits and could call a run with a faulted master "ok" | An undecodable committed word is `master_fault`, checked first |
| P-fetch veto margin after a LOAD flip was 62 ms against the 55 ms rule (7 ms slack) | P fetch at +166 ms, data read at +187 (two more hops each) |
| A fault after W_S strands the FSM in COMMIT, not FETCH as the note said | Note corrected (fail-stop in either state) |
| Watchdog re-arm after a sustained cancel unbudgeted (~+10 ms) | Recorded; fails loud if violated |
| Dead code and drift (`gates.swap`, `Memory.not_word`, stale docstrings) | Removed / corrected |

Audited clean by the review: the ordered-gate veto tables, the Z tree-node choice, reset
domains, power-up sequencing, encoders, simulator parity.

### 3.2 Second review (2026-09-14, on `26fa035`; Kimi stalled, the `claude-fable` fallback reviewed) and what changed

| finding | fix |
|---|---|
| The exact-channel sender `LOAD`s the timer's status word, which is empty until the first timeout; an empty read is both rails and the machine halts (reproduced: A faults at PC 17 with `acked` = 0) | The status word is a memory word from power-up (0), written to 1 by the timer through a write port of its own and stored back to 0 by the handler (§2.2); both scenarios of the exact channel now pass |
| The refusal window after completion is ~25 ms (ordered grant), not ~60: a late rail at 30–50 ms leaves M double-railed with NEXT lit; at 60–160 ms M empty with the FSM hung in COMMIT and no timeout | Claim corrected above. A **commit watchdog** (`commit_wd_hops`, ~800 ms from COMMIT's rise, vetoed by FETCH and NEXT) lights a TIMEOUT latch and clears the stage: the hang is a counted fail-stop. The 30–50 ms case remains a silent-error window for the perturbation campaign to size |
| Any hardware-completed word (a link arrival, the timer's status) during a STORE's pending window fires that word's "written" pulse into NEXT ~300 ms before the commit: the store is lost and the machine hangs | The written / cleared pulses into NEXT are vetoed by the mismatching address rails, so only the addressed word's completion counts |
| Timing audit: LINK.out re-fire gap 4–8 ms over the 86 ms recovery; the STORE port's copy relays after `CLR; STORE` to the same word have no margin; the P-fetch veto margin is 9.7 ms | Recorded as the margins the mix-B machine campaign must exercise (queued on Juno); not changed |
| A late ACK after a timeout: INTP.clear merges the two interrupts and the held input word cancels the resend's timer | Open: the protocol needs the receive credit (CLR of the port word) to re-arm the timer; noted in `docs/stage_c_flylink.md` |
| `lower.py` dropped RET, so an early `return` in a function fell through into the rest of the inlined body | An early RET is a JMP to a label placed after the inlined body |
| The golden shim's ports were `u8` while the IR uses the program width; `in_read()` re-reads one static word | The shim's port type follows the width. One input word per run is a v0 limit of the sequencer (the kernel path streams input) |
| §6 of this document still said there was no interrupt, port or IR interpreter | Corrected | The guard also changed how a
COMMIT is issued: `StagedRegister.commit_in` is fired once (by the FSM's token relay or by
the host), lighting the COMMIT latch and starting the guard; a chain fed by the latch's
train would have re-ignited the guarded latch continuously (measured: the grant fired
before completion).

## 4. Campaigns (mix B)

| block | transactions | ok | wrong values | refused / hung | non-ok observed / 95 % upper | wall |
|---|---|---|---|---|---|---|
| 8 × 4 data RAM, random writes and reads (every word written first) | 4,000 | 4,000 | 0 | 0 | 0 / 7.5 × 10⁻⁴ | 96 min, 3 CPU threads |
| control machine, random 8-word programs (probe) | 10 programs / 65 instructions | 65 | 0 | 0 | 0 / — | 39 min, 4 CPU threads |
| control machine, 100 random programs (713 instructions), fixed ALU | 100 programs | 98 | 0 | 2 (1 short, 1 no halt: fail-stop) | 2.0 % / 6.2 % (silent wrong ≤ 3.0 × 10⁻²) | 262 min on one H200 |

Before the doublet fix (§3.5) the RAM block's first 10-node probe had one node with 11
non-ok in 20 (a word that could not hold its copy); after it, 200/200 and then 4,000/4,000.

## 5. The compiler path: DrosoC → IR → interpreter → machine

`compiler/frontend_c.py` (pycparser) accepts the v0 DrosoC subset: one scalar width
(`u8`…), static globals, `void f(void)` functions without recursion, `if`/`else`, `while`,
`+ - & | ^` with nested temporaries, `== 0`/`!= 0`/`a == b`, `x = in_read()` and
`out_pixel(e)` as ports. `isa/ir.py` is the three-address IR and its interpreter (the
executable oracle, on `isa/semantics`). `compiler/lower.py` lowers to the accumulator
machine: `dst = a op b` → `LOAD a; OP b | OPM [b]; STORE dst`; a branch → `LOAD v; JZ/JNZ`
(a LOAD commits Z = (v == 0)); calls inlined (bounded, no recursion); ports memory-mapped;
HALT = JMP self. `compiler/golden.py` compiles the same source with clang and UBSan into a
harness that prints the canonical state (declared statics in declaration order) and the
emitted pixels.

Three-way comparison on a program that reads an input, sums in a loop through a call,
masks, branches, stores and emits a pixel (17 IR ops, 31 machine words, 6 data words):
C reference, IR interpreter and the lowered program's machine reference agree on the
canonical state and the pixel for four fresh inputs (`tests/test_compiler.py`). The neural
run of the same program on machine v1 (8-bit, 32 program words, 8 data words, 12,450
neurons) for input 2 executes 37 instructions and ends with the same canonical state and
the same pixel as the C reference, commit for commit against the machine reference
(`tests/test_compiler.py::test_neural_machine_runs_compiled_program`, ~40 s of neural time,
~50 minutes on the CPU reference simulator). With the safe-point interrupt of §2.1, this is
the Stage A2/B exit (plan): a DrosoC program that reads runtime input, stores neurally,
computes, branches, loops, calls and emits a pixel, matching the C reference and the IR
interpreter on canonical state for fresh inputs, and an interrupt that lands at a safe
point and resumes. What is not yet met from that list is under §6.

Arrays: `static u8 a[N]` is a range of data words; `a[i]` lowers to `X <- base + i` then
`LOADI`/`STOREI` through the index word (`__x`). The array-sum program (`tests/test_compiler.py`)
agrees across the C reference, the IR interpreter and the lowered machine reference (38 words).

`bench/capacity.py` produces the whole-program capacity report the plan requires before any
run above 4 nodes: for this program, 713 program-image bits, 59 live state bits, one ALU,
neurons by class (7,392 program image, 2,048 data memory, 1,449 ALU, 782 control, 779
registers and handshake), and a critical path of 55 instructions ≈ 58 s of neural time for
input 4. The program image is the cost driver: ~230 neurons per instruction word (two
latches per bit plus a fetch relay per bit), which is what a resident-kernel (spatial
dataflow) design avoids and what sets the direction for Stage B's resource sharing.

## 6. Costs, and what the machine is not yet

An instruction costs about one second, of which ~180 ms is fetch spacing imposed by veto
residuals, ~400 ms the ALU, ~300 ms the commit and ~70 ms the PC update. Four fifths of the
machine's neurons are memory (program words as latches, data words as masters with their
own completion and reset). The machine is fail-stop: a refused instruction (stage fault or
watchdog) clears the producer and stage and leaves the FSM in FETCH; recovery, retry and the
protected commit controller are Stage C/F work. A wider address is a namespace with
allocated words only. Calls are inlined by the lowering (no call stack); interrupts land at
safe points (§2.1); the ports are memory-mapped words exported and imported by FlyLink
(`docs/stage_c_flylink.md`); the IR interpreter is the compiler's oracle (§5). The sequencer
executes one input word per run and about one instruction per second: the hot loops belong
to resident kernels (`docs/a3_kernels.md`).
