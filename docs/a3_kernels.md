# Resident kernels: a loop body as a spatial dataflow pipeline

`lib/kernel.py` (the pipeline), `compiler/kernel.py` (IR loop body → pipeline spec),
`bench/run_kernel.py`, `tests/test_kernel.py`. Plan: "the default execution model should be
spatial dataflow"; `docs/capacity_doom.md` §3–4 gives the reason (the sequencer is four
orders of magnitude short of a frame budget).

## 1. What a kernel is

A chain of cells. Every cell owns a master register (its output value, a level) and an ALU
datapath or a RAM read port; a cell computes one operation on the levels of its sources (the
masters of earlier cells, the input register, or constants the image lights once) and commits
the result to its own master. Tokens flow down the chain; there is no fetch, no decode and no
program counter. The kernel *is* the program.

```
request (upstream done) -> REQ  ─┐
                                  ├─ start pulse -> ACT, ACT^d -> operand gates sample the sources
idle (done + 106 ms)   -> IDLE  ─┘                  -> ALU / read port -> stage -> commit request
                                                     -> commit (when the consumer is free) -> master -> done
```

The input register is a staged register whose producer the host loads (transduced input), one
token after each READY. The output cell's master is read by the host at each completion: a
pixel record.

## 2. The handshake between cells (measured, in the order the failures appeared)

The first build had none: a cell was triggered by the upstream done pulse, and the host paced
the input. Four failures, each measured on the renderer's column loop (four cells, 8 bits):

1. **A cell re-triggered while busy drops the token.** The input register cycles in ~680 ms and
   a cell takes ~880 ms, so tokens piled into the first cell; its ACT latch was already lit, so
   the operand gates never sampled again. (Host rate-limiting to 1 s hid this for five tokens.)
2. **Constant unit-select rails do not re-drive the flags.** In the sequencer's ALU the C and V
   "zero" for non-adder units come from the unit-select rail's rise, a token per instruction.
   In a kernel the unit select is a level lit once by the image, so after the first commit
   cleared the flag latches nothing re-lit them and the second AND never completed. The zeros
   are now driven by ACT^d, a rise per run.
3. **A chain fed by a latch train has a tail.** ACT^d was a delay chain from the ACT latch;
   relays driven by a train need ~86 ms of source silence to fire again, and the chain kept
   firing ~85 ms after ACT was cleared, so a cell restarted less than ~170 ms after its done
   pulse sampled nothing. ACT^d is now a chain of pulses from the start pulse.
4. **Latencies vary with the data** (830–990 ms for the same cell), so no fixed host rate is
   safe: a 990 ms run followed by an 873 ms upstream run re-triggered the second cell 11 ms
   after its done pulse, inside its reset train's paralysis; a fault, and the token was lost.

So the cells carry the handshake themselves, from three kill pairs per cell (a two-rail bit
whose rails kill each other at their rise, `control.add_kill_pair`):

| pair | true means | set by | cleared by |
|---|---|---|---|
| REQ | a new source value is waiting | the upstream done pulse | the start pulse |
| IDLE | the cell may start | done delayed 20 hops (~106 ms: the reset train's paralysis is over) | the start pulse |
| CREQ | the stage holds a result to commit | the stage's completion | the commit pulse |

A cell starts when REQ and IDLE are both true, and commits when CREQ is true and its consumer's
REQ is false (the consumer has started on the previous value, so its operand gates have
sampled it; the output cell's consumer is the host and is always free). "Both true" is the
`guarded_pulse` element: two veto relays, one driven by A's rise delayed 12 hops and vetoed by
"B false", the other by B's rise delayed 20 hops and vetoed by "A false". A veto rail that died
less than ~55 ms before its relay is driven still blocks it, so the two delays differ by ~45 ms
and the windows overlap: whichever rail flipped second, one relay sees its veto long dead. Both
may fire within the overlap; a doublet into a latch or a pulse chain is one event.

Two rules found on the way, now stated in the code:

- **A pair flipped twice within ~86 ms does not flip back.** The pair's own kill is a relay
  driven by the other rail's train; REQ's "false" rail was killed by the request and re-lit by
  the start ~60 ms later, inside that relay's recovery, and the request rail survived. The
  start and commit pulses kill the rails they clear with their own kill trains.
- **Poll a live simulator from its per-step spike lists.** The trace property sorts the whole
  event array on every access; the first runner read it every 200 steps and a 40 s run took
  fifty minutes in the sort (70 s without). `protocol.token.decode_recent` and `recent_spikes`
  now serve every runner, the machine's included: Hello World's 31.8 s of neural time takes
  57 s of wall time on one core, not 1,878.

## 3. Measured (clean model, Profile 3, 2026-09-15)

The renderer's column loop (`examples/render.c`, `d = map[(col + heading) & 7]; h = htab[d];
out_pixel(h)`) compiles to four cells — ADD (input + heading), AND 7, LOAD map, LOAD htab —
with the array bases folded away and `heading` an image-time parameter.

| quantity | value |
|---|---|
| neurons | 10,820 (8-bit cells; the two 8-word memories are ROMs: the contents are wired into the read relays, ~25 neurons per word against ~250 for a RAM master; 13,727 with masters) |
| pipeline latency, load to first pixel | 4.54 s |
| throughput, self-paced | one pixel per 1.18 s (8 columns, 8 correct) |
| the same loop on the sequencer | ~32 instructions per column ≈ 32 s |
| gain | ~27× per kernel at about the machine's neuron count |
| simulation | 25 s wall for 13 s of neural time on one CPU core |

The kernel reference (`compiler/kernel.kernel_reference`) equals the IR interpreter's pixels for
three headings, and the neural pipeline equals the kernel reference for eight columns
(`tests/test_kernel.py`, the slow test ~30 s).

## 4. State kernels: the world update as dataflow (2026-09-15)

`examples/tick.c` is the Stage D shape: `p_move` (the player moves by the input velocity and
is clamped by two walls) and `m_move` (the monster steps toward the player; contact costs
health), three ticks, two pixel records per tick. Its loop body has calls, four `if`s and
three loop-carried variables. The compiler now handles all three:

- **Calls** are inlined (the closing RET dropped; an early RET is rejected).
- **`if` / `if-else`** become select cells: both arms are computed and a SEL cell picks per
  bit from the arms' levels by the condition cell's Z flag (four veto relays per bit, no
  ALU). A JZ over the then-arm means "the arm applies when the condition is not zero"; the
  compiler orders the arms accordingly. Nested ifs recurse; a variable defined on one arm
  only is rejected.
- **State** (read before written, and written): the variable's reads before its first write
  refer to the cell that writes it last, a feedback edge to a cell built later. That cell
  carries `init` (the prologue's constant) and the image lights its master and the readers'
  requests for it, so the first token finds the state there. A cell whose every cell source
  is a feedback edge is paced by the input token (`trigger: ["input"]`): the tick.
- **Several outputs**: every OUT names an output cell; the host decodes each master.

Two changes to the builder came with it. A cell is requested by **every** cell source's done
pulse (one REQ pair per source, joined by chained guards), not by its first operand's alone:
with one request a cell could sample a second operand mid-rewrite. And a producer's commit
waits for **every** reader (chained guards through "passed" pairs). The renderer's kernel and
the fan-out test run unchanged on the new builder.

| tick kernel | value |
|---|---|
| cells | 13 (5 SEL, 5 ALU, 3 state carriers among them), 28,420 neurons |
| ticks | 3, both outputs correct on every tick (px 25/30/35, mx 88/86/84) |
| latency per tick | 5.98 s (the state loop is ~9 cells deep and cannot overlap ticks) |
| the same on the sequencer | 133 instructions / 3 ticks ≈ 45 s per tick |
| gain | ~7.5× (a dependent loop pipelines nothing; the gain is the sequencer's overhead) |
| simulation | 75 s wall for 22 s of neural time |

With fresh input every tick (`examples/tick2.c`: `vel = in_read()` at the top of the loop, so
the velocity is the token and px, mx, health the state) the same kernel runs eight ticks with
velocities that wrap the player into the west wall and clamp it at the east one: all sixteen
outputs equal the IR interpreter's, 5.92 s per tick, no fault or timeout.

## 5. Wider cells (2026-09-15)

| cell | neurons (with the input register) | note |
|---|---|---|
| 32-bit ADD cell | 10,754 | a three-cell 32-bit kernel (ADD, XOR, SUB): 28,507 neurons, four tokens correct, 1.99 s per token, 4.8 s latency (the 32-stage carry ripple) |
| 16-bit MUL cell | 27,877 | built, not yet run |

The input register's producer watchdog now scales with the width (60 + 4n hops): the stage's
completion comes 148 ms after the load at 8 bits and 308 ms at 32 (a deeper completion tree
and more rail events), and the fixed 55 hops (~290 ms) timed the 32-bit register out before
its stage completed; the timeout cleared the stage, the commit then copied an empty stage as
double rails, and the operand gates of the first cell were vetoed on every bit. Two more
rules from it: the runner decodes the R bits only (a master carries C, Z and V above them:
bit 32 read as 2³² until masked), and every kernel run reports its fault-latch and timeout
counts, which the pipeline's `stats` now carry.

### 5.1 Shifts, ROM tables and a perspective column (E1 shape)

A shift by a constant is wiring: `SHL k` / `SHR k` cells route the source's rails k bits over
and drive zeros into the vacated bits (four relays per bit, no ALU); DrosoC gained `<<` and
`>>` by constants (the sequencer lowers a left shift to doublings and has no right shift).
Memories a kernel reads are ROM relays (§3). With those, `examples/render2.c` is the E1 shape
at 16 bits: the wall distance from the map, its reciprocal from a Q8 table, the projected
height as `(recip * scale) >> 8` — a perspective divide done as a table and a multiply.

| perspective kernel (16-bit) | value |
|---|---|
| cells | ADD, AND, LOAD map, LOAD recip, MUL, SHR 8 |
| neurons | 47,775 (the 16-bit multiplier cell is ~28k of them) |
| eight columns | all heights correct against the C reference and the IR interpreter |
| per column | 6.6 s — the 16 × 16 array multiplier's latency (n rows of n-bit ripple adders, ~n² × 29 ms) |
| latency to the first pixel | 12.1 s |
| simulation | 383 s wall for 60 s of neural time |

The array multiplier is the bottleneck of the E1 path by a factor of ~6 over every other
cell, and its latency grows as n². The dataflow answer is the **pipelined multiplier** (`MULP`):
the n rows of the array become n cells, each carrying the running sum and both operands in
its word (acc | flags | A | B) so the next row reads them from its master, each with the
standard handshake. Latency is n cell latencies; throughput is one product per cell latency,
whatever n. Measured at 8 bits: six products correct, 9.9 s latency (8 rows), 1.58 s per
token, 24,428 neurons (the array: ~2.3 s per token at 8 bits); at 16 bits the cell is
90,364 neurons against the array's 27,877 — 3.2× the area for 4× the throughput. The
perspective column with it: eight columns correct at **1.74 s per column** (6.6 s with the
array), 26.7 s to the first pixel, 110,262 neurons. The compiler picks it with
`compile_kernel(..., mul="pipelined")`; the throughput-bound kernels (frames) want it, the
latency-bound world update keeps the array. A carry-save multiplier
would cut the latency too; it is not built.

## 7. Two loops, two streams: the game loop and the render loop together

`compile_program` takes a loop nest of depth two — a frame loop around a column loop, the
shape of `examples/render.c` — and emits one pipeline with two token streams: the outer
body without the inner loop is the tick kernel (stream `input:frame`), the inner loop the
column kernel (stream `input`). What the columns read and the tick updates (the heading) is a
**parameter edge**: the column cells read the tick's state master as a level, with no request
and without holding the tick's commit. That is the one place the dataflow handshake is not
closed inside the substrate: a parameter's producer must not commit while a reader is in
flight, and the host keeps that order. The tick kernel's state carriers are outputs, so the
host can see the state land.

The host schedule is a list of tokens, each with its stream and the number of outputs that
must be out before it goes in: a frame's eight columns, then the tick once the eight pixels
are out, then the next frame's columns once the tick's outputs are out. This is the plan's
hybrid control plane (Python pacing, neural computation), and the benchmark label says so;
Stage F2 replaces it with a neural readiness signal from the columns' request pairs.

Measured (8 bits, two frames of eight columns): every pixel and both frame records equal the
interpreter's, with the heading advanced by the tick between the frames; 16,601 neurons (the
two kernels and two input registers), 14.6 s per frame (eight columns at ~1.2 s and a tick at
~2 s, plus the host's pacing gaps), no fault, no timeout (`tests/test_kernel.py`).

## 9. A frame in the substrate: one pixel per token, one kernel per brain

`examples/frame.c` is the first picture-producing program: a 160 × 100 Doom-like view. The
token packs a pixel's column and row; the column's view angle (the heading plus the column,
20 angle steps across the screen) indexes a 64-entry table of wall distances compiled from
the level's walls by the asset compiler (a room with a doorway and a pillar, ray-cast at
compile time); a reciprocal table gives the projected wall height; the pixel's colour is
ceiling, wall shaded by distance, or floor, by comparing the row with the wall's top and
bottom (the sign bit of a subtraction, then a select). Sixteen cells, 64,948 neurons at 16
bits — most of it the four 16-bit ALU cells and the two selects.

A frame is 16,000 tokens. On one kernel that is 16,000 × 1.2 s ≈ 5.3 hours of neural time;
the cluster's shape is one kernel per brain and the host dealing tokens, which the batched
simulator gives directly (one node per copy: `run_pipeline_batched`, two-node test). On 128
copies a frame is ~150 s of neural time; on 1,000 brains ~20 s. `bench/render_frame.py`
deals a frame (or a subsample of it) to B nodes and writes the neural picture beside the
reference (`display/frame.py`, the host's only addition being the palette).

**Measured (Juno, one H200, 128 copies):** the full 160 × 100 frame, 16,000 of 16,000 pixels
equal to the reference, 411 s of neural time, 7,992 s of wall time (`docs/img/frame160_neural.png`
beside `frame160_reference.png`). The three-frame 80 × 50 slideshow with the tick turning the
view: 12,000 pixels, all correct, 344 s of neural time, 7,381 s of wall time
(`docs/img/game80_{0,1,2}_neural.png`). `examples/game.c` puts the frame loop inside a tick loop whose token is the
turn (`heading = (heading + turn) & 63`): `compile_program` makes it a tick kernel and a
pixel kernel, `bench/render_game.py` deals each frame's pixels to the B copies and then the
same tick token to every copy (the game state is replicated in every brain, which is what
the plan's TMR will vote over), and writes one picture per frame: a slideshow computed in
the substrate, the host dealing tokens and adding a palette. The neural 40 × 25 picture on 32 nodes (Apple GPU) and the full frame on
128 nodes (Juno H200) are running; their results go here.

## 10. Independent review of the kernel path (2026-09-15, `claude-fable` worker; Kimi timed out twice)

Audited clean: the cell handshake (guarded pulses, kill trains, commit gating, reset domains,
the runner) and the control-machine changes (the timer's write-port marks, the
address-qualified NEXT relays, the commit watchdog, the ALU's dual veto), on the timing rules
and by re-running the neural tests. Defects found and fixed:

| finding | fix |
|---|---|
| `semantics.py` redefined `shl`/`shr` (logical) over the spec's arithmetic ones: `test_semantics` failed, `fromfix(-65536)` returned 65535 | Duplicates removed; the IR's SHR is the spec's `shru`, SHL its `shl`; one `ir.apply` for the interpreter and the kernel compiler |
| A state's initial value was the *last* CONST anywhere in `main`, a loop-body assignment included (silent wrong state: [5, 10, 15] for [25, 30, 35]) | Initial values come from the prologue before the first loop only |
| A loop inside a kernel body made the compiler spin forever | A backward branch is rejected with `NotAKernel` |
| The reference masked a ROM address with `n_words − 1`: a wrong oracle for tables whose length is not a power of two | Out-of-range addresses raise (the relays would match nothing and the cell would stall) |
| `out_pixel` of a constant or a parameter made a MOV cell with no request, which free-ran | Such a cell is paced by the token |
| A state variable used directly as an `if` condition faulted on the first token: the image lit the value's rails but not its Z rails, and both SEL arms fired | The image lights a state master's C, Z and V rails too |
| The lowering's return label used a stale inline id when the callee itself called a function | The call's own id is kept for its label |
| Induction-variable updates were dropped, so a read after the update disagreed with the interpreter | The update is a cell like any other; dead cells are pruned |
| `decode_recent`'s status was ignored: a faulted or partial output word read as a value | A non-valid word is `None` in the outputs and counted in `bad_outputs` |
| The commit watchdog's TIMEOUT latch had no reset (a false timeout was permanent) and the campaign classifier did not count it | It clears with the stage; the classifier counts it |

Recorded, not changed: a doublet on a multi-pair guard (gap up to ~43 ms, measured 39) against
the passed pair's kill-relay recovery (~50 ms) is a plausible spurious start under
perturbation, not seen on the clean model — the mix-B campaign on a two-source cell must size
it. The 32-bit kernel has a test now.

### 9.1 The Doom-shaped program (`examples/doom1.c`)

Three kernels from one program: a tick kernel (19 cells: the input's turn and forward/back
bits move the player, the level checked first as Doom's `p_map` does), a column pass
(48 cells: a four-step ray walk through a 16 × 16 grid in Q4.4 with trig tables, the
projected height and the distance stored into two RAM buffers), and a pixel pass (13 cells:
the buffers read by column, ceiling / shaded wall / floor by row). 443,330 neurons per copy
at 16 bits. Validated neurally on one copy (Juno, CPU): two frames of a 2 × 3 sample, all
pixels equal to the reference, 142 s of neural time. Each copy renders the pixels of its own
columns, so a copy's buffers only ever hold what it wrote (`bench/render_doom.py`). At 40 × 25
on Juno's H200, 32 copies (14.7 M neurons), host pacing: two frames with the player moved
between them, 2,000 / 2,000 pixels correct, no fault or timeout, 310 s of neural time in
2 h 31 min of wall time (`docs/img/doom40_0_neural.png`, `docs/a2/doom1_40_h200.json`).

### 10.1 Third round (2026-09-15, on `a26495f`; Kimi stalled again, the `claude-fable` worker reviewed)

Audited clean: the pipelined multiplier's timing, the STORE and LOAD ports' timing, the ROM
factoring, the runners' trimming and decoding, the drivers' barriers, and `doom1.c`'s
semantics (the kernel oracle equals the interpreter on 32,000 pixels). Nine defects, mostly
programs the compiler accepted and compiled to something other than their meaning:

| finding | fix |
|---|---|
| An inner loop writing a variable the outer loop carries compiled to two copies of the state | Rejected |
| A body reading its loop counter beside `in_read()` read the token instead | Rejected (the counter's own update may read it) |
| A load after a store of the same array in one body has no ordering edge | Rejected (read it in a later pass) |
| Two STORE cells on one array: both ports' COPY latches light at the word's READY and copy over each other | Rejected at first; since `7ea4cd4` the ports carry marks (the timer recipe of `control.py`): a port's select lights a mark for the word that vetoes the other ports' copies until the word completes, and the completion clears every port's COPY. Two stores to one word within one write's window (the select to the completion plus ~50 ms, ~340 ms after ACT^d at 4 bits; the first draft said 150 ms, the review measured it) double-rail it: a fail-stop at the next LOAD. Under neural pacing the phase counter waits for every STORE of the pass, not only the last output. Measured: two STORE cells writing disjoint words on every token, eight words read back, no fault (`test_kernel_ram_written_by_two_store_cells_in_one_pass`) |
| An `if` on a second stream's token crashed the builder | The condition gets its MOV cell for any stream |
| A parameter first read inside one arm was refused as "defined on one arm only" | The other arm sees the parameter's constant |
| ROM and RAM addresses with bits above the table's width aliased silently while the oracle raises (`doom1` reached it after seven back-steps) | The high address bits veto every word: an address beyond the table selects nothing, a fail-stop; and the tick now checks the level before moving (Doom's `p_map`), so the player never leaves the grid |
| The batched runner reported no faults, timeouts or bad outputs | It counts them per node |
| An outer loop that is only a counter was refused | Allowed: its stream only paces |

### 10.3 Multi-pair guard replay (2026-09-16)

The warning from the first review was real. In the mix-B fan-out campaign (8-bit, eight
copies, seed 0 on Juno's H200), node 6 produced
`[3, 11, 11, 19, 19, 203, 203, 83]` instead of
`[3, 11, 11, 19, 19, 203, 83, 11]`: the sixth value completed at 10,348 ms and again at
11,517 ms. The same signature appeared in the 16-bit perspective kernel and in 8 of 100
fan-out copies. Mix 0 and the ROM renderer (one reader per value) did not show it.

This was a **second run of the output cell**, not a second input token or two commits of one
stage. A two-source cell has three start conditions (`REQ.a`, `REQ.b`, `IDLE`), so
`_chain_true` first caches `REQ.a && REQ.b` in a `passed` kill pair, then guards that pair
against `IDLE`. The two-order guard can legitimately emit a doublet. In an unperturbed
one-bit fan-out used to retain the whole handshake record, c2/c3 DONE both fired at
2,413.6 ms, both c4 REQ-true rails at 2,422.0 ms, and `c4.go.p0.pulse` fired at 2,496.6 and
2,536.0 ms (39.4 ms apart). The passed-true rail rose at 2,500.8 ms; c4 started once at
2,572.8 ms, reset both REQs and IDLE at 2,577.0 ms and reset the passed pair to false at
2,578.7 ms. Its first commit, master completion and DONE were at 3,015.3, 3,338.6 and
3,342.8 ms.

That 39.4 ms doublet is inside the passed pair's kill-relay recovery margin. Under mix B the
cached true rail can survive or re-arm around the start reset. A deterministic reproduction
injects that stale passed-true spike after the first DONE: it rose at 3,365.3 ms, IDLE rose at
3,453.0 ms, and the old circuit started c4 again at 3,567.4 ms although there was no new
c2/c3 DONE and both c4 REQ-false rails were live. The second run committed at 4,009.9 ms and
completed the same value at 4,332.1 ms. Thus the failure is the passed pair flipping back;
the commit pulse's kill train and the input register are not the source of the duplicate.

The fix is an end-to-end check on the final guard: when its left input is a cached passed
pair, both final veto relays also see every original source's false rail. Those rails are
already dead well before a legitimate final driver, so the normal timing is unchanged. If a
passed rail is stale when IDLE later rises, any reset REQ blocks the start. The same rule
protects chained commit guards from a stale cache. The fast two-copy regression injects the
passed-true spike and re-arms IDLE: before the fix its start-cluster steps were `[2534]` and
`[2534, 5703]`; after the fix they are `[2534]` and `[2534]`. In the full two-copy one-bit
fan-out, both fixed copies likewise started only at 2,572.9 ms and completed once at
3,338.7 ms; the injected copy produced no replay through the 4,500 ms ceiling.

The local Apple-GPU seed rerun could not be measured in the isolated worker: although
`system_profiler` reports the M1 Pro and Metal support, this PyTorch process reports zero MPS
devices and rejects allocation as an unsupported OS. The seed-0/1/2/3 eight-copy rerun and
the 100-copy cluster campaigns therefore remain external confirmation runs.

**Rates before the fix, with the ceiling removed** (Juno 407291; 100 copies × 8 tokens, mix B,
90 s of neural time allowed; every copy's outputs kept):

| block | ok | wrong | missing | faults | copies with errors |
|---|---|---|---|---|---|
| render (ROM tables, one reader per value) | 799 / 800 | 1 | 0 | 0 | 1 |
| fan-out (one value read by two cells) | 775 / 800 | 16 | 9 | 0 | 8 |
| perspective (16-bit, pipelined multiplier) | 760 / 800 | 7 | 33 | 0 | 7 |
| tick (13-cell state kernel, two state outputs) | 666 / 1,600 | 296 | 638 | 18 | 94 |

Every wrong value found in the records is one of two shapes. (1) The **duplicate** above: render's
single wrong value is its seventh output emitted again in the eighth slot; the tick records hold 56
such signatures. (2) A **stall**: 33 tick copies produced one output and nothing more, 33 two, 19
three — a third of the copies stop at the second token. And in a *state* kernel a duplicate is not a
shifted stream but a corrupted state: the duplicated run applies the update twice, so every later
output is wrong and nothing detects it (copy 0: `25, 25, 30, 30, 19, 33, 30, 59` for `25, 30, 24,
27, 27, 67, 107, 0`). Whether the guard fix above removes the stalls as well as the duplicates is
the question the fixed campaign answers.

(The same pre-fix campaign in single precision on G2's RTX 3090s, 30 s ceiling — fan-out 769 /
25 wrong, tick 516 / 208, render 799, perspective 276 — matches the H200's double-precision
numbers within noise: the GeForce cards' fp32 is fine for campaigns.)

**After the fix** (Juno 407450, the same 100 copies, seeds and 90 s): fan-out **795 / 800, 0
wrong** (one copy stalled at its fourth token); tick **1,598 / 1,600, 2 wrong, 0 missing, one
copy** — the stalls were the same stale-guard replay, now gone; render 799 / 800 (the same single
case); perspective **unchanged to the digit**, 760 / 7 / 33 in the same seven copies, and the
eight-copy seed-0 rerun still duplicates the fifth value. So the multi-pair guard was the fan-out
and state kernels' whole problem, and the pipelined multiplier's duplicate — and the ROM reader's
single case — come from somewhere else (§10.4 when found).

### 10.4 A REQ rail is not permission until its pair has settled (2026-09-16)

The remaining perspective and ROM-reader records have one cause at two different places in
the pipeline: a per-source REQ kill pair can be observed while both rails are live. The old
`guarded_pulse` checked the *other* pair's false rail but not the false rail belonging to its
driver. Thus the same ambiguous REQ could mean "pending" to the reader's start guard and
"free" to the producer's commit guard.

That gives both recorded shapes:

- If REQ-true survives or reappears as the reader becomes idle, the reader runs once more on
  the source's still-current master. In a MULP row this injects an extra old 3n+3-bit row word;
  it advances through the remaining rows and adds a duplicate product and output (shape A).
- If REQ-false survives or reappears while REQ-true is pending, it opens the producer's commit
  guard early. A final LOAD can consequently start before its address source has established
  the new master and read the preceding address (shape B). A run that stops after the expected
  output count makes the extra old result look as though it replaced the last token; with a
  longer ceiling the displaced request can appear as a later extra result.

The timing explains why this is a perturbation-only failure. A general kill pair used three
kill pulses, about 5.3 ms apart. The losing rail can still emit for about 15 ms, reset
inhibition paralyses a latch for about 80 ms, and a relay needs about 86 ms of source silence
to re-arm. A mix-B event is 150 quanta, `150 * 0.0171875 = 2.578 mV` of conductance jump: it
does not ignite a resting rail across the 7 mV gap, but it can change the last spike of a rail
which is already being killed or re-lit. Four-percent log-normal noise is not a ±4 % bound;
the long tail over the perspective copy's roughly 110k neurons supplies occasional weak kill
edges and strong loop edges. The one-reader ROM pipeline has four opportunities per token;
the pipelined perspective kernel has those plus sixteen row handshakes, matching the large
difference in observed incidence.

The alternatives in the completion path were tested and do not make these stream shapes:

| candidate | result and relevant margin |
|---|---|
| Master's completion train counted twice | No. Its nominal period is 4.7 ms and the runner requires a gap greater than three periods (~14 ms) to count another rise. Excitatory stray input does not extinguish the completion latch. |
| Master's DONE relay re-fires | No. The fast inhibitor lands at about 3.6 ms, before the next 4.7 ms completion spike, and a live completion train continues to hold the relay down. It needs roughly 86 ms of source silence and a new rise; the next such rise is the next master rewrite. A close doublet is absorbed by the REQ latch and by the runner's three-period grouping. |
| Stage-to-master COPY runs twice | No. `M.ready` lights one COPY latch; the COPY train holds every per-rail edge relay after its first firing. A close READY or COPY doublet only re-ignites the same state. A second copy needs a second grant/reset transaction, but CREQ is cleared by the commit pulse. |
| A false completion-tree AND | A 150-q sweep of a one-input completion AND did not fire in the reference model, including coherent +4 % excitation and −0.2 mV threshold/+0.2 mV bias. If forced, its signature is an incomplete/faulted master or a stall, not a complete replay of the preceding word. |
| 10.3 `passed` cache in a one-source cell | Not present. One cell source gives the direct `REQ && IDLE` guard; a MULP row has exactly this form. Intermediate `passed` pairs begin only with three guard inputs and retain the end-to-end false-rail recheck from §10.3. |

The normal prev/row ordering itself has ample margin. A row samples at ACT^d, 11 hops or
about 58 ms after start. Once its REQ-false acknowledgment is valid, the predecessor's commit
takes the long guard path (now 22 hops, about 117 ms) and the staged register's 12-hop guard
(about 64 ms) before master reset: over 120 ms after the sample. An erroneously restored
REQ-false while the row is still busy bypassed that ordering by opening the commit guard
*before the next start*; no increase to ACT^d could repair it.

The fix is local to that permission boundary:

1. Both final veto relays of every two-pair guard now also see both driver pairs' false rails.
   A pair must be one-hot before either true rail has meaning. The earlier §10.3 source
   rechecks remain in addition to these local checks.
2. Per-source REQ pairs use four kill pulses, as register reset does, and the explicit start
   clear uses four too. This closes the three-pulse transition in which the losing rail could
   be restored.
3. The two guard paths move from 12/20 to 14/22 hops. Their eight-hop (~42 ms) order overlap
   is unchanged, while the short path now arrives after the fourth kill pulse and the ~55 ms
   veto-recovery tail, with about 10 ms of clean-model margin.

Two deterministic CPU-reference regressions retain the failure and the fix in one run each.

**Outcome (Juno 408086, the same 100-copy mix-B campaigns): the remedy was reverted.** With
the one-hot guards, the four-pulse arbitration and the 14/22-hop paths, copies *stall*:
fan-out 737 / 800 (63 missing in 11 copies, against 795 / 800 after §10.3 alone), render
725 / 800 (75 missing, 12 copies, against 799), tick 1,171 / 1,600 (429 missing, 40 copies,
against 1,598), and perspective 470 / 800 with its 7 duplicates still there. The margins pass
every unperturbed slow test and fail under noise — the wider veto sets and longer paths are a
liability at 4 % weight noise, not a margin. `lib/kernel.py` is back at the §10.3 state; the
analysis above stands as a hypothesis, its two reproductions stay as expected failures, and
the campaign runner's `--dump-node` / `--dump-roles` / `--dump-out` spike dump (added in the
same round) is how the perspective copy's handshake will be captured on the cluster for the
next attempt.
They use two copies and remove exactly the new self-false veto synapse on copy 1 to reconstruct
the old guard. A timed injection holds the losing REQ rail across evaluation — the deterministic
postcondition of the rare mix-B transition, rather than a random search. The fixed copy rejects
it. The 4-bit MULP circuit has 7,156 neurons and runs 8 s of neural time: copy 0 starts the
chosen row once and outputs `[9]`; the old-guard copy starts it twice and outputs `[9, 9]` from
one input token. The two-LOAD ROM chain has 3,586 neurons and runs 5 s: the fixed copy outputs
`[9]`, while the old-guard copy reads the old address again and outputs `[9, 9]`. Both are below
30k neurons and use `RefSim` on CPU (`tests/test_kernel.py`). The cluster campaign is still the
statistical confirmation; the campaign runner can now retain the relevant evidence with
`--dump-node N --dump-roles REGEX --dump-out trace.npz`.

### 10.5 The perspective duplicate is an early multiplier-row replay (2026-09-16)

The spike dump for perspective copy 5, seed 0, mix B changes the localization from §10.4.
The repeated word is already in row 10's stage at the start of the retained window; row 13
does not create it.  Decoding the carried `A` and `B` fields (master bits 19--34 and 35--50)
at each completion gives this diagonal through the array:

| event | step | decoded row word |
|---|---:|---|
| `c4_mulp.r10.Q.comp.c5_0.L.u` is live at the window boundary | 255,025 (first retained spike) | `A=51, B=100, acc=5100` |
| `c4_mulp.r10.commit_pulse` | 255,090 | the already-complete stale stage is granted |
| `c4_mulp.r10.M.comp.c5_0.L.u`; `r10.done.edge` | 260,498; 260,541 | second `51 * 100` publication |
| `c4_mulp.r9.M.comp.c5_0.L.u`; `r9.done.edge` | 260,837; 260,880 | the next source word is `A=43, B=100`, after r10's stale stage was complete |
| `c4_mulp.r11.Q.comp.c5_0.L.u`; `r11.M.comp.c5_0.L.u` | 255,046; 258,912 | the first `51 * 100` wave already advancing |
| `c4_mulp.r11.start`; `ACT^d`; master completion | 261,345; 261,922; 274,161 | the legitimate r10 DONE advances the second `51 * 100` wave |
| `c4_mulp.r12` master completions | 272,307; 287,541; 301,545 | `51 * 100`, `51 * 100`, then `43 * 100` |
| `c4_mulp.r13` starts / `ACT^d` | 273,130 / 273,690; 288,367 / 288,931; 303,063 / 303,626 | samples those three r12 words in order |
| `c4_mulp.r13` master completions | 285,547; 300,790; 314,888 | `51 * 100`, `51 * 100`, then `43 * 100` |
| `c4_mulp.r14` master completions | 299,115; 314,395; 328,644 | `51 * 100`, `51 * 100`, then `43 * 100` |
| `c4_mulp` (row 15) master completions | 312,340; 327,689; 341,234 | `51 * 100`, `51 * 100`, then `43 * 100` |
| `c5_shr.start`; master completion | 313,179 / 322,101; 328,522 / 337,441; 342,075 / 351,012 | `19`, repeated `19`, then displaced `16` |

The detailed row-13 handshake is ordered.  For the repeated wave its source DONE is 287,584;
`REQ` true rises at 287,665 while false dies at 287,766, IDLE has been true since 286,680
(its false rail dies at 286,824), `go.g0.pa.edge` fires at 288,324, START at 288,367, and
`ACT^d` at 288,931.  START re-lights REQ-false at 288,413 and the true rail's last spike is
288,470.  The stage completion, commit pulse, master completion and DONE are 294,727,
295,515, 300,790 and 300,830.  Rows 14 and 15, and `c5_shr`, show the same valid
DONE -> REQ -> START ordering.  Thus neither row 13 nor the shift cell starts without a
source publication; they faithfully propagate the duplicate already emitted by row 10.

At the shift cell the fourth result's row-15 master completes at 312,340 and DONE fires at
312,381.  `c5_shr` REQ-true rises at 312,465, its false rail dies at 312,588, the guard's
`pa.edge` fires at 313,139, START at 313,179, and `ACT^d` at 313,757.  Stage completion,
commit, master completion and DONE are 316,694, 317,493, 322,101 and 322,144.  The duplicate
has the same sequence: row-15 completion/DONE 327,689/327,729, REQ-true 327,811, guard/START
328,483/328,522, `ACT^d` 329,102, stage/commit 331,988/332,785, and master completion/DONE
337,441/337,484.  Finally the displaced `43 * 100` word completes row 15 at 341,234, raises
REQ at 341,358 and starts `c5_shr` at 342,075; its completion is the `16` at 351,012 outside
the all-cell window.  No c5 request, IDLE or commit event is out of order.

The out-of-order neuron was therefore `c4_mulp.r10.start`: it ran on row 9's still-current
`A=51, B=100` before row 9 published the fifth word (`A=43`) at 260,837.  Its exact START,
REQ and IDLE transition is not in this artifact: the NPZ begins at 255,000 with r10's stale
stage completion already live.  The older full-run input-only dump does establish the host
side: values 3 and 4 (the fourth and fifth tokens) complete in `IN.M.comp.c3_0.L.u` at
42,471 and 55,109, and `IN.done.edge` fires at 42,513 and 55,151 (values 4 and 5, if tokens
are named by their zero-based values, complete at 55,109 and 67,620).  It contains no cell
roles.  `c1_and`/`c2_load`/`c3_load` are already at rest when the all-cell window opens.
The available dumps can therefore prove the stale-operand run and its complete propagation,
but cannot prove which pre-window spike kept or re-lit r10's REQ.

| perturbation of the false rail (loop, kill, V_th, bias) | re-light fails up to | works from |
|---|---|---|
| none | Δ = 300 | 350 |
| 2 σ: loop ×0.92, kill ×1.08, +0.2 mV, −0.2 mV | 600 | 650 |
| 3 σ: ×0.88, ×1.12, +0.4, −0.4 | 700 | 800 |
| 4 σ: ×0.85, ×1.15, +0.6, −0.6 | the latch no longer sustains | — |

The real Δ is 740–770 steps: the 3 σ corner of one latch's six parameters, before the stray
stream's ±0.4 mV. A 16-row multiplier has 32 such rails.

What a dark `REQ` false rail does is the whole shape. The start has killed `REQ` true, so
both rails are dark. (1) The producer's commit guard sees no `REQ` true and commits its next
word through its `pa` path as soon as its stage completes — the reader sampled at `ACT^d`,
58 ms after the start, long before that rewrite. (2) The reader's IDLE rises 106 ms after its
done, its `pb` relay is vetoed by nothing, and the row **starts with no request**, sampling
whatever the producer's rails hold: with tokens 15.8k steps apart and a row cycle of 14.1k
that is the *next* word, already in place. (3) That word's real request arrives 1–2k steps
after this start, is not reached by the start's kill train, and pends (`pa` vetoed by IDLE
false). (4) At the next IDLE the pending request starts the row on the producer's master,
which still holds the same word — the producer's commit for the word after is blocked on
the pending `REQ` true — and the row runs it a second time. This start re-lights `REQ` false
(the kill was long ago), the producer commits, and the pipeline resumes: the sequence is n,
n+1, n+1, n+2 — copy 5's `64, 51, 51, 43` — with nothing lost and no fault. The same dark
rail gives the campaign's other perspective shape when the phase differs: if the producer's
rewrite is in flight at the request-less sample (rails dark for ~85 ms; token intervals of
~19–20k) the operand gates take nothing and the row stalls; beyond that the old word is
sampled and n itself is duplicated.

**Reproduction** (`test_dark_request_rail_replays_the_next_word_and_actd_relights_it`,
4-bit `MULP`, 7,108 neurons, two `RefSim` copies, tokens `[3, 5, 2]` × 3, 1.4 s host gap,
8.8 s of neural time, 18 s wall). Copy 1 carries only the 3 σ corner above on
`m.r2.req.m.r1r0` (two loop synapses, two kill synapses, `V_th` and bias of its two members).

| copy 1 (`m.r2`) | steps |
|---|---|
| trigger, `pa`, start (token 1) | 29,861, 30,581, 30,623 |
| `REQ` false: one spike, no train | 30,676 |
| done, IDLE, `pb`, start **with no request**; `ACT^d` samples token 2 | 40,281, 41,383, 42,485, 42,527; 43,110 |
| token 2's trigger; `REQ` true rises and pends | 44,154; 44,196 … 54,932 |
| done, IDLE, `pb`, start: token 2 **again** | 52,662, 53,764, 54,866, 54,908 |
| token 3's trigger (released by that start's `REQ` false rise), `pb`, start | 60,435, 67,246, 67,288 |
| outputs | `[9, 15, 15, 6]` against copy 0's `[9, 15, 6]` |

**Fix: the row's own `ACT^d` pulse re-lights every rail the start pulse re-lit.** One synapse
per rail (`act_d → idle[0].u` and `act_d → req[0].u` per request), no new neurons, no change
to any delay, veto set or kill train. `ACT^d` is the start pulse 11 hops later, so the second
ignition lands ~111 ms after the kill train — twice the 3 σ boundary — and when the first
ignition worked it is a re-ignition of a lit rail, which its consumers (veto interneurons
and the producer's 20-hop guard chain) do not see. In the guard-doublet case `ACT^d` is a
train (eleven spikes at r10, 215,292–215,739): a lit rail fed eleven ignite pulses at 213 Hz
runs at a 31-step period for those 45 ms and is back at 47 afterwards, alive, its own kill
relay silent (measured on an isolated kill pair, loop ×1.0 and ×0.88). Timing when the first
ignition failed:
`REQ` false rises at start + ~60 ms instead of + 4 ms, the producer's `pb` commit follows
20 hops later and its master reset ~64 ms after that, all after the reader's sample at the
same `ACT^d`. On the reproduction's 3 σ copy with the fix, the start's ignition still gives
the one spike at 30,676, `ACT^d` at 31,206 re-lights the rail (train from 31,249), every
start follows its trigger by the guard's 12 hops (30,623, 44,916, 58,866 against triggers
29,861, 44,154, 58,104 — the clean copy's steps to the step) and the outputs are `[9, 15, 6]`.
The test keeps both: copy 0 is the 3 σ rail with the fix, copy 1 the same rail with the
`ACT^d` synapse zeroed (`[9, 15, 15]`, four starts, the second before its trigger). The fast
kernel tests pass; the 100-copy mix-B campaigns are the confirmation.

**Against §10.4.** Refuted in its specifics: no both-live pair, no restored losing rail and no
one-hot condition is involved — the failing transition is a *dark* pair, which a one-hot
guard does not see either, and the wider veto sets and 14/22-hop paths only lengthened the
races that stalled copies. Refined in its location: the fault is at a per-source `REQ` pair
and the replay goes through the `pb` relay, as §10.4 guessed for shape A. The commit pair
has the same primitive (the commit pulse re-lights `CREQ` false ~50 ms after `CREQ` true's
rise killed it); its failure would open a request-less commit at the reader's next start,
which the reference showed as a stall, not a duplicate, and is left for the 100-copy rerun
to weigh.

**Outcome (Juno 408540, the same 100-copy mix-B campaigns, 90 s): the remedy was reverted.**
The re-light does what it was built for — perspective **0 wrong** (against 7 duplicates in
every earlier campaign) — and costs more than it saves everywhere else:

| block | §10.3 alone (407450) | with the `ACT^d` re-light (408540) |
|---|---|---|
| fan-out | 795 / 800, 0 wrong | 768 / 800, 0 wrong, 32 missing in 7 copies |
| render | 799 / 800, 1 wrong | 773 / 800, 0 wrong, 27 missing in 5 copies |
| tick | 1,598 / 1,600, 2 wrong | 1,471 / 1,600, **5 wrong**, 124 missing in 17 copies |
| perspective | 760 / 800, 7 wrong, 33 missing | 714 / 800, **0 wrong**, 86 missing in 14 copies |

The stalls are the §10.4 shape (a copy stops after its first, third or fifth token and never
resumes), and tick copy 77 shows a shape the earlier campaigns never had: `25, 30, 24, 24,
64, 104, 0` and `88, 86, 84, 82, 80, 82, 80` — one update applied twice and the state
corrupted from there, i.e. a replay *created* by re-igniting a rail that had been killed on
purpose. One synapse per rail with no new margin analysis was the same wager as §10.4: it
passes every unperturbed test and loses under 4 % weight noise. `lib/kernel.py` is back at
the §10.3 state, the reproduction above is an expected failure, and the dark-rail mechanism
stands as the localization. What the next attempt needs is a re-ignition that cannot land
on a rail the kill train is meant to keep dark — gated by the request's own true rail, or a
longer-recovery kill pair — and the 100-copy campaign before it is believed.

### 9.2 Textures and a sprite (`examples/doom2.c`, `examples/doom3.c`)

`doom2.c` textures the walls: the column pass also stores the hit position's texture column
(`ubuf[col] = ((hx >> 1) + (hy >> 1)) & 7`), and the pixel pass reads the texture row by a
reciprocal table and a multiply (`v = ((prow - top) * recip2[h]) >> 8`) from an 8 × 8 brick
texture with a bright and a dark bank (beyond two cells). Measured neurally on Juno's H200
under **neural pacing** (8 copies × 686,351 neurons, 40 × 25, two frames with the player moved
between them): 2,000 / 2,000 pixels equal to the reference, no fault or timeout, 1,982 s of
neural time in 7 h 53 min of wall time (`docs/img/doom2_40_0_neural.png`, `_1_`;
`docs/a2/doom2_40_h200.json`). The neural time is ~7 s per token, against ~3 s for `doom1.c`
under host pacing: the pacing counter's cost, §11. `doom3.c` adds one *thing* in Doom's
sense, a billboard sprite at a fixed level position: the tick kernel projects it into camera
space once per frame with the trig tables (depth and side offset, then the screen column,
half-width and height by the reciprocal table), a fourth kernel — a sprite pass over the 160
columns — compares the sprite's depth with the column's wall distance (`dbuf`, stored by the
column pass, read here in a later pass as the compiler requires) and stores the sprite height
and texture column per column (`sbuf`, `stex`), and the pixel pass overlays a non-transparent
sprite texel over the wall or floor. The thing's projection for the first frame is set in the
prologue as state. Four kernels: tick 60 cells, column 67, sprite 18, pixel 52; 1,308,973
neurons per copy at 16 bits (the tick's multiplies are most of it). Validated by the kernel
oracle against the interpreter and the C golden build on two 160 × 100 frames
(`tests/test_doom3.py`, `docs/img/doom3_0_reference.png`: the sprite half hidden behind the
near wall; `_1_`: the view turned); not yet run neurally (an H200 job at 40 × 25 is next).
`doom4.c` makes the thing *move*: its position is frame state of the tick kernel, and each
tick it takes one 8-unit step toward the player along the larger axis, refused into a wall
cell or the player's cell (Doom's `P_Move` shape); 96 tick cells. Three frames, the imp
approaching and growing (`docs/img/doom4_0_reference.png` … `_2_`). The first draft of the
program reused a temporary between the sprite's row clamp and the overlay gate, so the imp
was never drawn although the interpreter, the C golden build and the kernel oracle agreed
on every pixel — three oracles of one program agree on its bugs; the test now also asserts
that the imp's colours appear inside the sprite band and grow.

## 11. Neural pacing: the phase order moves into the substrate (Stage F2, first step)

Until now the host kept the frame's phase order — a frame's columns, then its pixels, then
the tick — by holding tokens back until earlier outputs were out (§7, the hybrid control
plane). Now the order is the substrate's. Each phase has an OK pair (a kill pair: "this phase
is open"); a stream's input register commits a token only while its phase is open (the pair
joins the register's commit chain like a reader's "free" rail). A phase ends with a pulse
that closes its pair and opens the next one: for a pass of K tokens per frame, the compiler
adds a one-hot ring counter (§11.1) whose wrap, every K tokens, is the end pulse; the tick
phase ends at the tick kernel's state landing (its state carrier's done — measured: ending
it at the tick *token's* landing let the next frame's columns read the old heading). The
first phase is open at power-up.

The host then only deals tokens in program order, as fast as each register's READY allows;
tokens for a closed phase wait in their register's stage. Measured on the two-pass renderer
(three kernels, 36,880 neurons with the two ALU counters of the first step; 24,721 with the
rings): two frames, sixteen pixels and two frame records equal to the interpreter's, no
fault, with a barrier-free schedule
(`tests/test_kernel.py::test_neural_pacing_replaces_the_host_barriers`). What stays with
the host: dealing tokens to copies, the per-copy token counts (image constants), and the
tick's *input* (the game's controls, which are input by definition). The benchmark label
for the frame loop's ordering moves from *hybrid* to *neural*; the dealing stays hybrid.
Both drivers take `--pacing neural` (`bench/render_doom.py` deals columns,
`bench/render_game.py` pixels); the per-copy counts are image constants, so the columns or
pixels must divide evenly across the copies. The phase counter counts every STORE of the
pass as well as its last output (a STORE deeper in the dataflow than the last output would
otherwise land after its phase had ended; the review of §10.1's round found it).
Measured on the Doom-shaped program itself (`doom1.c`, 93 cells with the two ALU counters,
482,596 neurons, one CPU copy on Juno): two frames of a 4 × 3 sample under neural pacing,
every pixel equal to the reference, no fault, 279 s of neural time (job 405122).

### 11.1 The counter is a ring, not a datapath (2026-09-16)

The first step's counter was three cells, `xor` (cnt == K−1), `add` (cnt + 1) and `cnt` (a
select with state), and it cost the pass its speed: a pass ran at ~7 s of neural time per
token under neural pacing (doom2 on the H200: 1,932 s for 260 tokens per copy; doom1 at 4 × 3:
279 s for 26 tokens) against ~1.5–3 s per token host-paced, for the same kernels. Two
reasons, both structural. The count was a state cell fed back through the other two, so each
token's count was a three-handshake round trip (~5 s); and the counter was a *reader* of the
pass's output cell and STOREs, so commit gating (§2: a producer rewrites its master only once
every reader has started on the previous value) made those cells wait for the counter to
start before each rewrite. Every token of the pass waited for the counter's loop, and the tick
kernel was gated the same way.

Now the count is a one-hot ring (`lib/kernel.add_pacing_ring`): K lines, each a kill pair
[dark, lit], line 0 lit at power-up like the machine's PC ring. One advance pulse — the
trigger cell's done relay — drives K veto relays at once, `inc k` vetoed by line k's dark
rail, so only the lit line's relay passes: it ignites line k+1's lit rail and line k's dark
rail (each pair's own kill train clears the other rail), and `inc K−1` also fires the wrap.
The ring has no datapath and holds no commit: its relays listen to the done rails the way a
request pair does, but no cell's commit guard waits on them, so the pass pipelines at its own
speed. A line's dark rail is a level that stands for the whole of a cell's cycle (≥ 300 ms)
before the next advance, which is the veto relay's ordering assumption (§2), and a relay
vetoed at one advance is driven again ≥ 300 ms later, past its 55 ms recovery.

Every trigger cell gets a ring of its own. The triggers (the pass's last output and every
STORE) each fire once per token but can run several tokens apart in a deep dataflow, and a
ring shared through one advance neuron would lose two dones that fell within one hop; with one
ring per trigger no two advances of a ring are closer than the cell's own cycle. The phase's
end is the join of the rings' wraps: each wrap sets a kill pair [not wrapped, wrapped], the
chained guard of §2 pulses when all are set, and that pulse clears them (measured: three rings
wrapping 100 ms apart gave three guard pulses over 18 ms — the two paths of the guard and a
passed pair — so the end passes through an edge relay and is one pulse). A ring cannot lap
another before the join fires: the phase's gate closes at the wrap, and the host deals the
next frame's tokens of this stream only after the other phases' tokens, which wait one per
register stage. The compiler emits the ring as a `RING` pseudo-cell (`{pfx}ring`: K, the
trigger cells, the stream); it is nobody's source, the reference skips it, and K stays an
image constant (the ring has K lines). Cost: ~17 neurons and ~30 synapses per line (a kill
pair, a veto relay, its kill train), 4,400 neurons for doom2's 260-token pixel ring against
the 12,000 the two ALU counters cost the two-pass renderer.

Measured on the tiny pass of `test_pacing_ring_wraps_every_k_tokens` (two cells, K = 3, a
one-cell tick, 5,546 neurons, RefSim): the ring advances 4 ms after every done of its trigger
and wraps 13 ms after the third, sixth and ninth; the wrap opens the tick's phase within 4 ms,
the tick's done reopens the pass's; the register commits every 750–1,000 ms inside a frame,
which is the two-cell pass's own pace (the ALU counter's phase end came ~5 s after the K-th
done). With both cells as triggers (the join) the wrap comes 146 ms after the later ring's.
The two-pass renderer compiles to nine cells and two rings (`ph0_ring`, K = 4, on the column
STORE; `ph1_ring`, K = 8, on the pixel output); 24,721 neurons where the counters made it
36,880. Measured (Juno 408459, CPU): the two-pass renderer's slow test, two frames under
neural pacing, went from 42.6 s to **29.4 s** of neural time — within 4 % of the same program
host-paced (30.6 s), so the pacing now costs nothing there; `doom1.c` at 4 × 3 (one copy, two
frames, 26 tokens) from 279 s to **255 s**, every pixel right — a smaller gain, because at
that size the Doom passes are bound by their own cells' latency, not the counter. The 40 × 25
textured run on the H200 (1,982 s with the ALU counter) is queued again on the ring for the
figure at scale.

One caveat the tiny test exposed, and it is the host's, not the ring's: with a program of
one inner loop and the tick (two phases), the host deals the next frame's first pass token
right after the tick token, while the pass's phase is still open draining its last token —
nothing holds that token back, since the tick's register takes its one token into its stage
and lights READY for the pass again. The token commits before the tick has landed and reads
the old parameter. The three-phase renderers cannot do this (the next phase's register holds
its first token and its READY stays dark until that phase opens), and the ALU counter had the
same exposure with a longer drain; a one-inner-loop program under neural pacing (`game.c`,
`bench/render_game.py --pacing neural`) needs either a host barrier on the frame's first
token or a third phase. The fast test deals each frame behind the previous tick's output.

### 11.2 One token in flight: the pipelined multiplier is the wrong multiplier (2026-09-16)

Three H200 runs of `doom4.c` (the chasing thing, 245 cells) measured the same thing twice.
With the array multiplier (Juno 405517: 40 × 25, 8 copies × 1.47 M neurons, ALU counter) a
pixel cost ~30 s of neural time; with the pipelined multiplier (406972: 32 × 20, 16 copies ×
2.10 M, neural pacing; 408444: the same under **host** pacing) a pixel cost **63 s** in both
runs, output for output, and neither could finish inside its limit (36 h; 16 pixels an hour
against 2,880 outputs). Pacing made no difference because under both the pass's tokens go
one at a time: host pacing deals the next token only after the previous output, and the
ring's phase gate lets one token per stream through. A pixel's cost is therefore the pixel
kernel's *latency*, and the pipelined multiplier's latency is its sixteen row handshakes
(~20 s at 16 bits, §4) against the array's 6.6 s — its 1.7 s throughput never sees two tokens.
`doom4.c`'s pixel pass has two products (the wall's texture row and the sprite's), so ~40 s
of its 63 s is multiplier latency. `bench/render_doom.py` now defaults to `--mul array`; the
40 × 25 ring rerun of `doom2.c` (408524, pipelined) was stopped at ~10 s per output, above the
ALU counter's 7 s, and resubmitted with the array (408916) so that the ring is compared on
the same multiplier; `doom4.c` runs at 24 × 15 × 2 frames on the array (408917). The
pipelined multiplier earns its area only where the compiler lets several tokens of one
stream overlap, which the phase order of §11 forbids by design: the next step for
throughput is a pass whose tokens pipeline *within* a phase, with the ring counting dones,
not starts.

## 8. What it is not yet

- Loops inside a body (a while inside the tick) are not kernels yet; a nested loop is a kernel
  of its own with a stream, and joining two kernels is the next compiler step.
- Memories are read-only (ROM relays); a kernel that writes memory (a framebuffer, a
  z-buffer) needs RAM masters with the write port's handshake, which the sequencer has and
  the kernels do not yet use.
- Kernel parameters are image-time constants; a per-frame parameter (the heading) needs a
  parameter port the host rewrites between frames, which is a staged register like the input.
- The cells are 8-bit; the Q16.16 datapath of `minidoom` needs 32-bit cells and a multiplier
  cell of ~50k neurons (`docs/capacity_doom.md` §2).
- Kernel perturbation campaigns now cover the renderer, fan-out and perspective blocks. The
  first fan-out run exposed and fixed the multi-pair replay in §10.3; the fixed 100-copy
  cluster rerun is still due.
