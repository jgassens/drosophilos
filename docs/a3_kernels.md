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

### 9.2 Textures and a sprite (`examples/doom2.c`, `examples/doom3.c`)

`doom2.c` textures the walls: the column pass also stores the hit position's texture column
(`ubuf[col] = ((hx >> 1) + (hy >> 1)) & 7`), and the pixel pass reads the texture row by a
reciprocal table and a multiply (`v = ((prow - top) * recip2[h]) >> 8`) from an 8 × 8 brick
texture with a bright and a dark bank (beyond two cells). `doom3.c` adds one *thing* in Doom's
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
adds a wrapping counter (three cells: `cnt = cnt == K−1 ? 0 : cnt + 1`, requested by the
pass's output cell) and the end pulse is the counter's done, delayed 32 ms, vetoed by the
count's "not zero" rail; the tick phase ends at the tick kernel's state landing (its state
carrier's done — measured: ending it at the tick *token's* landing let the next frame's
columns read the old heading). The first phase is open at power-up.

The host then only deals tokens in program order, as fast as each register's READY allows;
tokens for a closed phase wait in their register's stage. Measured on the two-pass renderer
(three kernels, 36,880 neurons with the two counters): two frames, sixteen pixels and two
frame records equal to the interpreter's, no fault, with a barrier-free schedule
(`tests/test_kernel.py::test_neural_pacing_replaces_the_host_barriers`). What stays with
the host: dealing tokens to copies, the per-copy token counts (image constants), and the
tick's *input* (the game's controls, which are input by definition). The benchmark label
for the frame loop's ordering moves from *hybrid* to *neural*; the dealing stays hybrid.
Both drivers take `--pacing neural` (`bench/render_doom.py` deals columns,
`bench/render_game.py` pixels); the per-copy counts are image constants, so the columns or
pixels must divide evenly across the copies.

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
- No perturbation campaign yet on a kernel; the handshake's margins (55 ms veto recovery, 86 ms
  relay recovery, 80 ms reset paralysis) are the ones measured on the machine.
