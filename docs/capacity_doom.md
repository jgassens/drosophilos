# Capacity analysis: what Doom costs on this substrate

The plan requires a whole-program capacity report before any run above four nodes. This is
the first one, from measured numbers, for the path from the current machine to a Doom-like
engine. It is the honest scale of the problem, not a promise.

## 1. Measured unit costs (clean model, Profile 3, 2026-09-15)

| item | cost | source |
|---|---|---|
| one sequencer instruction (fetch, ALU, commit, PC) | 0.92–1.13 s of neural time | `docs/a2_ram_control.md` |
| ALU op alone (4-bit) | 0.24–0.52 s; MUL 0.93–1.08 s | `docs/a2_alu_register.md` |
| program word (latches + fetch relays) | ~230 neurons | `bench/capacity.py` |
| data word (master with completion and reset) + ports | ~250 neurons (8-bit, with two read and two write ports) | `bench/capacity.py` |
| 4-bit ALU with multiplier / 8-bit ALU without | 2,451 / ~1,450 neurons | contracts |
| toy tick (player, monster, walls, health) | 133 instructions per 3 ticks ≈ 45 instructions per tick | `examples/tick.c` |
| toy frame (8 columns, table lookup) | ~250 instructions per frame | `examples/render.c` |
| simulation speed, CPU reference simulator | ~1 s of neural time per 45 s wall at 11k neurons | Hello World run |
| simulation speed, batched CPU torch (100 copies) | ~1 s of neural time per ~40 s wall at 1.3k neurons | accumulator campaign |

## 2. Scaling the same machine to Doom's numbers

Doom's world update on a small level touches tens of objects with ~10² fixed-point
operations each per tick (35 ticks per second in the original); a 160 × 100 frame is
16,000 pixels, each the end of a column loop of tens of operations. On the sequencer as it
stands:

| quantity | sequencer estimate |
|---|---|
| one tick (10 objects × 100 ops, Q16.16 at 32 bits) | ~10³ instructions ≈ 17 min of neural time |
| one frame (16,000 columns-pixels × ~30 ops) | ~5 × 10⁵ instructions ≈ 6 days of neural time |
| 32-bit datapath | measured: a 32-bit ADD cell ~10.8k neurons (with its input register), 1.99 s per token in a pipeline; a 16-bit MUL cell 27.9k neurons (`docs/a3_kernels.md` §5); a 32×32 multiplier cell is estimated at ~100k, so it must be shared |
| program of 5,000 words | ~1.2 M neurons of program image: seven brains of nothing but code |

The sequencer is the wrong execution model for the renderer by four orders of magnitude,
which is what the plan says: "the default execution model should be spatial dataflow".

## 3. What spatial dataflow buys, with the same primitives

A resident kernel is the IR of a loop body laid out as a pipeline of ALU stages joined by
staged registers, the composition already built (ALU channel → stage → master → next
stage's operand gate). Tokens flow; no fetch, no decode, no PC. Measured pieces give:

| pipeline property | value |
|---|---|
| stage latency | 0.3–0.5 s (an ALU op plus commit) |
| throughput with K stages in flight | K tokens per stage latency |
| a column kernel of ~10 stages, 8 columns in flight | one column per ~0.4 s, a frame of 16,000 columns per ~1.8 hours on one kernel |
| 1,000 brains × ~10 kernels per brain (15k neurons each) | ~10⁴ columns in flight: a frame in ~1–2 minutes of neural time |

**Built and measured (2026-09-15, `docs/a3_kernels.md`):** the renderer's column loop as a
four-cell kernel of 13.7k neurons runs one column per 1.18 s self-paced (4.5 s latency), eight
columns correct, against ~32 s per column on the sequencer: ~27× per kernel. The "stage
latency 0.3–0.5 s" above was optimistic: a cell is ~0.9 s (operand gates 60 ms, ALU ~400 ms,
commit ~400 ms), so a frame of 16,000 columns on one kernel is ~5.2 hours, and the cluster
estimate below scales accordingly (~3 minutes per frame at 10⁴ columns in flight).

**Measured frame (2026-09-15, Juno H200, 128 copies of the pixel kernel):** 16,000 pixels in
411 s of neural time, every pixel right, 7,992 s of wall time; the slideshow's three 80 × 50
frames in 344 s of neural time. The cluster estimate below is measured at 128 brains.

**Measured Doom-shaped program (`examples/doom1.c`, 2026-09-15):** a tick kernel moving the
player (19 cells), a column pass casting a four-step ray per column into RAM height and
distance buffers (48 cells), a pixel pass (13 cells): 443k neurons per copy at 16 bits, so
one copy is about three brains — the column kernel alone is ~265k. The cells, not the
tables, are the cost (~5.5k neurons per 16-bit cell); the constant bits of a ROM are now
one relay each. An 8-bit column kernel would halve it; a shorter ray walk (two steps) or a
distance table per (cell, angle) would halve it again.

**Measured pixel kernel (2026-09-15):** `examples/frame.c`, one pixel per token, sixteen
cells, 64,948 neurons at 16 bits, ~1.2 s per pixel per kernel: a 160 × 100 frame on 128
kernels is ~150 s of neural time, on 1,000 ~20 s. The simulator, not the substrate, is the
wall-clock cost: 128 copies are 8.3 M neurons, ~30–50 ms per step on an H200, so a frame is a
run of hours (`docs/a3_kernels.md` §9).

So the target regime for "Doom running on fruit-fly brains" is a frame per minute or two of
neural time on the 1,000-brain cluster, with the world update on a few control brains at a
tick per ~20 s. In simulation each neural second of 166 M neurons costs GPU hours; a frame is
a run, not a video. That is the endpoint this plan can reach; it is a slideshow computed
inside the substrate, and every pixel of it is a spike-decoded word.

## 4. Consequences for the order of work

1. **Kernel compiler before more ISA.** Done for straight-line bodies (`compiler/kernel.py`,
   `lib/kernel.py`): the toy renderer's column loop runs as a kernel with the columns in flight
   and a measured ~27× throughput gain; select cells, fan-out values, feedback state and
   several outputs followed, and the toy world update runs as a state kernel (6 s per tick,
   ~7.5× the sequencer: a dependent loop gains only the sequencer's overhead). Next inside
   this item: nested loops as joined kernels, per-frame parameter ports. Shifts, ROM tables
   and a 16-bit perspective column (reciprocal table × scale) run; the array multiplier's
   n² latency (6.6 s per 16-bit product) is now the number to beat: a carry-save multiplier.
2. **32-bit datapath as a resident kernel**, not as the sequencer's width: the multiplier is
   the cost driver and only the kernels need it.
3. **Memory-serving nodes** with ROM as fetch relays (3 neurons per bit) for map and texture
   data; RAM masters only for live state.
4. **Simulation throughput**: GPUs (Juno's H200 partition is not yet open to this account;
   the CPU partition is queued). The batched torch simulator is the production path; the
   CPU reference simulator is for correctness.
5. The exact channel (Stage C) and TMR/commit log (Stage D/F) stay on the path unchanged;
   they cost latency, not the frame budget.

## 5. Profile 2: can a kernel live in the fly's own wiring? (measured 2026-09-15)

Everything above runs under Profile 3 (free synthesis, connectome-inspired). Stage H0 showed
a 15-neuron circuit running in real MCNS neuron IDs under Profile 2 (anatomical edges only,
weights scaled at most k_max = 4 above the synapse count, parasitic edges zeroed, a silent
surround). What a kernel would need, against what the connectome offers:

| the four-cell render kernel needs | MCNS v1.0 offers, by the weight scale allowed |
|---|---|
| 10,736 neurons; 19,404 edges (13,259 excitatory, 6,145 inhibitory); Dale's law holds but for 5 neurons; out-degree mean 1.8, max 455 (the reset and ACT^d hubs) | 166,700 neurons; 25.6 M edges with ≥ 1 synapse |
| ~5,300 latches = reciprocal cholinergic pairs with ≥ 57 synapses each way (k_max 4) | 1,223 such pairs (k_max 4); 4,040 at ≥ 30 synapses (×1.9); **12,315 at ≥ 15 (×3.8)**; 36,330 at ≥ 8 |
| 13,300 excitatory edges of a median 57 synapses (k_max 4) | 99,501 at ≥ 57; 987,070 at ≥ 15 |
| 6,100 inhibitory edges | 75,486 at ≥ 57; 576,457 at ≥ 15 |

So: a 4-bit ordered adder (~300 latches) fits the H0 weight policy in count; a whole cell
(~1,300 latches) needs about ×2 the scale; a four-cell kernel needs weights scaled ~4× above
the anatomical count (k_max ≈ 15), or a Profile 3 supplement of added latch edges. Counts
are necessary, not sufficient: each latch's two members must also be adjacent, with the
right signs and strengths, to their relay, veto and reset neurons, and the hubs (a reset
interneuron driving 455 latch members) have no anatomical counterpart — the placement search
must split them into trees of inhibitory neurons, the relay overhead H0 measured as zero for
its 15 neurons. That search, netlist-driven rather than hand-designed as H0's was, is the
first task of Stage H; its target is the 4-bit adder, then one cell.
