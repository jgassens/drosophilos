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
| 32-bit datapath | ALU ~9k neurons, array multiplier ~50k (16 rows re-timed), fits one 166k-neuron brain as a resident kernel |
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
   this item: nested loops as joined kernels, per-frame parameter ports, 32-bit cells.
2. **32-bit datapath as a resident kernel**, not as the sequencer's width: the multiplier is
   the cost driver and only the kernels need it.
3. **Memory-serving nodes** with ROM as fetch relays (3 neurons per bit) for map and texture
   data; RAM masters only for live state.
4. **Simulation throughput**: GPUs (Juno's H200 partition is not yet open to this account;
   the CPU partition is queued). The batched torch simulator is the production path; the
   CPU reference simulator is for correctness.
5. The exact channel (Stage C) and TMR/commit log (Stage D/F) stay on the path unchanged;
   they cost latency, not the frame budget.
