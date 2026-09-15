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
| neurons | 13,727 (8-bit cells; the two memories are 8 words each) |
| pipeline latency, load to first pixel | 4.54 s |
| throughput, self-paced | one pixel per 1.18 s (8 columns, 8 correct) |
| the same loop on the sequencer | ~32 instructions per column ≈ 32 s |
| gain | ~27× per kernel at about the machine's neuron count |
| simulation | 25 s wall for 13 s of neural time on one CPU core |

The kernel reference (`compiler/kernel.kernel_reference`) equals the IR interpreter's pixels for
three headings, and the neural pipeline equals the kernel reference for eight columns
(`tests/test_kernel.py`, the slow test ~30 s).

## 4. What it is not yet

- One consumer per master: a value read by two cells needs a joined "free" rail (a C-element
  over the consumers' REQ-false rails); `NotImplementedError` today.
- Straight-line bodies only: no branch inside the body (a select cell — both arms computed, one
  chosen by a condition token — is the dataflow form of `if`), no loop-carried values other than
  the streamed induction variable, one OUT per token.
- Kernel parameters are image-time constants; a per-frame parameter (the heading) needs a
  parameter port the host rewrites between frames, which is a staged register like the input.
- The cells are 8-bit; the Q16.16 datapath of `minidoom` needs 32-bit cells and a multiplier
  cell of ~50k neurons (`docs/capacity_doom.md` §2).
- No perturbation campaign yet on a kernel; the handshake's margins (55 ms veto recovery, 86 ms
  relay recovery, 80 ms reset paralysis) are the ones measured on the machine.
