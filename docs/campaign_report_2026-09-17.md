# DrosophilOS campaign report — 17 September 2026

Four days of work (13–17 September) against the plan approved on 13 September. This is the
status of the campaign: where each stage stands, what worked, what did not and why, what the
numbers say the machine is, and where the goalposts can go. Every figure is in `RESULTS.md`
or the stage notes under `docs/`, and every run is reproducible from the repository.

## 1. The bottom line

The feasibility question is answered. A C program in the Doom shape — a level, ray-cast
walls, a z-buffer, textures, a sprite that chases the player, a world update that moves the
player from input — compiles through our own compiler into a network of leaky
integrate-and-fire neurons, and that network computes the game update and draws the frame
with nothing on the host doing game logic. The frames it produces are pixel-for-pixel equal
to the C reference. Under 4 % synaptic noise, threshold and bias drift and stray spikes, the
kernels produce **no silent wrong value in 4,000 outputs** on that build and realization.
Three more realizations on 2026-09-20 (`docs/tick_stalls.md`) found three ways a wrong value
can still arise — a refused input word the host did not resend, a state cell's completion
root lit by a stray coincidence, and a commit request's idle rail whose re-ignition failed —
each localized from a spike capture and fixed with a regression test; the build with those
fixes gives **0 wrong values in 4,800 outputs over three seeds**, with 4 % of copies stalled
(fail-stops the machine itself can see).

Three limits are also measured, and they do not move with more of the same work:

1. **Speed.** A frame at 40 × 25 costs ~1,000 s of neural time and ~4 h of wall time on an
   H200. The cost is the handshake protocol, not the neuron: ~60 hops of bookkeeping per bit
   step, stacked in series. More brains cut the frame time to about a minute of neural time
   and no further.
2. **The fly's wiring cannot host it.** A 16-bit pixel kernel needs ~350,000 designed
   neurons; the fly has 166,700. On the fly's own synapses, one neuron per designed neuron,
   the wiring holds about one kernel cell. The 4-bit adder places at 61–63 % of its edges
   carried, and carried edges alone never compute.
3. **The whole simulated brain storms.** Under a 2 Hz sensory drive the connectome model
   erupts on its own (Shiu et al. parameters, no adaptation); a placed circuit's own output
   wakes 24,000 neurons and the echo swamps its hosts. This is a modelling problem shared
   with every published whole-brain LIF model at these parameters.

So the honest label on everything that runs is: **Profile 3** (free synthesis, fly-model
neurons, not fly wiring), **isolated execution**, **hybrid orchestration** (the host deals
tokens; the phase order is neural), **external compilation**. Section 6 is what to do about it.

## 2. The campaign against the plan

| stage | plan exit | status | evidence |
|---|---|---|---|
| **0** semantics, reference simulator, loader | three-level simulator agreement; Brian2 parity; loader reproduces published counts | **done** | float64 vs PyTorch identical traces; Brian2 spike-for-spike; 166,700 neurons / 25.58 M edges; 7,989 arithmetic vectors under UBSan |
| **A1** token storage, completion, reset, adder | 10⁶ handshakes under perturbation, 0 wrong | **done** | 10⁶ 4-bit transfers at mix B: 0 wrong (≤ 3 × 10⁻⁶), 22 incomplete; 4-bit adder 10⁵: 0 wrong |
| **H0** minimal circuit in MCNS wiring | works or not, with a report | **done, negative in the live brain** | 1-bit AND + register on 15 real neurons: 4/4 isolated, 4/4 with outputs silenced, 0–1/4 with the surround live |
| **A2/B** ALU, register, RAM, control machine, compiler | a compiled program reads input, computes, branches, stores, emits a pixel; interrupt at a safe point | **done** | 4-bit ALU/accumulator/RAM campaigns 0 wrong; control machine 100 random programs, 98 ok, 2 fail-stop, 0 wrong; Hello World on the sequencer, three-way agreement |
| **C** two nodes over FlyLink | exact channel with retries, duplicates, late ACK | **first milestone** | clean, dropped-event and late-ACK scenarios pass; no epochs, credits, corruption or backpressure yet |
| **D** world update kernel | 1,000 ticks equal to the reference | **shape done, not the exit** | 13-cell state kernel, 8 ticks of fresh input correct; doom4's tick moves player and imp; TMR and commit log not built |
| **E1** exact renderer | pixels identical to the reference | **done at small scale** | 160 × 100 frame 16,000/16,000; doom1 40 × 25 2,000/2,000; doom2 textured 2,000/2,000; doom4 sprite 720/720 |
| **F** cluster on Juno | scale by the capacity report; recovery | **stand-in only** | the batched simulator runs 8–128 brains on one GPU; no multi-GPU, no checkpoints, no recovery |
| **F2** neural execution control | Python control plane absent | **first step** | phase order in the substrate (phase gates, one-hot ring); token dealing still host |
| **E2** population coding | analog accelerator with fallback accounting | **not started** | — |
| **G** self-hosted compiler | neural VM bytecode | **not started** | — |
| **H** Profile 2 at workload scale; shuffled controls | measured statement of what the wiring contributes | **measured, negative** | adder 61 % carried vs 50 % random graph, 44 % weights shuffled, 65 % transmitters shuffled; ceiling 343 disjoint latch pairs; whole brain 0/30 live |

## 3. What worked

### 3.1 The substrate is trustworthy

The simulator is certified three ways and integer-weighted so spike traces are bit-identical
across CPU, GPU and batch sizes. The fp32 GeForce path on G2 reproduces the fp64 H200
campaign numbers within noise. The physics that everything else rests on was measured before
anything was designed: a synaptic jump peaks 9.2 ms later at 15.75 % of its size, so a gate
needs ~160 synapse-equivalents arriving together, and stage latencies are milliseconds, not
steps.

### 3.2 A primitive library with contracts, and the rules it settled

Two-neuron loop latches at 213 Hz; kill pairs; edge relays for one-pulse ignition; veto relays
(a one-shot AND of ordered operands with no exposure window); delay chains; a latched
completion tree; reset trains; watchdogs. Every primitive has a measured contract with a trial
count. The design rules that were paid for in dead ends: every gate reads latch trains; a relay
never races its own inhibitor; a level is not a token; a relay must be driven ≥ 55 ms after
any veto rail dies; FAULT is a state, not a pulse; a reset train paralyses a latch for ~80 ms.
Ten M1 designs and five A2 mechanisms were rejected on measurement, each recorded.

### 3.3 Arithmetic and a machine

A 4-bit ALU on veto relays only (1,150 neurons, 241–524 ms), a word register with staged
commit, an 8 × 4 RAM with exact veto-relay decoding, and a control machine (one-hot PC and
FSM rings, 4,230 neurons, ~1 s per instruction) that fetches, decodes, executes, commits,
loads, stores, branches, halts, and takes interrupts at safe points. All campaigned at mix B
with zero wrong values.

### 3.4 The compiler

DrosoC (a C subset) → pycparser → three-address IR → the IR interpreter (the oracle) → either
the accumulator machine or, for loops, **resident kernels**: one cell per operation, arrays as
memories, loop-invariant values as image-time parameters, the induction variable streamed as
tokens. Three comparison points on every program: C reference under UBSan, IR interpreter,
neural run. Kernels gave a 27× gain over the sequencer at the same neuron count and are what
made frames possible.

### 3.5 Frames, in the substrate

| program | size | copies × neurons | pixels right | neural time | wall |
|---|---|---|---|---|---|
| `frame.c` static view | 160 × 100 | 128 × 65k | 16,000 / 16,000 | 411 s | 2 h 13 m |
| `game.c` turning view, 3 frames | 80 × 50 | 128 × 65k | 12,000 / 12,000 | 344 s | 2 h 3 m |
| `doom1.c` level, ray cast, z-buffer, moving player, 2 frames | 40 × 25 | 32 × 460k | 2,000 / 2,000 | 310 s | 2 h 31 m |
| `doom2.c` + textures, neural pacing, 2 frames | 40 × 25 | 8 × 667k | 2,000 / 2,000 | 1,982 s (ring: 1,989 s) | 7 h 53 m |
| `doom4.c` + sprite that chases the player, 2 frames | 24 × 15 | 8 × 1.44 M | 720 / 720 | 2,711 s | 18 h |

![doom4: the imp closes in](img/doom4_24na_1_neural.png)

### 3.6 Reliability under noise, on kernels

The first 100-copy mix-B campaigns on kernel blocks found a silent duplicate output (a token's
value emitted twice, shifting the stream; in a state kernel, corrupting every later output).
Five remedies were built and each was judged by the same 100-copy campaign:

| remedy | perspective | fan-out | render | tick | verdict |
|---|---|---|---|---|---|
| baseline (§10.3 fix of the guard replay) | 760, **7 wrong** | 795 | 799, 1 wrong | 1,598, 2 wrong | 10 wrong / 4,000 |
| one-hot guards, wider margins | 470, 7 wrong | 737 | 725 | 1,171 | reverted |
| unconditional re-light from ACT^d | 714, 0 wrong | 768 | 773 | 1,471, 5 wrong | reverted |
| gated re-light (veto by REQ true) | 779, 0 wrong | 783 | 793 | 1,396 (22 stalls) | off |
| request-priority pairs | 764, 0 wrong | 765, 3 wrong | 792 | 1,411 (19 stalls) | off |
| **+ veto on a live false rail** | **768, 0 wrong** | **791** | **800 / 800** | **1,569, 0 wrong** | **default** |

The last one came from a spike dump of a stalled copy: the repair pulse re-lit a rail that was
already lit, gave it a second circulating spike, and the next clear could only slow it. Zero
wrong values in 4,000 outputs; 72 fail-stops, all visible to the host's timeout.

### 3.7 The flip-flop line, for the fly's inhibitory wiring

Because the fly's strongest reciprocal pairs are inhibitory and useless to an excitatory
latch, a flip-flop latch was built: two inhibitory neurons on a 58 mV bias in mutual
inhibition, an excitatory proxy for sign-correct readout, 212.8 Hz so every reader sees the
same statistics. Its lockstep failure mode was mapped and fixed (four-pulse set train: 0 in
16,000). The register on flip-flops: 10,000 mix-B transfers, 0 fail-stop, 0 wrong. The adder
with flip-flop storage places at 62.8 % carried (latch adder 61.6 %), its ten pairs on real
driven inhibitory pairs, and computes 50/50 on its image. 32 proxied flip-flops place at
99.5 % of edges carried where 32 excitatory latches reach 86 %.

### 3.8 Method

Every handshake change goes through the 100-copy mix-B campaign before it is believed; three
remedies that passed every clean test failed it. Independent review between stages (Kimi,
with Claude and OpenAI workers for implementation) found real defects each round: the
pair-aggregation bug in the full-graph image, the missing power-on pulse, the wrong §11.2
mechanism claim, the gated-re-light design.

## 4. What did not work

### 4.1 Connectome-constrained execution (Profile 2)

- **Capacity.** MCNS has 343 disjoint reciprocal pairs strong enough for a latch at the H0
  weight bound (1,047 at ×8, 2,896 at ×16). One 8-bit kernel cell needs 375; the pixel kernel
  ~5,300; the three Doom kernels ~35,000. Counting inhibitory pairs as flip-flops roughly
  doubles the supply, to about one pixel kernel at ×16.
- **What the wiring lacks is specific.** The placed adder carries 61 % of its edges; the
  missing ones are the broadcast reset and watchdog neurons (no inhibitory neuron in MCNS
  reaches 115 targets at strength), a relay fan-out node, and ~30 relay-inhibitor edges.
  Carried edges alone never reach ACCEPT; adding the missing logic edges gives a first sum,
  adding the resets gives the second. Loosening the weight bound or splitting hubs each shrinks
  the gap by a fifth and they do not add up.
- **The live brain.** Inside the whole simulated brain the placed adder computes 10/10 only
  with the hosts' 327,497 inputs from the brain silenced, or with its 302,263 outputs into the
  brain silenced and nothing else driven. With either live it computes 0/30. Latch-capable
  neurons are the brain's hubs (~6,500 external inputs each); scoring quietness cost a third
  of the coverage and still computed nothing.
- **What the wiring does contribute** is real but modest: 60 % of the adder against 50 % on a
  degree-matched random graph, 44 % with synapse strengths shuffled — and 65 % with
  transmitter labels shuffled, because the fly's best pairs are inhibitory.

### 4.2 Speed

- A cell cycle is ~300 ms: request, start, sample 58 ms later, compute, commit gated on every
  reader, master reset, done. A 16-bit array multiply is 6.6 s; a pixel 7–10 s; a tick
  30–60 s. The neuron-to-neuron hop is 5 ms; the rest is protocol.
- The pipelined multiplier (1.7 s throughput, ~20 s latency) made doom4 pixels cost 63 s
  instead of 30 s, and two 32 × 20 × 3-frame runs could not finish in 36 h. Why a throughput
  gain becomes a per-pixel loss when tokens do overlap within a phase is **not identified**;
  the array multiplier is the default again.
- The one-hot ring counter replaced a three-cell feedback counter and saves 12,000 neurons,
  but at scale saves no time (1,989 vs 1,982 s): the passes are bound by their cells.
- The simulator runs 14× slower than neural time at 5 M neurons on an H200 and scales with
  neuron count: one 40 × 25 frame per ~4 h. That wall time has two parts, and only one is
  measured: the **neural workload cost** (steps × neurons, set by the protocol's latency) and
  the **not-yet-profiled simulator cost** (per-step spike extraction to the host, Python
  scheduling, decoding). The aggregate timings do not say how much is which;
  `docs/perf_campaign.md` §2 is the instrumentation that will.

### 4.3 The kernel duplicate, in five attempts

Three of the five remedies passed every unperturbed test and failed the campaign (§3.6). Two
lessons: a handshake fix is not a fix until 100 noisy copies say so, and the spike dump of one
failing copy localizes a fault that no amount of reasoning about the clean model did (the
perspective duplicate, the tick stall).

### 4.4 Process failures worth remembering

- Two Doom runs at 5 h were killed when a 160 GB dump job landed on their node; the
  first dump itself came out truncated. Keep dump jobs ≤ 64 GB and off nodes running our
  own GPU jobs.
- Kimi reviews stalled four times on a 600 s idle watch until `idle_timeout_sec: 0`; workers
  cannot commit in their sandbox, cannot see the network, and commit the `data` symlink.
- A test passed while the imp was never drawn (a reused temporary); the doom3/doom4 tests
  now assert drawn pixels. A 30 s campaign ceiling truncated tick and perspective and was
  read as stalls. Pipes hid exit codes.

## 5. What the machine is, in the plan's own labels

| label | value | what would change it |
|---|---|---|
| substrate profile | **3**, free synthesis on fly-model neurons | Profile 2 needs ~10× the latch supply or ~100× smaller kernels |
| execution | **isolated**; full-graph runs only with the hosts' edges silenced | a stable active regime for the whole-brain model, which does not exist at these parameters |
| orchestration | **hybrid**: phase order neural, token dealing and frame pacing on the host | Stage F2's dispatch and commit in neurons |
| compilation | **external** | Stage G |
| reliability | 0 silent wrong in 4,000 kernel outputs at mix B; ≤ 3 × 10⁻⁶ on the channel at 10⁶ | the 10⁻¹⁰ objective stays analytical |
| speed | ~1,000 s neural per 40 × 25 frame; floor ~1 min per frame with unlimited brains | a different primitive library (graded values), not more brains |

The claim that can be made today, with those labels: a Doom-shaped engine — world update and
renderer both in spikes — runs on a network of fly-model neurons that our compiler emitted
from C, and its frames equal the reference. The claim that cannot: that it runs on the fly's
wiring, or in a living brain's traffic, or at any speed a player would notice.

## 6. Where the goalposts can go

| goal | what it would show | risk |
|---|---|---|
| **A. A graded-value primitive library (E2)** | values as levels, one hop per stage; exact arithmetic only for state; 10–50× per pixel with the same compiler | a second library measured from zero; noise correctness is the whole problem again |
| **B. Profile 2 at kernel scale (H)** | one kernel cell in real wiring, computing with the surround live; local resets | capacity and the storm; the negative result is already the strongest Stage H statement |
| **C. The hypervisor (C/F)** | tick brain + pixel brains over FlyLink, live input, a 160 × 100 frame per hour on ~100 GPUs | plumbing; the floor stays ~1 min neural per frame |
| **D. Write up the feasibility result** | fixes the claim at its four labels while the numbers are fresh | none |

Recommendation (revised 2026-09-17, `docs/perf_campaign.md`): write up the demonstrated
feasibility result now. In parallel, establish an instrumented performance baseline, improve
simulator execution without changing the neural computation, and test operation-specific
exact datapaths. Diagnose the multiplier regression and remaining tick failures with bounded
traces. Use the resulting complete-frame measurements to determine the scope of E2 and any
later cluster expansion. A's 10–50× is a proposed gain, not a measured one; E2 begins with
one bounded primitive whose benchmark includes encoding, settling, decoding, uncertainty
detection and the neural fallback, and the exact renderer is not replaced until the
complete-path comparison supports it. B's negative result is already the paper's most
interesting finding. C makes a slow thing wider.

## 7. Open items, in priority order

1. The remaining tick fail-stops — the recorded five copies do not transfer to the current
   build (a different noise realization); today's realization has three stalled copies and
   one refused input word, the latter localized and fixed on the host side
   (`docs/tick_stalls.md`, 2026-09-20); the copy-8 capture is the next spike dump.
2. Why the pipelined multiplier doubles the per-pixel cost. (`docs/a2/doom4_24na_h200.json`
   holds aggregate counters only — neuron and node counts, neural ms, faults, wall time — not
   the inter-commit event history; that history has to be captured first:
   `docs/perf_campaign.md` §5.)
3. Stage C's remaining protocol: epochs, credits, corruption, backpressure.
4. Stage D's exit: 1,000 ticks against the reference; TMR and the commit log.
5. A stable active regime for the whole-brain model (adaptation, depression, tonic
   inhibition) before any further full-graph claim.
6. The non-spiking cell classes need a model of their own.
7. Timestep refinement and CUDA transaction-level certification, specified and still not run.
