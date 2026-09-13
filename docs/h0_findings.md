# Stage H0 findings — what the fly wiring can and cannot host (2026-09-13)

Numbers come from `docs/h0_report.md` (generated from `data/h0/results.json`). Labels:
substrate **profile 2**; **isolated and full-graph** execution; **hybrid** orchestration
(the host injects DATA tokens and reads spikes); **external** compilation (hand-designed).

## The question H0 was built to answer

Does MCNS v1.0 contain usable motifs for a register, a gate, a completion detector, and a
reset path? Can they be connected without disproportionate relay circuitry? Can they run
with the rest of the brain present?

## Answers

**1. Motifs: yes, plentifully.** Under the measured physics of the model (one synchronous
input of ≈ 160 synapse-equivalents fires a neuron; a two-neuron excitatory loop holding a
spike runs at 213 Hz), the requirements are: latch edges ≥ 57 synapses both ways, AND
inputs ≥ 4, OR / completion / reset-drive edges ≥ 12–57, and reset inhibition ≥ 85
synapses onto **both** latch members (k_max = 4 weight scaling). The connectome offers
1,223 mutual cholinergic pairs for latches, 5,004 cholinergic completion candidates, 141 of
them with four resettable latches, and the search returned 50 complete embeddings in 1.8 s
(stopped at 50, not exhausted).

**2. Relay overhead: none.** Every role is a single anatomical hop: latch member → gate →
completion → inhibitory neuron → latch member. The chosen circuit is 15 neurons: latches
built from gnathal-ganglion local neurons (GNG108, GNG457, GNG014) paired with the
descending neurons DNge059 left and right; AND = GNG120; OR = GNG169; completion = GNG236;
reset = four GABAergic GNG neurons. Weight edits stay within bound (minimum margin 0.27
against the 0.25 floor, i.e. the tightest edge is scaled 3.6× its anatomical count).
107 parasitic anatomical edges among the 15 neurons had to be zeroed.

**3. Function in real wiring: yes, when isolated.** All four input words give the right rail
with completion and reset, and the result does not depend on the arrival offset of the
second operand (0–12 ms all correct). Latencies: completion 23 ms (00), 37 ms (01, 10),
56 ms (11); ready-for-next 27–61 ms. With the 107 parasitic edges left at anatomical
strength, the (1,1) case fails: parasitic zeroing is mandatory.

**4. Function with the surround present: no, unless the surround is quiet and the circuit
is shielded.**

| condition | correct | surround neurons recruited | surround spikes / 150 ms |
|---|---|---|---|
| full graph, surround live, silent | 0 / 4 | 19,680 – 23,917 | 99 k – 217 k |
| full graph, surround live, 2 Hz sensory background | 1 / 4 | ≈ 30,000 | ≈ 460 k |
| full graph, surround live, 1,000-neuron burst | 1 / 4 | ≈ 27,000 | ≈ 410 k |
| full graph, circuit outputs zeroed, silent | **4 / 4** | 0 | 0 |
| full graph, outputs zeroed, background | 1 / 4 | ≈ 30,000 | ≈ 460 k |
| full graph, outputs zeroed, burst | 1 / 4 | ≈ 27,000 | ≈ 410 k |

An unshielded latch broadcasts its 213 Hz train through the circuit's 4,830 outgoing
anatomical edges (72,274 synapses; 697 edges of ≥ 24 synapses, each enough to fire its
target on its own). That ignites 12–18 % of the brain within the transaction (the most
active recruits are octopaminergic VUM neurons, ExR6, olfactory local neurons, APL), and
the eruption feeds back through the circuit's 5,493 incoming edges at up to ±7,500
quanta per step (≈ 130 mV per step) and 1–5 million quanta per transaction. No designed
drive survives that. Zeroing the outgoing edges (documented Profile 2 silencing) restores
exact isolated behaviour — the same spike times as the isolated run — so that
configuration is classified *equivalent under stated isolation assumptions*, where the
assumption is a silent surround.

**5. The surround itself has no stable active regime in this model.** A 2 Hz drive on the
≈ 17,000 sensory neurons, or one synchronous burst of 1,000 cholinergic neurons, tips the
whole brain into sustained runaway (≈ 30,000 neurons active) with no circuit involved.
Uniform 0.275 mV synapses with no adaptation, no synaptic depression, and no tonic
inhibition make the connectome bistable between silence and seizure. "Operating within a
largely active surround" is therefore not testable for any circuit until the surround
model has a stable active state; that is a modelling task for Stage H, flagged now.

## Consequences for A1 / A2 and the compiler

- **Isolation cost is a placement objective.** The search now scores embeddings by
  outgoing anatomical strength; Profile 2 images will carry thousands of documented zero
  weights per circuit, and the manifest reports them separately from designed edges.
  Prefer neurons with few strong partners; consider a no-touch list for sensory, motor,
  and descending classes (the register bits here landed in descending neurons).
- **Stray-input tolerance becomes a contract field.** The circuit's incoming surround
  edges must be zeroed too unless the surround is provably quiet; the A1 harness will
  measure how much stray input per step each primitive tolerates (the AND's rejection
  margin is 2.4 mV of a 7 mV gap).
- **Stage latency is tens of milliseconds, not 2 ms.** Latches emit trains, so gates on
  latches work in rate mode (AND ≈ 56 ms, OR ≈ 23–37 ms); the spec's 2 ms per stage
  assumption is off by 10–30× for this substrate at default parameters. Single-pulse
  (coincidence) gates on quiet inputs and bounded threshold edits are the levers A1 will
  measure.
- **Reset contract:** a single inhibitory spike must deliver ≥ 1.5× the loop drive to
  both latch members (85 synapses at k = 4); one member alone needs 3× and still misses
  phases; a train of four pulses 8 ms apart needs only 1× on one member.
- **Latch timing is phase-free.** Because gates integrate sustained trains, operand
  arrival offsets up to at least 12 ms do not change the result — a useful property for
  the self-timed protocol, paid for in latency.
