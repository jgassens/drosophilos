# DrosophilOS architecture specification (v1, 2026-09-13)

> Author: Jeremiah Gassensmith. Archived verbatim from the design document that set the
> project's direction; the OS was renamed from FlyOS to DrosophilOS afterwards (sub-names
> FlyISA, FlyASM, FlyLink are unchanged). The plan that implements it is `docs/plan.md`;
> the review that revised the plan is `docs/review-2026-09-13.md`.

# FlyOS: a connectome-constrained, asynchronous neural computer

**The strongest architecture is not "1,000 flies pretending to be x86 processors." It is a distributed neural dataflow machine with exact spike-coded control and memory, specialized population-coded arithmetic, and explicit communication between independently scheduled neural nodes.**

The key design decision is to separate **how information is physically represented** from **what the program is allowed to assume about that information**. Spikes can replace voltage levels without abandoning exact integers, addresses, and control flow. Conversely, population activity can represent vectors efficiently without being trusted to decide whether a pointer addresses the player's position or an enemy's health.

The specification below treats the gate library, ISA, compiler, and operating system as proposed engineering. Where an existing fly circuit supplies a relevant computational operation, that biological precedent is identified separately.

## 0. Define the machine and the benchmark boundary

### 0.1 The hardware specification needs a dataset identity

Your approximately **166,000 neurons and 125 million synapses** specification now corresponds to the **male central nervous system connectome**, including the brain **and ventral nerve cord**, rather than the older female brain-only reconstruction. Codex currently lists **MCNS v1.0 with 166,700 neurons**; the September 3, 2026 release describes approximately **125 million synaptic connections**. The distinction matters because the nerve cord expands the available sensorimotor circuitry. Also distinguish synaptic contacts from aggregated neuron-to-neuron connections when importing the graph.

Every FlyOS hardware image should therefore specify:

```text
Connectome identity and version
Neuron IDs and cell-type annotations
Synaptic contacts and aggregated adjacency
Neurotransmitter/sign assumptions and confidence
Neuron and synapse dynamical models
Permitted parameter changes
Permitted external input/output neurons
Added, suppressed, or rerouted connections
```

A connectome is the starting topology, not a complete executable electrical model. A relevant precedent is the 2025 Loihi 2 simulation preprint, which implements a roughly 140,000-neuron fly-brain model across 12 chips using simplified neuron dynamics. That demonstrates a route to executing connectome-derived dynamics on neuromorphic hardware—not an already programmable fly CPU.

### 0.2 Maintain three clearly distinguished substrate profiles

| Profile | Allowed construction | Meaning of a successful Doom run |
|---|---|---|
| **Native-function model** | Preserve connectivity and physiologically constrained dynamics; manipulate defined inputs | Existing fly circuitry supports the computation under the permitted interface |
| **Connectome-constrained programmable model** | Preserve the graph while adjusting bounded weights, thresholds, gains, and permitted silencing | The fly graph can host an engineered computing architecture |
| **Connectome-inspired neural hardware** | Add relays, alter connections, or synthesize new circuits | A fly-inspired SNN runs Doom; stronger biological claims require additional evidence |

I would develop the ISA first on the third profile, then progressively constrain its implementation toward the second. The first profile is a separate, much stricter scientific target.

For living preparations, independent control of every simulated weight cannot simply be assumed. Each preparation would require measured input accessibility, output observability, stability, and trainable circuit behavior. An identical graph can be copied in simulation; a living individual is not an interchangeable copy of the specimen used to reconstruct that graph.

### 0.3 Define "runs Doom" so the host cannot quietly do the work

Let game state be \(S_k\), player input \(U_k\), assets \(A\), and displayed pixels \(P_k\):

\[
S_{k+1}=F(S_k,U_k),\qquad P_k=R(S_k,A).
\]

For a neural Doom execution claim, **both \(F\) and \(R\) must execute through the neural substrate**.

An external system may integrate neuron equations, transport spike events, transduce user input, load an initial program image, and display already computed pixels. It must not calculate collisions, look up textures on behalf of the neural renderer, choose game branches, or rasterize geometry.

A neural network that supplies movement commands to an externally running Doom instance is **playing Doom**, not running it.

There is also a separate distinction between **cross-compilation into neural hardware** and **compilation performed by neural hardware**. The latter requires a self-hosted compiler or interpreter, addressed below.

---

## 1. Instruction set architecture: FlyISA

### 1.1 Use a mixed-code ISA with explicit precision types

For an initial spiking implementation, a normalized current-based neuron model can take the form

\[
\tau_{m,i}\frac{dV_i}{dt}
=
E_{L,i}-V_i+
R_i\left[
I_i^{\mathrm{bias}}+
\sum_j w_{ij}(\kappa_{ij}*s_j)(t-d_{ij})
\right],
\]

where

\[
s_j(t)=\sum_n\delta(t-t_j^{(n)}).
\]

Threshold crossing emits a spike, followed by reset and refractoriness. Synaptic kernels \(\kappa_{ij}\), delays \(d_{ij}\), and weights \(w_{ij}\) determine temporal integration.

This is an implementation model, not a claim that every fly neuron behaves this way. In particular, the mushroom-body APL neuron is nonspiking and has spatially localized activity. A physiologically grounded implementation should retain graded or compartmental dynamics where needed rather than forcing every cell into one uniform SNN model.

I would define four architectural data classes:

| Architectural type | Physical representation | Permitted uses |
|---|---|---|
| **Exact symbols and integers** | Dual-rail spike tokens with validity and acknowledgment | Instructions, addresses, counters, branches, game state |
| **Bounded approximate scalars** | Population rate or interspike interval | Lighting, interpolation, selected geometry |
| **Angles and vectors** | Population phase and amplitude | Heading, coordinate transforms, vector operations |
| **Sparse signatures** | Sparse population activation | Associative lookup, novelty detection, cache candidate selection |

Conversions between these types must be explicit instructions. An approximate scalar does not silently become an address.

### 1.2 Exact values: dual-rail spike tokens

Represent each logical bit using two distinguishable neural channels:

\[
b=0\rightarrow \text{spike on rail }b_0,
\]

\[
b=1\rightarrow \text{spike on rail }b_1.
\]

The architectural interpretation is:

| Rail activity within a transaction | Meaning |
|---|---|
| Only \(b_0\) | Valid zero |
| Only \(b_1\) | Valid one |
| Neither | Not yet present, or missing |
| Both | Invalid token |

**Silence is not zero.** Otherwise, a failed neuron or lost event becomes an apparently valid data value.

A word consists of multiple bit tokens plus a completion condition. Receiver-side neural buffers hold arriving bits until the entire word is valid. A consumer then acknowledges receipt; the producer clears its transaction state before reusing the channel.

This supports asynchronous execution without demanding simultaneous arrival of all 32 bits. Coincidence-sensitive arithmetic circuits receive locally regenerated, aligned signals from those buffers.

Physical redundancy can be introduced within each rail, but "two rails" does not imply "two neurons per reliable bit." Buffering, restoration, acknowledgment, and error detection all consume additional circuitry.

### 1.3 Why 10 Hz versus 100 Hz should not be the main digital encoding

Rate coding is useful, but its precision depends on observation time and population size.

As an illustrative noise model, suppose spike counts are Poisson:

\[
N\sim\operatorname{Poisson}(rT).
\]

During a 10 ms window:

\[
E[N\mid 10\text{ Hz}]=0.1,\qquad
E[N\mid 100\text{ Hz}]=1.
\]

The probabilities of seeing no spike are approximately

\[
P(N=0\mid10\text{ Hz})=0.905,
\]

\[
P(N=0\mid100\text{ Hz})=0.368.
\]

Thus, a silent 10 ms window does not reliably distinguish the two states.

For \(M\) independent, similarly tuned neurons, the relative count noise scales as

\[
\mathrm{CV}\approx\frac{1}{\sqrt{MrT}}.
\]

That is a tradeoff among neurons, latency, and precision—not free information density. Correlations would weaken the benefit of increasing \(M\).

**Design choice:** use rates for quantities that tolerate uncertainty, but use explicit symbol codes for instruction selection and memory addresses.

### 1.4 Temporal values: intervals with declared resolution

A scalar can be represented by two spikes separated by an interval:

\[
x=\frac{\Delta t-T_0}{\alpha}.
\]

Here \(T_0\) supplies a baseline interval and \(\alpha\) sets units. Addition can be implemented through controlled accumulation of delays, with reference-offset compensation; comparisons can use race or coincidence circuits.

There is an established engineered precedent: the **STICK** architecture constructs arithmetic, memory, and other computational operations using interspike-interval representations. It is evidence that temporal-code computation can be synthesized—not that an unmodified fly connectome already implements those circuits.

Temporal resolution must be budgeted. If adjacent representable values require a guard interval \(g\), then \(b\) bits require approximately

\[
T_{\mathrm{range}}\geq(2^b-1)g.
\]

For an **illustrative**, not experimentally established, \(g=1\) ms, 256 levels span at least 255 ms. More compact timing requires lower jitter, more elaborate coding, or more neurons.

Across brain boundaries, interval-coded values should be **received, validated, and regenerated against a local reference**. Letting accumulated network jitter directly modify numerical values is unsuitable for exact computation.

### 1.5 Population vectors: a genuine biological accelerator opportunity

For preferred directions \(\phi_j\), an engineered angular representation could use

\[
r_j=r_{\mathrm{base}}+\alpha a\cos(\theta-\phi_j),
\]

with parameters constrained to maintain nonnegative rates. The first population harmonic represents a complex vector approximately proportional to

\[
z=a e^{i\theta}.
\]

This provides a natural representation for heading and two-dimensional vectors.

The biological connection here is unusually strong: fly central-complex circuitry has experimentally supported mechanisms for vector addition and body-centered-to-world-centered coordinate transformation. The reported representation uses activity-pattern amplitude for vector length and phase for direction.

FlyISA could expose:

```text
VADD.POP       destination, vector_a, vector_b
ROT2.POP       destination, vector, heading
ANGLE.UPDATE  heading_state, angular_increment
QUANTIZE      exact_result, uncertainty, population_value
```

`QUANTIZE` should return uncertainty or a status flag. A comparison near a collision boundary must either establish a sufficient numerical margin or fall back to exact arithmetic.

### 1.6 Instructions are neural processes, not arbitrary firing frequencies

An opcode should select a resident neural operation. A simple decoder could convert an exact opcode word into a one-of-\(K\) operation-selection population.

There is no architectural benefit in defining "10 Hz means ADD; 100 Hz means MULTIPLY." That makes instruction decoding an estimation problem.

A compact initial instruction set would be:

| Family | Instructions | Neural implementation |
|---|---|---|
| Data movement | `CONST`, `MOV`, `SELECT` | Token routing, buffered selection |
| Exact arithmetic | `ADD`, `SUB`, `CMP`, `SHIFT`, `MUL` | Verified Boolean/threshold circuit macros |
| Bit operations | `AND`, `OR`, `XOR`, `NOT` | Dual-rail logic |
| Memory | `LD`, `ST.STAGED`, `COMMIT` | Address decoding and controlled state updates |
| Control | `BR`, `CALL`, `RET`, `JOIN`, `WAIT` | State-machine transitions and completion tokens |
| Communication | `SEND`, `RECV`, `MCAST` | Neural mailboxes and port codecs |
| Approximate arithmetic | `ROT2.POP`, `VADD.POP`, `RECIP.APPROX` | Population or interval circuits |
| Conversion | `ENCODE`, `QUANTIZE`, `CHECK.RANGE` | Explicit code-conversion circuits |
| Kernel | `IRQ`, `IRET`, `SET.EPOCH`, `CALIBRATE` | Privileged control networks |

An exact arithmetic instruction must specify word width, signedness, overflow, rounding, and fault behavior. For example:

```text
ADD.WRAP.I32
ADD.SAT.I16
MUL.Q16_16
CMP.SIGNED.I32
```

Original Doom uses 16-fractional-bit fixed-point definitions, making a signed 32-bit Q16.16 path a natural compatibility target. Products generally need a wider intermediate before rescaling. Reducing this precision is a legitimate simplification, but changes the numerical specification.

### 1.7 Prefer spatial compilation, with a small interpreter as a secondary path

The default execution model should be **spatial dataflow**: an instruction becomes a resident circuit, and tokens activate it when its inputs are ready.

For example, an illustrative FlyASM fragment might be:

```text
.on TICK(k):

    LD.Q16_16       x,       [player.x]
    LD.Q16_16       vx,      [player.vx_per_tick]

    ADD.WRAP.I32    x_trial, x, vx
    CALL            collision_resolve, x_trial, player, world
    RECV            x_next, collision_resolve.result

    ST.STAGED       [player.x], x_next
    COMMIT          world_state, k + 1
    MCAST           render_workers, SNAPSHOT(world_state, k + 1)

    RET
```

Here, register names identify logical channels or storage assemblies, not necessarily registers in a centralized processor. The compiler routes the dependency graph.

A neural instruction interpreter remains useful for irregular control, dynamic programs, and eventual self-hosting. It trades much lower circuit duplication for repeated fetch/decode overhead.

Each compiled primitive needs a measurable contract:

\[
\mathcal C=
\{\text{resources, input code, output code, jitter tolerance,
latency, initiation interval, error rate, retention}\}.
\]

Without these contracts, adding instruction names produces a language, not a computer architecture.

---

## 2. Logic gates, memory, program counters, and interrupts

### 2.1 Construct a cascadable gate library

Assume buffered inputs produce normalized postsynaptic contributions of magnitude one during a local evaluation window.

A threshold unit evaluates

\[
y=H\left(\sum_i w_i x_i-\vartheta\right).
\]

Illustrative constructions are:

| Function | Excitatory inputs | Threshold |
|---|---|---:|
| OR | \(a+b\) | \(0.5\) |
| AND | \(a+b\) | \(1.5\) |
| Three-input majority | \(a+b+c\) | \(1.5\) |

These are mathematical constructions requiring input alignment and bounded disturbance. A 0.5 threshold margin is not automatically preserved under leakage, variable synaptic strength, or background activity.

Dual-rail encoding makes inversion particularly simple:

\[
(\neg a)_1=a_0,\qquad(\neg a)_0=a_1.
\]

A complete dual-rail AND is

\[
y_1=a_1\land b_1,\qquad
y_0=a_0\lor b_0.
\]

XOR requires multiple threshold stages:

\[
y_1=(a_1\land b_0)\lor(a_0\land b_1),
\]

\[
y_0=(a_1\land b_1)\lor(a_0\land b_0).
\]

The completion network must account for all required operands, even where one output rail can be determined early.

For single-rail implementations, inhibitory veto circuits can construct NOT or NAND using a reference/evaluation pulse. However, the reference pulse must arrive only after the inhibitory decision is established; otherwise a transient false output can escape.

**The reusable primitive is therefore not merely a threshold neuron. It is a threshold computation plus input buffering, timing control, output restoration, completion detection, and reset.**

### 2.2 Build the ALU from these primitives

A full adder is defined by

\[
s=a\oplus b\oplus c_{\mathrm{in}},
\]

\[
c_{\mathrm{out}}=
\operatorname{majority}(a,b,c_{\mathrm{in}}).
\]

Possible implementations:

**Bit-serial ALU.** Reuse a small adder across a word. This minimizes neural circuitry but adds repeated memory access and control transitions.

**Ripple-carry ALU.** Instantiate one stage per bit. Resource cost is moderate, but the carry path has \(O(w)\) depth.

**Carry-lookahead ALU.** Use parallel propagate/generate networks. Logical depth approaches \(O(\log w)\), at the cost of greater fanout and more demanding connectivity.

For FlyOS, I would combine a compact exact control ALU with spatially replicated adders and comparators in rendering kernels. Multipliers would be fewer, shared resources unless placement measurements justify replication.

Do not distribute the individual carry stages of one ordinary addition across separate brains. The interbrain latency would overwhelm the benefit.

### 2.3 Registers require engineered stable states

A candidate bit cell consists of two recurrent excitatory assemblies, \(E_0\) and \(E_1\), coupled through inhibitory interneurons. The intended stable regimes are:

\[
E_0\text{ active},\ E_1\text{ suppressed},
\]

or

\[
E_1\text{ active},\ E_0\text{ suppressed}.
\]

A coarse assembly-level model is

\[
\tau\dot x_0=-x_0+
\phi(g_{\mathrm{self}}x_0-g_{\mathrm{inh}}x_1+I_{\mathrm{reset}}),
\]

\[
\tau\dot x_1=-x_1+
\phi(g_{\mathrm{self}}x_1-g_{\mathrm{inh}}x_0+I_{\mathrm{set}}).
\]

Appropriate parameter regimes would need to be demonstrated to support bistability and tolerate perturbations. This does **not** mean any mutually inhibitory neuron pair is a usable flip-flop. Recurrent excitation may require a multi-neuron loop found in the actual graph.

The bit cell also needs a nondestructive readout, controlled write enable, invalid-state detection, and a retention specification. Word registers should use staged writes:

\[
\text{old committed word}
\rightarrow
\text{new word in staging cells}
\rightarrow
\text{validated commit}.
\]

This prevents a consumer from reading a mixture of old and new bits.

### 2.4 Separate RAM, ROM, and associative memory

These are different neural engineering problems.

**Working RAM** stores changing state in recurrent activity or another explicitly readable and writable state mechanism. It requires retention, refresh where applicable, and error correction.

**Program and asset ROM** can encode constants in stable connectivity, synaptic parameters, or dedicated state circuits. However, a weight is not automatically an addressable memory location. A decoder and readout circuit must transform an address into the requested data.

**Associative memory** maps patterns to related patterns or membership estimates. Mushroom-body-inspired circuitry is relevant here, but it is not a substitute for byte-addressable RAM.

The fly-inspired hashing and novelty-detection literature supports sparse similarity signatures and Bloom-filter-like membership mechanisms—not lossless recovery of arbitrary stored files.

For an ideal conventional Bloom filter,

\[
p_{\mathrm{FP}}\approx
\left(1-e^{-kn/m}\right)^k,
\]

where \(m\) is the bit-array size, \(n\) the number of inserted keys, and \(k\) the number of hashes. This trades storage against false positives; it does not retain the original values.

Within FlyOS, a sparse signature could nominate a texture-cache entry. An exact tag comparison must confirm the match before the texture is used. Neural corruption may also introduce failure modes absent from the ideal data structure.

Similarly,

\[
\log_2 {N\choose K}
\]

counts the distinguishable \(K\)-active patterns in an idealized population; it does not mean the circuit can independently store and reliably retrieve all those patterns.

### 2.5 Addressing should use owned objects, not globally shared neural RAM

An illustrative 32-bit logical word address could be divided as

```text
node: 10 bits | bank: 6 bits | word: 16 bits
```

Ten node bits accommodate the 1,000-node cluster. This describes an address namespace, not installed capacity.

An actual load requires neural logic to identify the destination node, select a bank, decode the word index, read the stored value, and return a validated response. Arbitrary neuron IDs are not intrinsically memory addresses.

For the first implementation, statically allocate objects and assign each a single writer. Later versions can use capability-like handles containing an object identity, access rights, bounds, and a generation number.

The fundamental synchronization event should be **object commit**, not hardware cache coherence across all brains.

### 2.6 Program counters are control-state machines

For spatially compiled code, use a one-hot program-state token:

```text
FETCH_INPUT → UPDATE_MOTION → COLLISION → COMMIT → NEXT_TICK
```

Only the active state enables its corresponding computation. It advances after receiving a valid completion token and, where required, a branch predicate.

A loop routes a completion token back to an earlier state. A conditional routes it into exactly one successor. A call requires a stored return continuation; recursion requires an explicit bounded stack.

The interpreter implementation instead stores a binary program counter, fetches an instruction from neural ROM/RAM, decodes it, and invokes resident circuits.

The native heading ring should not be treated as an exact program counter. Its useful variable is a bounded population phase, not an arbitrary, indefinitely reliable instruction address.

### 2.7 Native connectome motifs: useful starting points, not pre-existing computer parts

| Native circuitry | Established computational relevance | Proposed FlyOS use |
|---|---|---|
| Projection-neuron input to Kenyon cells with APL feedback | Sparse representations and inhibitory regulation | Sparse signature generation and associative indexing |
| Compartmental Kenyon-cell-to-mushroom-body-output pathways | Plastic associative mapping | Trainable lookup policies and calibration |
| Central-complex E-PG/P-EN/P-EG/Δ7-associated circuitry | Heading representation and recurrent navigation motifs | Bounded angular state and heading updates |
| PFN-to-hΔB-associated pathways | Vector operations and reference-frame transformation | Population-coded vector accelerator |
| Optic-lobe circuitry | Structured visual computation | Candidate substrate for repeated local compute kernels |
| Descending and nerve-cord pathways | Sensorimotor communication | Candidate external communication ports |

The mushroom-body assignments follow experimentally characterized local inhibition and compartmental organization; the navigation assignments draw on central-complex anatomical and functional studies. They do not establish a ready-made binary ALU or renderer.

The published circuits also come from particular datasets and specimens. Their corresponding cells and connections must be identified in the chosen MCNS version rather than treating neuron IDs across datasets as interchangeable.

### 2.8 The connectome compiler must perform constrained circuit embedding

Let \(A_{ij}\) specify permitted native connectivity. For a topology-preserving implementation,

\[
w_{ij}=A_{ij}\,\widetilde w_{ij}.
\]

The compiler may optimize \(\widetilde w_{ij}\) only within allowed strength and sign constraints. It must not invent an absent edge, arbitrarily change an excitatory pathway into an inhibitory one, or assume every unwanted projection can be perfectly suppressed.

The synthesis process should:

1. Find candidate subgraphs for each verified primitive.
2. Route logical connections through permitted paths, including required relays.
3. Match delays and insert neural buffering.
4. Optimize permitted parameters while preserving code contracts.
5. Test interference from the rest of the network.
6. Export the exact neuron IDs, connections, parameter changes, and measured behavior.

This is the principal unresolved implementation step: **a functioning circuit library on a freely designed SNN does not prove that all its elements can be embedded and isolated in the native fly graph.**

A legitimate output is a neuron-level netlist with provenance. A table saying "mushroom body = RAM, central complex = CPU" is not sufficient.

### 2.9 Neuromodulation belongs primarily to configuration and adaptation

Dopamine and octopamine are useful inspirations for selective learning and state-dependent modulation. For example, experiments show octopamine-dependent reinforcement acting through subsets of mushroom-body-targeted dopamine neurons. That is a structured modulatory pathway, not arbitrary independent control over every synapse.

An engineered three-factor learning rule might be

\[
\tau_e\dot e_{ij}=-e_{ij}+F(s_i,s_j),
\]

\[
\dot w_{ij}=\eta\,m_c(t)e_{ij},
\]

where \(e_{ij}\) is an eligibility trace and \(m_c\) is a compartment-specific modulation signal.

Within FlyOS, use this for calibration, adaptive associative modules, or compensating bounded drift. During exact-logic execution, the simulation should hold verified parameters fixed unless the instruction contract explicitly includes adaptation.

**Instruction dispatch should change which circuit receives tokens, not rewrite its synapses on every instruction.** Fast switching should use ordinary signal routing and inhibition/disinhibition; learning-based reconfiguration is a separate operation.

### 2.10 Interrupts should be precise at transaction boundaries

An interrupt circuit needs an event detector, pending latch, priority selection network, saved continuation, and handler-entry circuit.

A proposed sequence is:

```text
External event
    → latch pending interrupt
    → select highest-priority eligible request
    → stop issuing new work for that task
    → complete or discard its current uncommitted transaction
    → save architectural state and continuation
    → enter interrupt handler
    → IRET restores continuation
```

A living neural system cannot be treated as though all spikes in flight can be frozen instantaneously. Therefore, FlyOS should support **safe-point preemption**, not arbitrary microstate suspension.

State to preserve includes registers, stack pointers, committed object versions, and resumable task identity. Spikes belonging to abandoned work must remain confined to uncommitted scratch circuitry or be rejected through epoch-aware interfaces.

---

## 3. The 1,000-brain cluster and neural hypervisor

### 3.1 Organize the cluster around coarse tasks

Use a hierarchical, message-passing organization: for example, 32 pods of 31 nodes, with eight additional nodes available for control services or redundancy. This is a proposed layout, not an anatomical requirement.

Within a pod, place tightly interacting tasks together. Across pods, exchange coarse objects: a committed scene snapshot, a block of geometry, a rendered tile, or a state-replication record.

A useful initial role separation is:

| Role | Primary responsibility |
|---|---|
| Control and world-state nodes | Game updates, sequencing, ownership, commits |
| Geometry workers | Transforms, clipping, projection |
| Raster workers | Wall spans, floor/ceiling spans, sprites, texture sampling |
| Memory-serving nodes | Program constants, map data, texture blocks |
| Service and replica nodes | Scheduling, validation, redundancy, recovery |

All roles consume neural resources. There is no resource-free hypervisor outside the accounting.

### 3.2 Schedule resident neural graphs

The simplest scheduler assigns work to circuits already resident on a node. Its neural metadata stores whether a worker is ready, what object versions it needs, its output capacity, and the next task identifier.

Dynamic migration should initially mean **send a task to another node that already hosts the appropriate kernel**. Arbitrary migration of a whole neural program, including its learned dynamics and hidden state, is a much harder feature.

The hypervisor's core responsibilities are:

- Resource ownership and bounded allocation.
- Mailboxes, scheduling, and deadlines.
- Object-version tracking and commit coordination.
- Fault detection and recovery.

For a strictly neural FlyOS claim, these decisions must also be implemented neurally. An external Python scheduler makes the system a hybrid prototype, which is useful but should be labeled accordingly.

### 3.3 Separate biological time, machine time, and game time

Three clocks must not be conflated:

\[
t_{\mathrm{neural}},\qquad
k_{\mathrm{transaction}},\qquad
k_{\mathrm{game}}.
\]

Neural time governs membrane and synaptic dynamics. Transaction time tracks completed operations. Game time advances when the required world update commits.

A simulated cluster must preserve causal ordering: a node cannot advance past a point where an earlier incoming event could still affect it. Positive modeled inter-node delays provide a basis for conservative synchronization and bounded lookahead.

Living nodes cannot roll back their physical dynamics. They require bounded-jitter communication, buffering, local retiming, and timeout/recovery behavior.

Slowing the game does not require slowing membrane biophysics. It means maintaining state while allowing more physical time between logical game updates.

### 3.4 The interbrain bus: address-event transport with neural endpoints

A proposed **FlyLink** event contains:

```text
source node
destination or multicast group
logical port / channel
logical timestamp
transaction epoch
sequence number
event flags
transport checksum
```

This is an address-event-style transport: the packet identifies a spike's source and timing rather than containing a host-computed game result. Existing neuromorphic implementations provide precedents for mapping and routing large spiking graphs, although the particular protocol here is proposed.

At the neural boundary, the mapping is

\[
I_j^{B}(t)=
\sum_{i\in P_A}B_{ji}\,s_i^{A}(t-L_{ij}),
\]

where \(P_A\) is Brain A's output-port population, \(B_{ji}\) is a predetermined input mapping, and \(L_{ij}\) is transport delay.

Brain A must neurally encode the message. Brain B must neurally validate and interpret it.

For the requested motor-to-sensory connection, tap identified output-neuron activity before muscle movement and stimulate defined recipient input channels. Waiting for a fly to move, observing that movement, and then converting it into another fly's sensory stimulus adds an unnecessary behavioral transduction stage.

In a biological implementation, the number of independently observable and controllable channels is a measured hardware limit. A connectome containing many neurons does not establish equivalent external bandwidth.

### 3.5 Latency requires a complete budget

For each link,

\[
L_{\mathrm{total}}=
L_{\mathrm{readout}}+
L_{\mathrm{encoding}}+
L_{\mathrm{queue}}+
L_{\mathrm{transport}}+
L_{\mathrm{stimulation}}+
L_{\mathrm{neural\ response}}.
\]

A fast Ethernet or optical link only reduces one term.

As an illustrative traffic budget, suppose each node exposes 256 logical event channels averaging 100 events/s. At 16 bytes per event:

\[
256\times100\times16
=409{,}600\ \mathrm{bytes/s/node}.
\]

Across 1,000 nodes, that is approximately **3.28 Gbit/s**, before additional framing, multicast replication, and redundancy.

These are design assumptions, not demonstrated biological port capabilities. Nor are event bits equivalent to useful arithmetic-result bits: handshakes and multi-spike codes can consume much of the traffic.

### 3.6 Use different transport semantics for different data

**Exact channels** need ordered transaction delivery, credits, complete-word validation, acknowledgments, and explicit timeout handling. The sender retains a value until its transaction is acknowledged.

**Approximate channels** may tolerate bounded timing variation or limited event loss, but must expose the resulting uncertainty.

Late events must not be blindly replayed into a new computation. Receiver interfaces should stage them by epoch and release a completed, valid message into the computational circuit only once.

A task identity such as

\[
(\text{epoch},\text{game tick},\text{task ID},\text{sequence})
\]

supports duplicate suppression and idempotent recovery.

Transport checksums detect packet corruption. They do **not** detect a neuron that computed the wrong valid-looking word before transmission.

### 3.7 Replicate architectural results, not identical spike rasters

Faults include missing events, invalid rail combinations, timing violations, state drift, incorrect valid words, and complete node failure.

For important control operations, execute three replicas and vote on decoded logical results. Identical spike timing is unnecessary.

If independent replicas each fail with probability \(p\), ideal triple modular redundancy gives

\[
p_{\mathrm{TMR}}=3p^2-2p^3.
\]

This benefit excludes voter failures and correlated errors. Three copies of the same incorrectly synthesized circuit will agree on the wrong answer. Use different placements, different implementations where possible, separate failure domains, and protected voting circuitry.

For replicated state ownership, a majority-backed log is appropriate under a crash-failure model; Raft supplies a well-defined precedent for replicated-log organization. Computational corruption requires additional validation—it is not solved merely by using a crash-tolerant consensus protocol.

During loss of a required majority, stop committing new authoritative state rather than permit two conflicting game worlds.

### 3.8 Checkpointing differs sharply between simulation and biology

A faithful simulation checkpoint may require membrane states, synaptic traces, refractory states, delayed-event queues, weights, modulators, and random-generator state.

A biological recovery mechanism should instead rely on **architectural checkpoints**: committed game objects, exact register values, task continuations, and reproducible program initialization.

After failure, restart a calibrated replacement node from those explicit values. Do not assume that reading a few neurons captures the preparation's complete dynamical state.

Rendering is particularly recovery-friendly: a tile can be regenerated from an immutable snapshot. World updates are more sensitive because repeating "apply damage" or "advance the door" twice changes the game.

---

## 4. Compiling and executing Doom

### 4.1 Compile the program, not x86

The released Doom engine is principally C, and the original source already separates game logic, fixed-point operations, rendering, and platform-facing functionality. FlyOS should target those semantics directly rather than emulate an x86 machine.

The proposed compilation pipeline is:

```text
Restricted Doom source
    → defined integer/fixed-point intermediate representation
    → explicit control flow, object ownership, and memory operations
    → neural dataflow representation
    → verified arithmetic/control/memory circuit macros
    → connectome-constrained placement and routing
    → neural program image
```

The intermediate representation should make overflow, bounds, aliasing, and completion dependencies explicit. Pointer-heavy structures can become indices into bounded object pools. Recursive traversals can become explicit stacks. Function-pointer dispatch can become finite-state-machine selection.

Whole-program specialization can remove unused networking, sound, menus, weapons, and platform services. It must not precompute the runtime responses to every possible input sequence.

### 4.2 Specify a reduced engine that remains meaningfully Doom-derived

A credible first target would retain a small sector-based level with variable floor and ceiling heights, doors, collision, one weapon, a few enemy states, sprites, health, and an exit.

Rendering could begin at **160 × 100 pixels with a 16-color palette**, a small set of textures, and no sound.

A regular-grid raycaster would be substantially simpler, but should be called a **Doom-like engine**, not treated as equivalent to compiling Doom's sector/BSP architecture.

The golden reference should be the same reduced engine running on an ordinary computer. Its game-state and rendering rules define the comparison.

Original Doom defines a **35 Hz logical game tick**. FlyOS can preserve that game-time convention without completing 35 ticks per wall-clock second; slow execution is still execution. Real-time performance is a separate benchmark.

### 4.3 WAD handling becomes neural ROM and explicit decoders

Doom's WAD support indexes named data lumps and reads their contents through a directory/cache mechanism. The source uses eight-character lump names.

For the first neural executable, an asset compiler should extract only the required map geometry, sector properties, BSP data, textures, sprites, palette, and animation tables. It then produces a compact ROM image.

Compile-time extraction is permissible just as linking constants into a conventional executable is permissible. During execution, however, texture lookup and asset decompression must occur through neural circuits—not host callbacks.

A small illustrative texture set is surprisingly manageable:

\[
32\text{ textures}\times16\times16\text{ pixels}\times4\text{ bits}
=4{,}096\text{ bytes}.
\]

Lossless run-length coding, repeated tile dictionaries, and neurally generated procedural patterns can reduce storage further. Each has a decoding cost.

For a benchmark requiring arbitrary WAD loading at runtime, the directory parser, bounds checks, and loading logic must themselves execute on the neural machine. That is a later capability, not something the first statically linked demonstration should quietly imply.

### 4.4 Game logic should be exact neural state machines

World updates are a good fit for explicit state ownership:

\[
S_{k+1}=F(S_k,U_k).
\]

The authoritative world-state task reads one committed snapshot, processes input, computes movement and collision, advances object state machines, and commits the next snapshot.

Enemy behavior can be represented as states such as idle, pursue, attack, damaged, and dead, with exact transition predicates. Timers are counters. A pseudorandom generator is a neural integer circuit or indexed constant table, with state included in the checkpoint.

Assigning game logic to the central complex merely because it is a behavioral-control region is not enough. I would place exact state machines wherever verified circuits embed most efficiently and reserve native central-complex computations for operations whose semantics genuinely match navigation mathematics.

### 4.5 Use the central complex for heading and coordinate transforms

For player position \((p_x,p_y)\), heading \(\theta\), and world point \((X,Y)\), a world-to-camera transform can be written

\[
x_c=
\cos\theta(X-p_x)+\sin\theta(Y-p_y),
\]

\[
y_c=
-\sin\theta(X-p_x)+\cos\theta(Y-p_y).
\]

Here \(x_c\) is forward depth and \(y_c\) is lateral displacement.

A heading population and vector-transform circuit are plausible neural accelerators for this operation because related coordinate transformations are experimentally supported in fly navigation circuits. The proposed camera operation still requires its own calibration and error characterization.

For correctness, maintain exact or conservatively bounded versions of quantities used for collision, occlusion ordering, and address generation. Approximate geometry can be useful while uncertainty remains below a declared pixel or decision margin.

### 4.6 The optic lobes are not a renderer waiting to be assigned a framebuffer

Visual analysis and graphics generation are different computations. Connectome-constrained fly-visual-system models provide evidence for predicting visual neural responses, not for native texture mapping or scene rasterization.

The opportunity is architectural: structured visual circuitry may offer useful substrates for repeated local computations. A FlyOS renderer would still have to synthesize and verify span generation, comparison, texture addressing, sampling, and pixel output.

Therefore, the claim should be **"rendering kernels embedded in selected optic-lobe circuitry,"** not "the optic lobe naturally renders Doom."

### 4.7 Lower the renderer into geometry and span-processing tasks

The original renderer includes front-to-back BSP traversal, bounding-box visibility checks, sector handling, and sprite collection. Those operations supply a decomposition for a Doom-derived renderer.

A proposed pipeline is:

| Stage | Neural work | Output |
|---|---|---|
| Visibility | Traverse bounded BSP/sector structures; reject invisible regions | Candidate geometry |
| Transform | Rotate and translate into camera coordinates | Camera-space vertices |
| Clip/project | Near-plane clipping and perspective division | Screen-space edges |
| Span generation | Determine covered columns or scanline intervals | Span records |
| Texture/shading | Address, fetch, interpolate, apply palette/light rules | Pixel values |
| Composition | Resolve visibility and write output buffers | Completed frame or tiles |

Perspective projection includes

\[
u=u_0+f\frac{y_c}{x_c},
\]

\[
v=v_0-f\frac{Z-p_z}{x_c},
\]

with explicit clipping for \(x_c<z_{\mathrm{near}}\).

A reciprocal unit could use an initial lookup followed by

\[
r_{n+1}=r_n(2-zr_n).
\]

Its table size, iteration count, fixed-point width, and rounding determine accuracy. An approximate result should not be advertised as bit-exact without a proven error bound or corrective path.

A wall-span worker might hold

```text
screen column
top and bottom pixel
texture identifier
horizontal texture coordinate
initial vertical texture coordinate
vertical increment
depth/visibility information
lighting state
```

It then generates pixels using neural arithmetic and neural texture access. Floor/ceiling spans and sprites require corresponding kernels.

### 4.8 Partition rendering where communication is amortized

Distribute columns, strips, or tiles—not individual arithmetic operations.

Each worker should receive a compact camera description and committed scene version, then reuse local geometry and replicated texture blocks for many pixels. Asset replication consumes memory but can avoid repeatedly shipping the same texels between brains.

The final display interface may receive neural pixel records such as

```text
(frame, x, y, color)
```

or a neural framebuffer stream. The external display may translate that finished output into light. It must not receive triangles or wall endpoints and then calculate the pixels itself.

For 160 × 100 pixels at four bits per pixel:

\[
160\times100\times4=64{,}000\text{ bits/frame}.
\]

At a hypothetical 10 frames/s, the pixel payload is **0.64 Mbit/s**. That does not establish achievable frame rate, but it suggests that final display bandwidth can be much smaller than internal neural computation and communication costs.

### 4.9 Entirely neural compilation requires a self-hosted path

There are two distinct targets:

**Cross-compiled neural execution:** a conventional compiler produces the neural executable; neural circuits execute the game.

**Self-hosted neural compilation and execution:** neural circuits read source, tokenize and parse it, maintain symbol tables, emit a program, and then run it.

The practical route to the second target is a **small neural virtual machine** with a compiler for a restricted language. The compiler emits bytecode into neural RAM; a resident neural interpreter executes that bytecode and invokes pre-existing arithmetic or rendering kernels.

This avoids requiring arbitrary physical rewiring whenever a program is compiled.

An initial bootstrap compiler can be loaded as firmware. A stronger demonstration then has it compile new source, and eventually its own source, within the neural substrate. Compiling a restricted Doom implementation this way is conceptually cleaner than attempting to run an entire modern compiler toolchain on the first FlyOS machine.

---

## 5. Quantitative constraints that determine feasibility

The following calculations are **illustrative architecture budgets**, not measured performance.

### 5.1 Nominal neurons and synapses are not usable RAM capacity

For 1,000 nodes at the proposed scale:

\[
N_{\mathrm{neurons}}=166.7\text{ million},
\]

\[
N_{\mathrm{synapses}}\approx125\text{ billion}.
\]

Suppose one robust working-memory bit consumes an average of 16 neurons, excluding address decoding and replication.

Then 1 MiB requires

\[
2^{20}\times8\times16
=134{,}217{,}728\text{ neurons},
\]

about **81% of the cluster**.

Alternatively, allocating 25% of neurons to that memory design yields

\[
\frac{166.7\times10^6\times0.25}{16\times8}
\approx318\text{ KiB},
\]

or roughly **106 KiB after three-way replication**, before further overhead.

The assumed 16-neuron cost is unvalidated and could be substantially wrong. The point is that **reliable, addressable memory must be measured as an implemented circuit**, not inferred from synapse count. Synaptic ROM or different codes may have very different density.

### 5.2 Biological-depth latency does not disappear with parallelism

If a dependency chain contains \(D\) stages averaging latency \(\ell\),

\[
L_{\mathrm{critical}}\gtrsim D\ell.
\]

For an illustrative \(D=100\) and \(\ell=2\) ms:

\[
L_{\mathrm{critical}}\gtrsim200\text{ ms}.
\]

A thousand nodes can improve throughput across independent work, but cannot make that same sequential dependency finish in 1 ms.

This favors precomputed constants, spatial kernels, fewer serial memory round trips, and substantial work per interbrain message. Pipelining helps independent frames or tiles; it does not eliminate causal dependence between successive authoritative game states.

### 5.3 Simulating the substrate is itself a large computing problem

Storing 125 billion synaptic contacts at an illustrative 16 bytes each requires about **2 TB**, before neuron states, queues, and other metadata.

A naïve 0.1 ms timestep across 166.7 million neurons entails

\[
1.667\times10^{12}
\]

neuron updates per simulated second.

Assuming a uniform 20 Hz presynaptic rate for budgeting gives approximately

\[
125\times10^9\times20
=2.5\times10^{12}
\]

synaptic-contact deliveries per simulated second.

Aggregation, event-driven execution, inactive regions, and specialized hardware can change those costs dramatically. Published fly-brain neuromorphic work already distinguishes anatomical synaptic contacts from aggregated computational connections. Those reductions must preserve the chosen model rather than merely delete inconvenient biology.

### 5.4 Reliability must scale to program length

For \(M\) protected operations with individual failure probabilities \(p_i\),

\[
P(\text{any failure})\leq\sum_{i=1}^{M}p_i.
\]

With a common bound \(p\), keeping that upper bound below 1% over \(10^8\) operations requires

\[
p\leq10^{-10}.
\]

This is not a claim that every local spike must meet that rate. It is a requirement on the final protected architectural operations.

A visually convincing neuron responding correctly in a handful of trials is far from a reliable computer component. Error detection, restoration, redundancy, and recoverable transactions are central architecture features.

---

## 6. A development program that tests the actual proposition

| Stage | Required demonstration | What it establishes |
|---|---|---|
| **Circuit library** | Cascaded gates, adders, registers, and converters under timing and parameter perturbations | Reliable computation primitives |
| **Neural control machine** | Loops, branches, memory accesses, calls, and safe-point interrupts | Programmable execution rather than a fixed response network |
| **Two-node system** | Exact messaging, retiming, backpressure, duplicate rejection, recovery | A real neural interconnect contract |
| **World-update kernel** | Tick-by-tick agreement with the reduced engine's reference state | Game logic executes neurally |
| **Neural renderer** | Correct pixels from neural geometry, assets, and texture lookup | Graphics are not being supplied by the host |
| **Cluster execution** | Interactive input, sustained play, node-failure recovery | Distributed FlyOS operation |
| **Self-hosted compiler** | New source compiled into executable neural bytecode | Compilation as well as execution is neural |
| **Native-topology assessment** | Exact mapping manifests and comparisons with matched alternative graphs | How much the fly connectome contributes |

The benchmark should report wall-clock speed separately from simulated or logical time, alongside neuron use, memory density, network traffic, error rate, recovery frequency, and all external computation.

To establish that the system is executing a program rather than reproducing a learned trajectory, use new input sequences and newly compiled levels after the primitive library is fixed. To establish that fly wiring matters, compare the native-constrained mapping with matched shuffled or freely synthesized networks.

## Architectural conclusion

**FlyOS should preserve biological mechanisms where they offer a computational advantage and impose explicit digital contracts where a program requires exactness.**

Its most defensible organization is:

- A verified spike-coded control and memory substrate.
- Population-coded navigation and vector accelerators.
- Sparse associative indexing, not "Bloom-filter RAM."
- Coarse-grained, epoch-aware interbrain messaging.
- A Doom-derived dataflow pipeline whose state updates and pixels are both produced neurally.

The decisive engineering challenge is **embedding enough cascadable logic, reliable memory, and causally correct communication into the permitted connectome dynamics**. Neither reading C nor recognizing a Doom screen is the hard part. The credible endpoint is a versioned neuron-level executable whose intermediate states, game decisions, and final pixels can all be traced to neural computation—with no external processor filling in the missing semantics.
