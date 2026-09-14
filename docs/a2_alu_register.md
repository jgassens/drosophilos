# A2 step 6: ALU and word register with staged commit

Follows the A2 opening (`a2_liveness.md`): channel contract frozen, adder composition
campaign closed. This note records the first two datapath blocks of the machine, the
accumulator that closes the loop between them, what the perturbation campaigns exposed, and
what was changed in the library as a result.

## 1. What was built

### ALU (`lib/alu.py`, 4-bit: 1,191 neurons)

| | |
|---|---|
| operand word (producer P) | `A[4] | B[4] | U[5] one-hot unit | SUB` = 14 dual-rail bits |
| result word (consumer Q) | `R[4] | C | Z | V` = 7 dual-rail bits, with completion tree and fault gates |
| units | ADDER (A + bx + SUB with bx = B xor SUB: ADD when SUB=0, A−B when SUB=1), AND, OR, XOR, PASSB (R = bx: MOV, or NOT-B with SUB=1) |
| flags | C carry-out (for SUB: 1 = no borrow), Z result is zero, V signed overflow = C xor carry into the top stage; C = V = 0 for the logic units |
| select | every unit computes on every transaction; per result rail one veto relay per unit (driven by that unit's rail, vetoed by the unit's *deselect* rail) ignites one result latch |
| ISA mapping | ADD/SUB = `ADD.WRAP`/`SUB.WRAP` with OVF = V; unsigned compare from SUB's C and Z, signed from Z, N (top bit) and V. Checked against `isa/semantics.py` on all 256 4-bit pairs |

Timing structure. B enters every unit through bx = B xor SUB, a rate-mode stage (~70 ms),
so bx is the *later* operand against A and the select rails by ≥ 50 ms. Everything that
pairs bx with A or with a select rail is an ordered gate built from veto relays (§2.4):
the three logic units, the adder's first XOR (x = A xor bx), the mux and the flag gating.
The adder's sum (x xor carry), the carry majority, bx itself and the zero tree have inputs
with no fixed order and stay rate-mode. Per bit: 6 rate-mode gates (bx) + 16 neurons
(x) + 8 gates (sum, carry) + 40 neurons (three logic units) + 34 (mux); flags and Z add ~60.

### Word register with staged commit (`lib/staged.py`, 4-bit: 321 neurons)

```
producer P --DATA--> stage S (completion W_S, fault latch F) --COMMIT--> master M (completion W_M)
```

| phase | mechanism |
|---|---|
| stage | S is an ordinary channel consumer: DATA through edge relays, bit-valid ORs, completion tree, fault gates, F. ACCEPT (W_S) clears P as usual. What differs: P's CLEARED is **not** wired to S's reset. S holds until commit or discard |
| COMMIT | a control token (one ignition pulse) into a COMMIT latch that lives in S's reset domain. grant C = AND(COMMIT, W_S), latched |
| commit sequence | C → M's edge-detected reset trigger → M reset train (4 × 0.75×) → M.ready (15 hops) → COPY latch → per-rail veto relay (COPY's rise ignites M rail r unless S holds rail 1−r) → M's bit-valid / tree → W_M |
| commit done | W_M's first spike, through an edge relay as a **single pulse**, into S's reset trigger: S, COMMIT, C, COPY and the copy gates clear; S.ready is the READY to the upstream |
| discard | F (both rails on a staged bit) or the producer's TIMEOUT also trigger S's reset; F holds W_S and C down so a faulty word is never granted. M is untouched |

Why W_M must be a pulse into S's trigger: W_M is a level that holds for as long as M is
valid (almost always). A train into an edge-detected trigger holds that trigger down for the
whole train, which would have blocked the F and TIMEOUT discards.

Measured (clean reference simulator, 4 bits, 321 neurons): ACCEPT 150 ms, grant → W_M
247 ms (of which 80 ms is M's READY chain letting its latches recover from the reset
train), READY 93 ms after commit-done; cycle 530 ms with an immediate COMMIT.

Tests (`tests/test_register.py`): early COMMIT (before completion) waits and is honoured;
late COMMIT (600 ms, 2 s) is honoured; a duplicate COMMIT inside a transaction applies once
(exactly one M reset per staged word); a corrupted staged word is discarded with F, never
granted, M unchanged, READY issued and the next word commits; a severed data path times out,
S is discarded, M unchanged; a COMMIT issued before the data applies to the next word
(documented protocol behaviour, not a feature).

### Accumulator (`build_accumulator`, 4-bit: 1,410 neurons)

P carries (B, U, SUB); operand A is the master's low four bits, entering through the
operand gate of §2.1; the ALU's stage is the register's stage; M holds R, C, Z, V. The host
loads (B, op) and one COMMIT per instruction and reads M; the value that feeds back as A is
never touched by the host. Seven-instruction and random programs match `alu_reference` on
every committed word and flag (clean sim: ACCEPT 300–480 ms by op, commit 293 ms,
720–910 ms per instruction).

## 2. What the campaigns exposed

Five separate mechanisms, each found from failure anatomy on a single perturbed node and
fixed on measurement. None was visible in the clean reference runs. The failure hunting
below ran at a harsher mix than the record's mix B (weights 5 % instead of 4 %, threshold
and bias ±0.25 mV instead of ±0.2; called B+ here); the rates quoted in this section are
at B+ and only order the mechanisms, the record numbers are the mix-B campaigns of §3.

### 2.1 A level is not a token (accumulator: 17 % of instructions hung)

Every hang was the same: a result bit whose "eager" OR gate (`1 OR x = 1`, fires on one
input) was driven by the master's rail alone. That gate restarts ~25 ms after the stage's
reset train ends, because the master holds continuously. Its ignition relay is a
once-per-activation relay: it re-arms only after ~50 ms of source silence, and a source that
restarts within 25 ms re-arms the relay's own inhibition before the relay can fire. The
latch never re-ignites and the next transaction waits for a bit that never comes. The ALU
channel cannot show this: its producer is cleared at ACCEPT, before the consumer resets.

Rule adopted: **the master is a level; the ALU consumes tokens.** Each master rail enters the
ALU through an operand gate driven by ACTIVE, an OR-latch over the unit-select rails that
lives in P's reset domain: it exists only while the producer holds a word, so every ALU input
is silent from ACCEPT until the next load. (First built as rate-mode ANDs, +55 ms; now veto
relays, §2.4: ACTIVE's rise ignites A rail r unless the master holds rail 1−r, ~10 ms.)
After this fix: hangs 17 % → 3 %, and the residual had other causes.

### 2.2 Edge relays re-fire on slow sources (ALU: ~1 % refused words; likely the adder's residual 3 × 10⁻⁴)

The once-per-activation relay holds itself off with feed-forward inhibition from its source
(one inhibitory spike per source spike, −2.2× loop). That holds for a *fast* source: a latch
train at 213 Hz keeps the inhibition saturated. A rate-mode gate fires at 25–40 Hz; the
inhibition from one gate spike peaks near 22 mV and has decayed to ~5 mV by the next gate
spike, and the relay's 12.6 mV pulse then crosses threshold again. Measured: an operand
latch's relay re-fired every 43 ms for a whole transaction. Every re-fire is an extra
1.8×-need ignition pulse into a running latch, so every gate-fed latch in the library ran
~10 % fast — exactly the regime in which one-input rate-mode ANDs at 0.65 leak (the A2
composition campaign had measured the window with single-shot ignition). The ALU shows it
first because it has the most AND stages with long one-input exposure (the XOR unit's
partial-product ANDs see one input for the entire transaction; the mux ANDs see the
unit-select rail for ~500 ms).

Fix: the ignited latch's own train holds the relay down (`hold_from` in `add_edge_relay`,
used by `_ignite_from`). First wired into the relay's existing 2.2× inhibitor, which
produced §2.5; now a separate light interneuron per gate-latch (1 neuron, −0.25× loop per
latch spike: ~17 mV sustained, blocks the 12.6 mV re-fire with ~11 mV to spare). A held
latch keeps that interneuron firing at 213 Hz, which is where the ~+25 % spikes per
transaction come from. Latencies unchanged.

### 2.3 Classification: a refused word is not a hang

The campaign classifier promoted a transaction to `fault` only if it had completed. A
FAULT-ACCEPT (fault gate fires, the completion latch is held down, the producer is cleared,
the four phases run, no word is consumed) was filed as `no_accept`, i.e. as a hang the
machine did not notice. It is the opposite: it is the machine refusing the word. Fixed
(no completion + fault-gate spikes → `fault`, neural). The 24 "missing completions" of the
adder campaign in `a2_liveness.md` §4 were measured under the old rule and are most likely
FAULT-ACCEPTs; they are re-labelled there as "no_accept (old rule; includes refused words)".

### 2.4 The veto relay: an AND with no exposure window (mux one-input leaks; master faults)

Even with single-shot ignition, a two-input rate-mode AND at 0.65 sits at 65 % of threshold
whenever one input is live and the other is not, and leaks under the ±10 % latch-rate spread
across nodes plus weight noise. In the ALU the mux ANDs see the select rail for the whole
transaction and the opposite rail of the operand never arrives (measured: a MOV refused
after 150 ms because `mux1r1u4` fired on the select rail alone). In the register the copy
ANDs for the *empty* stage rails sat on COPY alone for ~100 ms and, in a node whose COPY
latch ran fast, three of four fired: the master ended with both rails on three bits
(2 master faults per 5,000 commits).

The replacement uses the two facts dual-rail gives for free: NOT is the other rail, and one
operand of most ANDs is known to arrive first. A **veto relay** (`celement.py::add_veto_relay`)
is the library's one-shot relay driven by the *later* operand's rail, with an inhibitory
interneuron driven by the *earlier* operand's other rail (−0.5× loop per spike, ~33 mV
sustained against a 12.6 mV driver pulse). It computes `later AND NOT(veto)` at the
driver's rise and ignites its target latch once. There is nothing to leak: the relay never
integrates one input towards threshold. Cost 3 neurons (relay, its own inhibitor, veto)
against 5 for a latched rate-mode AND, and ~5 ms against ~35.

Stated assumption (bounded delay): a veto rail must be live ≥ 15 ms before the driver rises.
Where it is used and the margin by construction:

| gate | driver (later) | veto (earlier) | margin |
|---|---|---|---|
| operand gate A | ACTIVE (load + ~30 ms) | master rail (a level) | unbounded |
| logic units, x = A xor bx | bx (load + ~70 ms) | A rails (load, or load + ~40 ms in the accumulator) | ≥ 30 ms |
| result mux | unit output (≥ load + 75 ms) | deselect rail U_k.r0 (load ± 10 ms jitter) | ≥ 55 ms |
| flag gating | cout / V (≥ load + 100 ms) | U_adder.r0 | ≥ 80 ms |
| copy into master | COPY (commit + 90 ms) | stage rail (held since ACCEPT) | ≥ 300 ms |

Ordered forms of AND/OR/XOR (`Gates.and2_ordered` etc.) express `EARLY0 OR LATE0` as
`LATE0 OR (EARLY0 AND LATE1)`, so every relay is driven by the later operand and nothing
is eager any more: the pre-charge from a lone operand that §2.1 had to work around is gone
with it.

### 2.5 A relay's head start is ~5 ms; anything that hyperpolarises it must decay first

The first version of the §2.2 hold ran the hold through the relay's 2.2× inhibitor. That
holds the relay at ~146 mV below rest, which needs ~86 ms to decay to the ~2 mV a relay
can tolerate (its pulse crosses threshold ~1.8 ms after arrival and its own inhibition lands
at ~5.3 ms; a residual above ~2 mV moves the crossing past the inhibition and the relay
loses). A consumer reset is ~85 ms before the next load, so every relay whose target had
held in the previous transaction failed: the V=0 select relay after an ADD that set V=0,
the copy relays for rails the master already held. Measured on the clean simulator, not
only under perturbation. Hence the separate light hold interneuron and the 0.5× veto.
Recovery is now a stated timing budget: hold ~40 ms, veto ~55 ms, source inhibition
~86 ms, against gaps of ≥ 85, ≥ 155 and ≥ 170 ms respectively.

## 3. Campaigns with the fixed build (mix B; `docs/a2/`)

Mix B = weights log-normal 4 %, threshold and bias ±0.2 mV, stray 5 Hz × 150 quanta,
arrival jitter ≤ 10 ms, the same as every M1/A2 campaign. One campaign per block on the
final build (veto relays, light hold, fixed classifier); random operands and random
programs; COMMIT issued with the load.

| block | transactions | ok | wrong value / wrong master | refused (fault, timeout) | hangs / stale (harness) | non-ok observed / 95 % upper | ACCEPT p99 / max |
|---|---|---|---|---|---|---|---|
| 4-bit ALU, random (A, B, op) | 5,000 | 5,000 | 0 / – | 0 | 0 | 0 / 6.0 × 10⁻⁴ | 497 / 536 ms |
| 4-bit staged register, random words | 5,000 | 5,000 | 0 / 0 | 0 | 0 | 0 / 6.0 × 10⁻⁴ | 170 / 182 ms |
| 4-bit accumulator, random programs | 2,000 | 2,000 | 0 / 0 | 0 | 0 | 0 / 1.5 × 10⁻³ | 490 / 542 ms |
| 4-bit channel, relay-hold build (re-validation of the frozen contract) | 100,000 | 100,000 | 0 / – | 0 | 0 | 0 / 3.0 × 10⁻⁵ | 170 / 187 ms |

Silent-error upper limits are what a zero-failure count of this size can say: 6 × 10⁻⁴ and
1.5 × 10⁻³. These are composition checks, not bound-pushing runs (M1 review).

What the same harness measured on the way here, for scale (at the harsher B+ mix; probe
sizes in parentheses): accumulator 7/40 hangs before §2.1, 4/120 after it, 0/120 after
§2.2–2.5; ALU 6/500 refused before §2.2, 2/500 after it, 0/500 after §2.4; register (mix B,
5,000): 10 non-ok including 2 master faults before §2.4, 0 after.

### 3.1 Channel re-validation

The relay-hold fix (§2.2) changes every block that ignites a latch from a gate, including
the frozen 4-bit channel (its completion-tree relays were re-firing every ~43 ms too).
The frozen contract is therefore re-measured on the new build and its 10⁵ mix-B campaign
re-run with the same period and 55-hop watchdog; the freeze is re-issued on that result.
Result: 100,000 ok, 0 non-ok of any class (the frozen build's run had 1 stale-activity case), ACCEPT p99 170 ms, 201 neurons. The freeze is re-issued on this build (`docs/contracts/channel_4bit.yaml`, status line).

## 4. The ordered datapath: an adder with no rate-mode gate

§2.4 left the symmetric-arrival gates (B xor SUB, the adder's sum and carry, the zero tree)
in rate mode, each exposing one input for a whole transaction whenever the other rail never
comes. The way out is to *make* the arrival order deterministic with delay chains
(`celement.py::add_delay_chain`: relays at pulse drive, ~5.3 ms per hop; only the rise is
faithful, which is all a veto relay's driver needs). With the order fixed, every AND in the
datapath is a veto relay.

| signal | rises at | how |
|---|---|---|
| U, SUB, B | load (± 10 ms jitter) | producer rails |
| B^d | load + 32 ms | each B rail through 6 hops |
| bx = SUB xor B^d | ~36–46 ms | ordered XOR: SUB early, B^d late (margin ≥ 24 ms) |
| ACTIVE | ~10 ms | OR-latch over the unit-select rails, in the producer's reset domain |
| A^d | ~70 ms | operand gate: ACTIVE delayed 11 hops drives one veto relay per rail, vetoed by the source's other rail (works for a master's level or a producer rail alike) |
| x_i = bx_i xor A^d_i; generate; kill | A^d + 4 | ordered: bx early, A^d late (margin ≥ 24 ms) |
| carry-in to bit i ≥ 1 | c_{i−1} + 27 ms | each carry rail through 5 hops |
| c_i | max(A^d + 2, carry-in + 2) | generate / kill relays (driver A^d, veto bx) + propagate relays (driver delayed carry-in, veto x_i.0) |
| sum_i | carry-in + 2 | ordered XOR: x_i early, delayed carry late (margin ≥ 19 ms) |
| V | delayed top carry + 2 | p_top = 1 → V = 0; generate → V = not c; kill → V = c, all as relays on the delayed carry with A^d, bx vetoes |
| Z | R-completion + 20 ms | on the consumer: Z0 from any R.1 (relays); Z1 = the consumer tree's node over the R bits, delayed 3 hops, vetoed by every R.1 (§4.2) |

The carry chain is therefore 29 ms per stage (27 ms of delay, 2 ms of relay), against ~35 ms
for a rate-mode majority, and the sum of the top bit arrives 29 ms after the top carry
instead of 35. The whole 4-bit adder is 594 neurons as a channel (the rate-mode one was 603)
and the ALU 1,150 (was 1,191). Clean-model ACCEPT: ordered adder 330–384 ms; ALU 241 ms
(MOV) – 524 ms (SUB with a full propagate chain and Z = 1); accumulator 663–827 ms per
instruction. The completion tree (three levels of rate-mode ANDs on the bit-valid latches,
~105 ms) is now the largest single cost in the ALU; its exposure is bounded because every
bit always becomes valid.

What is still rate-mode anywhere: the completion trees, the fault gates (0.55 per rail, one
rail always live, measured clean over 10⁶), ACTIVE's OR (one input suffices), and the
register's grant AND(COMMIT, W_S). The grant's exposure is the FSM's latency (COMMIT alone
until completion, or W_S alone until COMMIT); a control machine that issues COMMIT in response
to completion bounds it to its own reaction time.

Ordering assumptions now stated in the contracts: A^d rises ≥ 24 ms after B^d; each delayed
carry-in rises ≥ 19 ms after its stage's x; the mux's select rails are valid ≥ 26 ms before
any unit output; the R rails are valid ≥ 50 ms before Z's driver.

### 4.2 No result bit is reliably last: Z needs a completion, not a delay

The first version drove Z from the top result bit delayed 32 ms, on the assumption that
in a ripple adder the top sum arrives last. It does not: a lower sum can wait on a long
propagate chain while the top carry was decided early by a kill or a generate. At mix B+
every ALU refusal (13 in 500) was a Z double rail, and the accumulator's one refusal was the
same (bit 3's sum at 164 ms, bit 2's at 204 ms, Z1 fired at 196 ms). The spread between the
earliest and the latest result bit grows linearly with the width, so no fixed delay is the
answer. "Every result bit is valid" is a completion, and the consumer's tree already has that
node (the subtree over bits [0, n)): its train, delayed 3 hops, drives Z1's relay with every
R rail 1 as a veto, straight into the consumer's Z rails; Z0 is an OR of the R rail-1
latches. Z therefore costs one tree level of latency (Z=1 words complete ~110 ms after Z=0
words) and no neurons beyond the relays.

### 4.1 Campaigns (mix B)

| block | transactions | ok | wrong values | refused / hung | non-ok observed / 95 % upper | ACCEPT p99 / max |
|---|---|---|---|---|---|---|
| 4-bit ripple adder, ordered, random operands and carry-in | 30,000 | 30,000 | 0 | 0 | 0 / 1.0 × 10⁻⁴ | 397 / 420 ms |
| 4-bit ALU, random (A, B, op) | 5,000 | 5,000 | 0 | 0 | 0 / 6.0 × 10⁻⁴ | 516 / 555 ms |
| 4-bit accumulator, random programs | 2,000 | 2,000 | 0 (stage and master) | 0 | 0 / 1.5 × 10⁻³ | 509 / 535 ms |

For comparison, the rate-mode adder's 10⁵ campaign at the same mix (`a2_liveness.md` §4)
observed 6.4 × 10⁻⁴ non-ok (30 faults, 1 timeout, 24 refused-or-hung, 9 stale), all of them
refusals or hangs and none a wrong sum. Probes at the harsher B+ mix on the final ordered
build: ALU 500/500, accumulator 120/120 clean (the delay-based Z had 13/500 and 1/120).

## 5. Costs and the accumulator's timeline

| block | neurons | ACCEPT (clean) | cycle |
|---|---|---|---|
| 4-bit channel (frozen) | 194 | 155 ms | 334 ms |
| 4-bit ripple adder | 558 | 366 ms | 544 ms |
| 4-bit ripple adder, ordered | 594 | 330–384 ms | 510–560 ms |
| 4-bit ALU | 1,150 | 241 ms (MOV) – 524 ms (ADD/SUB, Z = 1) | 420–700 ms |
| 4-bit staged register | 321 | 150 ms | 530 ms incl. commit |
| 4-bit accumulator | 1,323 | 241–405 ms | 660–830 ms per instruction |

An instruction costs 0.67–0.83 s of neural time. The commit alone is 293 ms, of which
80 ms is the master's READY chain and ~70 ms is the master's completion tree. Both are
protocol safety margins, not computation; overlapping the next operand load with the commit
(a double-buffered stage) is the obvious throughput step and is deferred to the control
machine, which decides when a register may be read.

## 6. Carried forward

- The one-input AND margin was the library's weakest point; after §4 no rate-mode AND with
  a whole-transaction exposure remains in the datapath. The completion trees and the grant
  are the last rate-mode ANDs, with bounded exposure.
- Register-file semantics (many masters, read ports gated by the control machine) and the
  reset-quiescence rule (a domain's inputs must be silent for ≥ 50 ms after its reset before
  they restart) are now design rules for the RAM and control FSM.
