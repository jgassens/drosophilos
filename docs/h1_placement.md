# Stage H1 — motif placement of the 4-bit adder on MCNS (measured 2026-09-15)

Tool: `drosophilos/connectome/embed_netlist.py` (`place_netlist`). Target: the 4-bit ordered
adder netlist, `build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100)`: 614 designed
neurons, 1,146 designed edges. Substrate: MCNS v1.0 (166,700 neurons, 25.6 M edges), Profile 2
rules from H0: every designed synapse (src → dst, q quanta) must be carried by an anatomical
edge from the real neuron chosen for src to the one chosen for dst, with
count × 16 × k_max ≥ |q| (k_max = 4, so a 3,621-quanta loop edge needs ≥ 57 synapses, an ignition
edge ≥ 73, a relay inhibitor ≥ 125, a reset edge ≥ 43), acetylcholine carrying excitation, GABA or
glutamate carrying inhibition, and one sign per designed neuron. Every anatomical edge among the
chosen neurons that is not a designed edge is a parasitic edge to zero.

The number to beat was 10 % (the greedy neuron-by-neuron tool; re-measured here at 102 edges,
8.9 %, with the same audit). The goal was more than 50 %.

## Result

*Audit correction (commit `35c8b84`, after this section was measured):* the netlist has 1,146
synapse entries but 1,144 distinct (source, target) pairs — two reset edges onto one completion
latch are entered twice — and the audit now counts distinct pairs with their quanta summed and
checks the host's transmitter itself. On the same mapping that gives **701 of 1,144 (61.3 %)**:
one of the duplicated pairs needs 85 synapses when summed and its host edge carries 47. The
tables below keep the per-entry counts they were measured with (at most 3 edges high).


| | greedy (first tool) | motif search, best of 8 restarts |
|---|---|---|
| designed edges carried | 102 / 1,146 (8.9 %) | **704 / 1,146 (61.4 %)** |
| designed neurons placed | 219 / 614 | 612 / 614 |
| parasitic anatomical edges to zero | 950 | 17,844 (5,305 of ≥ 24 synapses, 3,108 of ≥ 57) |
| search time | 1.2 s | 65 s for 8 restarts (4–13 s each: setup 0.7 s, descent 4–13 s, repair 1–15 s); the 600 s limit was not needed |

The eight restarts landed between 678 and 704 edges (59–61 %); the search is stable to about
±2 %. The construction phase (backtracking with forward checking) carries 650–686 edges; the
repair phase (each motif taken out and put back with both sides of its neighbourhood known)
adds 18–45.

**Split by edge class.** Of the 1,146 edges, 884 are *hard* (both endpoints have at most 20
designed partners), 258 are the many-edges of three *hubs*, and 4 are impossible under Dale's
law (`Q.faultL.u` both excites its loop partner and inhibits four gates; it is placed with its
majority sign and its four inhibitory edges are lost by construction).

| edge class | edges | carried | fraction |
|---|---|---|---|
| hard (motif-internal and motif-to-motif) | 884 | 697 | 78.8 % |
| hub fan-out (`Q.reset_inh` 115, `P.wd.cancel_inh` 102, `P.reset_inh` 41) | 258 | 7 | 2.7 % |
| impossible (mixed sign) | 4 | 0 | — |

So the placement is a 79 % placement of the circuit's logic and a 3 % placement of its three
broadcast neurons. The two are separate problems, and the second is the one the connectome
cannot solve with one neuron per designed neuron (below).

**Motif outcomes** (complete = every node placed and every one of the motif's own edges carried):

| motif | count | complete | partial |
|---|---|---|---|
| latch (mutual pair, loop ≥ 57 both ways) | 72 | 58 | 14 (a member placed on a neuron whose loop partner was taken) |
| edge relay (E with its inhibitor I ≥ 125) | 79 | 49 | 30 |
| delay chain (paths of 5–16 hops, one of 100) | 18 | 17 | 1 |
| single (vetoes, gates, hold and reset interneurons, `start`) | 89 | 89 | 0 |
| hub | 3 | 3 (all inputs carried) | 0 (fan-out lost, see below) |

The full relay triple (S → E ≥ 73, S → I ≥ 57, I → E ≥ 125) is carried for 32 of 79 relays;
17 more carry both source edges and lack the inhibitor's edge, 9 lack S → I, 11 lack all but S → I.
The 100-hop watchdog chain is carried whole (99 / 99 hop → hop edges; the strongly connected core
of the ≥ 57-synapse cholinergic graph has 3,197 neurons and depth-first search finds a 100-path in
it in well under a second). The B-rail and carry-in delay chains (5–6 hops) are complete except
for one.

**Neurons left unplaced, by role:** 2 of 614 — two relay inhibitors (`Q.valid2.ign.edge_inh`,
`add.fa3.s.y0.b.edge_inh`) for which no free inhibitory neuron carried any of their edges. Every
latch member (72 u, 72 v), every relay neuron (77), every delay hop (89 d, 100 hop, 30
ready_delay), every veto (56), gate (9 and, 6 or) and reset neuron is placed.

**Where the missing edges are** (edges missing / edges of that class):

| class | missing | note |
|---|---|---|
| `cancel_inh → hop` | 100 / 100 | the watchdog's cancel neuron inhibiting all 100 hops at ≥ 85 |
| `reset_inh → u`, `→ v`, `→ or`, `→ and` | 151 / 156 | the two reset neurons inhibiting 73 latch pairs and 10 gates at ≥ 43 |
| `edge → u` (relay igniting its target latch) | 36 / 76 | |
| `edge_inh → edge` (the relay's inhibitor) | 29 / 77 | |
| `u → start` | 16 / 18 | the watchdog's OR over all 18 producer rails at ≥ 12 (an input hub) |
| `u → veto`, `veto → edge` | 23 / 88 | |
| `u → edge`, `u → edge_inh`, `d → edge`, `d → edge_inh` | 29 / 133 | relay source edges |
| `u → v`, `v → u` | 16 / 145 | latch loops |
| everything else | 42 / 208 | |

**Where the circuit landed.** Hosts by superclass: 288 central-brain intrinsic, 149 nerve-cord
intrinsic, 93 descending, 45 ascending, 12 sensory, 11 visual projection or centrifugal, 1 optic-lobe
intrinsic: the optic lobe is all but absent, as §5 of `capacity_doom.md` predicted. The most-used cell types are antennal-lobe
local neurons (lLN1_bc, lLN2X11, lLN2X12 for latch members; lLN2T_a, lLN2X05 for relays; lLN2F_b for
relay inhibitors), gnathal-ganglion neurons (GNG014, GNG457, GNG466 for delay hops and vetoes, as in
H0's hand-designed circuit) and nerve-cord interneurons (IN07B006, IN19B003 for the watchdog hops).
The reset neurons landed on l2LN21 and GNG391 (7 of their 8 own inputs carried, 5 of their 156
fan-out edges), the cancel neuron on GNG524 (2 of 102).

## What the wiring lacks

1. **Broadcast neurons.** No single inhibitory neuron in MCNS reaches the 115 targets of
   `Q.reset_inh` where the search put them; the best host carries 5 of 115. Splitting the hub into
   several real neurons (greedy set cover of the placed targets, `hub_split_analysis`) needs 38
   inhibitory neurons to reach 85 of the 115 targets at ≥ 43 synapses — 30 targets are reachable by
   no free inhibitory neuron at all — and **none of the 38 receives even one of the reset neuron's
   four designed inputs** (trigger and three relays at ≥ 57): the tree would need another level of
   relays, each of which is again a fan-out problem. `P.reset_inh`: 22 copies for 40 of 41 targets,
   one copy reached by one input. `P.wd.cancel_inh`: 68 copies for 95 of 102 targets (a 100-hop
   chain cannot lie under one inhibitory neuron: the largest inhibitory fan-out at ≥ 85 synapses in
   the whole connectome is 106 edges, PVLP011), none reached by either input. The input hub
   `P.wd.start` (18 producer rails at ≥ 12) carries 2 of 18.

   The reset hub is *not* impossible in principle. The antennal-lobe local neuron il3LN6 inhibits
   both members of 107 latch-capable pairs at ≥ 43 synapses (lLN2F_b: 105), and — measured for this
   report — **all 107 of those pairs can source a complete relay, all 107 can be ignited by a relay,
   and all 107 by a relay whose source is another pair under the same hub**. The antennal lobe is a
   uniform mesh of the latch-and-relay motif with a global inhibitor already in place. Anchoring
   `Q.reset_inh` there (`hard_hubs=("Q.reset_inh",)`: its 115 edges enforced instead of scored)
   carries 64 of its 115 edges and drops the total to 650 (56.7 %): the adder's latches move into
   the antennal lobe and lose their connections to the veto relays, delay chains and the `actd`
   fan-out, which live in the gnathal ganglion. The register and the arithmetic want different
   neighbourhoods, and the connectome has no strong path family joining them at these counts.

2. **The relay fan-out node.** `add.actd.d10`, the end of the 11-hop ACTIVE-delay chain, drives
   8 veto relays (16 edges of ≥ 73 / ≥ 57 to 8 distinct (E, I) pairs). Only 12 neurons in MCNS source
   ≥ 8 relay triples (AVLP374 12, AVLP400 10, PS124, LT1d, GNG014 9), and 5 of them survive the
   chain (a 10-edge walk into them) and arc-consistency filters. The best placement puts `d10` on
   PVLP137 and carries 10 of its 16 edges: all 8 relay neurons are driven and every one ignites its
   latch, but only 2 of the 8 inhibitors receive the ≥ 57 edge from `d10` (the other six carry their
   ≥ 125 edge onto the relay but not their input). Fourteen more designed sources need 3 relays each
   (234 real neurons qualify). This is the second-scarcest resource after the hubs.

3. **Relay inhibitors at ≥ 125 synapses.** 14,146 inhibitory edges of ≥ 125 synapses exist, but a
   relay needs one landing on an E that its own source also drives: 2,249 (S, E) pairs with an
   inhibitor in the whole brain, 1,228 distinct sources, 733 distinct relays. Placing 79 relays
   whose sources are latch members already fixed by other constraints carries the full triple for
   32 and the inhibitor edge for 49.

4. **Distinctness.** The strong-edge subgraph is small (3,197 neurons in the ≥ 57 cholinergic
   core; 1,223 latch-capable pairs, 2,446 members) and the same few neurons are the candidates for
   many designed roles; once one is taken, its neighbours lose it. Arc consistency without the
   one-neuron-per-role rule leaves every domain non-empty, so the losses above are conflicts over
   shared neurons, not absent motifs — except for the hubs.

## How the search works, and what did and did not help

Motifs are read from the netlist's structure (mutual excitatory pairs; an excitatory relay with
the inhibitory neurons that share its source and have it as their only output; maximal paths of
in-degree-one excitatory neurons; hubs with more than 20 partners; singles). Each designed neuron
gets a strict candidate set (transmitter, enough strong partners, membership in a real instance
of its motif — precomputed mutual-pair, relay-triple and k-step-walk masks — a source of k relays
restricted to neurons driving k real relays, then arc consistency over the hard edges) and a
loose one (the same without the context masks). The search is depth-first over motifs,
most-constrained first, assigning each motif as a unit — latch pairs from the real mutual pairs,
relays as real (E, I) pairs, chains as paths by depth-first search over ≥ 57-synapse cholinergic
edges with walk-reach pruning, hubs by coverage — ranked by a lookahead that asks whether every
neighbouring motif keeps a complete instance. Forward checking after each assignment; a candidate
that empties a neighbour's domain is rejected while another exists; a motif with no clean candidate
backtracks up to 8 times over 3 frames and then takes the least damaging candidate, or its loose
layer, or a best-fit placement that prunes no one else. A repair pass re-places each motif given
everything else, and restarts alternate hubs-first and hubs-last.

Measured on the way (each an A/B on the same seeds):

- Dynamic arc consistency beyond the direct neighbours (maintaining arc consistency) *lowered*
  coverage from ~630 to 413–555: it prunes candidates that would still carry most of their edges.
  Under a maximum-coverage objective, forward checking from placed neighbours is the right amount
  of propagation; stronger propagation is only useful as a static filter before the search.
- Placing a failed motif immediately by best fit and letting it prune its neighbours cascaded
  (`or` placed by best fit emptied both carry-in latches' domains): 567. Making best-fit
  placements soft anchors — rewarded, never enforced — restored 695.
- The context masks (a source of 8 relays needs 8 real relays) are necessary for ordering: without
  them the chain ending at `actd.d10` was placed before the relays it feeds and all eight failed.
  But a motif that cannot be completed in context must fall back to its loose layer, or the
  latch loops go too (612 → 687).
- All 48 candidate paths for a chain shared one start, and the path ran through the very neurons
  the chain's relays needed (GNG014's strong inputs are also its strong outputs). Spreading the
  candidates across the tight end's options and reserving small neighbour domains fixed the first
  motif of every descent.
- Candidate cap 24/48/96 and backtrack budgets 6/3 to 12/5 are within run-to-run noise (682–705
  before repair); the time is better spent on restarts and repair.
- A stronger reward for hub edges (×2, ×4) changed nothing: hub-adjacent latch pairs never
  satisfied the rest of their constraints, so the reward never broke a tie. Enforcing the hub
  instead (anchoring) moved the placement, at a net loss.

## Answer to the Stage H question

More than half of the adder's edges — 61 %, and 79 % of the edges between motifs — can be carried
by MCNS under the H0 rules with one real neuron per designed neuron, in about a minute. The
remainder is structural, not a search failure: the three broadcast neurons (22.5 % of all edges)
have no anatomical counterpart at these fan-outs and cannot be replaced by a tree of real neurons
that the designed trigger can drive; and the reset hub's natural home (the antennal lobe, with a
resident global inhibitor and 107 latch pairs each with relays) is not where the arithmetic's
scarce fan-out neurons live. What a Profile 2 image of the adder therefore needs from Profile 3
is small and specific: the hub fan-outs (reset and cancel, or a redesign with local resets), the
`actd.d10`-style fan-out, and about 30 relay inhibitor edges; the latches, delay chains, vetoes and
most relays are already in the fly's wiring. Not done here: hub splitting applied to the mapping
(reported only) and the whole-cell (~1,300-latch) target; the neural simulation of the placed circuit
follows in the next section.

Reproduce, from the repo root with the MCNS data present:

```python
from pathlib import Path
from drosophilos.sim.model import Params
from drosophilos.lib.adder import build_adder_channel
from drosophilos.connectome.mcns import load_mcns
from drosophilos.connectome.embed_netlist import place_netlist, hub_split_analysis, host_types
net = build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100).net
m = load_mcns()
pl = place_netlist(net, m, verbose=True, time_limit_s=600, restarts=8)   # 704 / 1146 (restart 5 of 8, seed 0)
pl.summary(net); pl.motifs; pl.missing_by_class; hub_split_analysis(net, m, pl); host_types(pl, net, m)
```

Tests: `tests/test_embed_netlist.py` — a single latch with its edge relay and inhibitor (built with
`protocol.latch.add_latch` / `add_edge_relay`) is found from the netlist's structure and placed
completely (6 / 6 edges, the planted neurons in their roles) on a synthetic connectome with decoys,
with a variant where the inhibitor edge is absent and the audit must say so; and on the real
connectome when the data is present.

## The placed adder, simulated

Tool: `drosophilos/connectome/embed_image.py` (`build_image`, `simulate_channel`, `run_h1_conditions`);
mapping used: `docs/h1_placement_mapping.json` (the best-of-8 placement above, re-run with
`place_netlist(net, m, restarts=8, time_limit_s=600, seed=0)`: 704 / 1,146 again, restart 5, 72 s).

**What the image is.** The circuit's 614 neurons keep the netlist's own indices, each hosted on
its real MCNS neuron (612 of them; the two unplaced relay inhibitors get synthetic neurons), so
the adder's own harness — the same injection, the same four-phase decode — runs on the image
unchanged. Of the 1,146 designed edges, 704 are *carried*: the anatomical edge between the two
hosts, of the right sign, rescaled to exactly the designed quanta by a factor k = |q| / (count × 16)
with k ≤ 4 (a parameter edit, recorded per edge; the histogram of k is 14 edges ≤ 0.5, 55 in
(0.5, 1], 202 in (1, 2], 206 in (2, 3], 227 in (3, 4]; the largest is 3.99, i.e. most carried edges
lean on the ×4 allowance). The other 442 are added as labelled *Profile 3* edges (434 with no
anatomical edge strong enough between the hosts, 4 with an unplaced endpoint, 4 of the wrong sign):
by class, `cancel_inh → hop` 100, `reset_inh → u/v/or/and` 151, `edge → u` 36, `edge_inh → edge` 29,
`u → start` 16, the rest 110 spread over 30 classes of the arithmetic's own wiring. The 17,845
anatomical edges among the hosts that are not carried designed edges (7.76 M quanta, 23 of them
under a designed edge that is too weak; one more than the audit's 17,844 because the netlist
gives the completion latch its reset edge twice) are parasitic: zeroed as documented zero-weight
edits, or kept at their anatomical quanta in condition E. The image's manifest lists the four
graphs, 704 rescalings, 17,845 zeroings, 442 added edges and 2 added neurons — a Profile 2 image
with 442 Profile 3 edges (39 % of the edges, 61 % of them the three broadcast neurons').

**What the simulation says.** Fifty random 4-bit additions (seed 0), each in a fresh simulator
(RefSim, dt 0.1 ms, up to 1.2 s of neural time per addition; a failing condition would otherwise
stop the harness at its first incomplete transaction); the conditions that compute were also run
chained, fifty words through one simulator, which tests the reset between words.

| condition | Profile 3 edges | parasitic | sums correct | how it fails | completions missing |
|---|---|---|---|---|---|
| A: every designed edge (missing ones added), parasitic zeroed | 442 | 17,845 zeroed | **50 / 50** (chained 50 / 50; ACCEPT 330–384 ms, cycle 508–562 ms — the netlist's own numbers) | — | 0 |
| B: carried edges only | 0 | zeroed | 0 / 50 | no ACCEPT: the producer loads, 132–145 arithmetic neurons run (~36 k spikes per addition, latches never reset), nothing reaches the output register | 50 |
| C: B + the three broadcast neurons' missing edges (`Q.reset_inh`, `P.reset_inh`, `P.wd.cancel_inh`) | 251 | zeroed | 0 / 50 | identical to B: the resets never trigger, so the hub edges are never used | 50 |
| D: C + the missing relay-inhibitor (`edge_inh → edge`, 29) and `actd.d10` fan-out (6) edges | 286 | zeroed | 0 / 50 | no ACCEPT; quieter (107–124 arithmetic neurons, ~26 k spikes: the relays are now held down) but the output register is still untouched | 50 |
| E: A with the parasitic edges kept at anatomical weights | 442 | 17,845 kept | 0 / 50 | ACCEPT after 15–32 ms (the parasitic edges drive the completion tree directly); 17 faults (both rails of an output bit lit: 2–8 of the 5 bits' 10 rails active), 27 watchdog timeouts, 6 incomplete decodes; 358 fault spikes | 0 (the fault path completes the four phases) |
| F (extra): A minus the three broadcast neurons' missing edges | 191 | zeroed | **50 / 50** fresh; chained 1 / 50 | the logic computes every first sum with the netlist's timing; without the reset and cancel fan-outs the registers are never cleared, so the second word never completes (and the watchdog fires 5 times after ACCEPT, uncancelled) | 0 fresh |

Condition A validates the mapping and the scaling: an image that carries 704 edges on real
neurons at rescaled anatomical weights and adds the rest computes every sum with exactly the
netlist's latencies. Conditions B–D say that the carried 61 % is not a working fraction of the
adder: the lost edges are not concentrated in one stage but scattered through the data path
(2 of the 10 output-relay ignition edges, 34 of the Q register's 155 internal edges — its
completion tree, valid latches and fault gates — 26 of the producer's 186, 115 of the
arithmetic's 499), and a four-phase channel with any of its relay or latch edges missing
stalls before ACCEPT rather than computing a wrong sum. Adding the hubs' 251 edges (C) or the
fan-out classes named in §What the wiring lacks (D) changes nothing because the failure is
upstream of them. Condition F is the informative split: the 191 missing edges of the logic are
what a fresh addition needs, and the 251 hub edges are what the *second* addition needs — the
hubs are the whole reset problem and nothing else. Condition E says the image is only a circuit
when the parasitic edges are silenced: kept at anatomical weight, the 17,845 anatomical edges
among the hosts (mean 27 synapses; 5,305 of ≥ 24) ignite output rails and the completion tree
within 15 ms of loading, and no addition survives. So the Profile 2 image of the adder is
Profile 2 in its neurons and in 61 % of its edges, needs 442 Profile 3 edges (191 for the logic,
251 for the resets) and 17,845 documented zeroings, and computes only with those zeroings applied.

Reproduce (about 8 minutes: five to six conditions × 50 fresh additions):

```python
from drosophilos.connectome.embed_image import build_image, load_mapping, run_h1_conditions
pl = load_mapping("docs/h1_placement_mapping.json", net, m)   # net, m as above
img = build_image(net, m, pl); img.counts; img.manifest
run_h1_conditions(ch, m, pl, width=4, n_cases=50, seed=0)      # ch = build_adder_channel(...)
```

Tests: `tests/test_embed_image.py` — the planted latch + relay + inhibitor of
`test_embed_netlist.py`, placed on its synthetic connectome and turned into an image (6 / 6 edges
carried at the designed quanta with their scales, parasitic edges zeroed, a knocked-out edge
reported as a labelled Profile 3 edge, an unplaced neuron as a synthetic one), simulated: one
source spike fires the relay once and ignites the latch, which holds for 400 ms; and, when the
MCNS data is present, the adder image in condition A on 5 chained additions, all correct.

## Coverage against the weight bound

Tool: `drosophilos/bench/h1_sweep.py`; data: `docs/h1_sweep.json`. Re-ran the placement and the
simulation of §Result and §The placed adder, simulated at four values of `k_max` (the bound on the
rescaling factor `count * 16 * k >= |q|`, k <= k_max), holding everything else fixed: the same
adder netlist, `place_netlist(net, m, policy, restarts=4, time_limit_s=300, seed=0)` (best of 4
restarts; the previous section used 8), and, per `k_max`, condition A (all missing edges added,
parasitic zeroed), condition B (carried edges only) and condition C (B plus only the three
broadcast hubs' missing edges — `Q.reset_inh`, `P.reset_inh`, `P.wd.cancel_inh`), 20 fresh
additions each instead of 50. "hub" below means an edge touching one of those three roles; "hard"
is every other designed edge (878 of the netlist's 1,146, close to but not exactly the 884 of the
main placement's "hard" class, which also excludes the 4 mixed-sign edges).

| k_max | carried | hard carried | hub carried | neurons placed | parasitic zeroed | A | B | C | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 693 / 1,146 (60.5 %) | 659 / 878 (75.1 %) | 34 / 268 (12.7 %) | 613 / 614 | 14,166 | 20/20 | 0/20 | 0/20 | 39.7 |
| 8 | 809 / 1,146 (70.6 %) | 699 / 878 (79.6 %) | 110 / 268 (41.0 %) | 614 / 614 | 13,942 | 20/20 | 0/20 | 0/20 | 79.5 |
| 16 | 650 / 1,146 (56.7 %) | 650 / 878 (74.0 %) | 0 / 268 (0 %) | 522 / 614 | 10,159 | 20/20 | 0/20 | 0/20 | 515.7 |
| 32 | 592 / 1,146 (51.7 %) | 592 / 878 (67.4 %) | 0 / 268 (0 %) | 462 / 614 | 5,908 | 20/20 | 0/20 | 0/20 | 444.5 |

**Condition B does not compute a first sum at any of the four `k_max` values tested.** Even at
`k_max = 8`, where 70.6 % of the edges are carried, no ACCEPT is reached in any of the 20 fresh
additions (the same failure mode as the `k_max = 4` report: the producer loads, the arithmetic
runs, nothing reaches the output register). Condition C, adding back only the three hubs' missing
edges, does not help at any `k_max` either — consistent with the earlier finding that the hubs are
purely a reset problem, upstream of what condition C restores. Condition A, which adds every
missing edge, computes all 20 sums at every `k_max`, confirming the mapping and rescaling are
sound at each bound.

**What a looser bound buys.** Doubling `k_max` from 4 to 8 halves the anatomical synapse count an
edge needs (`req_count = ceil(|q| / (k_max * 16))`), and the classes that need it most see the
first gains: `cancel_inh -> hop`, the watchdog cancel hub's 100 inhibitory edges onto the delay
chain (each needing >= 85 quanta at `k_max = 4`), goes from 94 missing to 31; hub coverage overall
rises from 34/268 (12.7 %) to 110/268 (41.0 %), and the total carried fraction from 60.5 % to
70.6 %, the best of the sweep. `reset_inh -> u` / `-> v` (the two reset hubs) shrink more slowly
(62->56, 57->53 missing) because they are gated by a scarce real neuron with the right sign and
fan-out, not only by the quanta bound. Past `k_max = 8` the numbers reverse, and not because the
wiring runs out of headroom: at `k_max = 16` and `32`, far more anatomical edges clear the
now-much-lower threshold, which roughly doubles the candidate-domain sizes the strict layer and its
arc consistency have to carry: the same 4-restart, nominal-300-second budget that placed the
`k_max = 8` design in 79.5 s took 515.7 s and 444.5 s and still left neurons unplaced (522/614 and
462/614, against 613-614/614 at `k_max` <= 8) and both remaining hubs uncarried (0/268 — `hub_carried`
drops to zero at exactly the point the search stops finishing). So the bound is the binding
constraint on coverage only up to about `k_max = 8` for this circuit and this search budget; past
that, the placement search's own scaling with domain size becomes the limiter, not the connectome.

**What it costs.** Loosening `k_max` does not touch the parasitic-edge problem directly — Profile 2
still zeroes every anatomical edge among the hosts that is not a carried designed edge, whatever
the bound — but it changes how much there is to zero and how far a rescaling has to reach.
Parasitic edges to zero fall from 14,166 at `k_max = 4` to 13,942 at `k_max = 8` (2,115 and 2,067 of
them >= 57 synapses, strong enough alone to carry a latch loop's 3,621 quanta), a small drop from
one more host placed and a few more anatomical edges absorbed as carried edges instead of parasitic
ones. The larger drops at `k_max = 16` and `32` (10,159 and 5,908) are mostly an artifact of the
search placing fewer neurons at all (522 and 462 of 614) rather than the bound doing less damage:
fewer hosts means fewer anatomical edges among them, carried or parasitic, full stop. The rescaling
cost is visible in the k histograms: at every `k_max` the carried edges' scale factors spread almost
uniformly from near 0 to the ceiling, and the observed maximum sits within 1-2 % of `k_max` itself
(3.99, 7.979, 15.958, 31.117) — raising the bound does not make the rescaling gentler on average, it
raises how far a single synapse's weight can be pushed, and the edges that already needed the top of
the range at `k_max = 4` still need close to the top of the new, wider range at every larger `k_max`
tested. This is what Profile 2's "weights within bounds" buys and costs at each point: up to
`k_max = 8`, real coverage of the resets and the cancel hub for synapse counts scaled as much as 8x
past their raw anatomical weight; beyond it, the connectome has not changed, but the search needed
to find that coverage has stopped keeping up with the time budget given here.

## Broadcast neurons split into trees

Tool: `Netlist.split_hubs(max_fanout)` (`drosophilos/lib/netlist.py`) and
`drosophilos/bench/h1_split.py`; data: `docs/h1_split.json`. A neuron with more than
`max_fanout` outgoing synapses keeps the first `max_fanout` and hands the rest, in slices, to
new copies of itself that receive every input the original receives — so they spike when it
spikes (verified spike-for-spike on the adder over 20 additions, `tests/test_netlist_split.py`):
no hop is added, no timing changes, and the placement search sees small motifs where it saw a
broadcast. Sources whose fan-out the copies raise are split in turn (a cycle raises `ValueError`).
The three broadcast neurons and, through their inputs, the reset relays and a few chain nodes
split into 40–49 extra neurons; then the placement, the image and conditions A (all missing
edges added, parasitic zeroed), B (carried only) and F (all but the hub class added), 20
additions each, four restarts, seed 0.

| netlist | k_max | carried | hard | hub (reset and cancel outputs) | tree inputs | placed | A | B | F fresh / chained | Profile 3 edges A needs |
|---|---|---|---|---|---|---|---|---|---|---|
| unsplit (§Result) | 4 | 704 / 1,146 (61 %) | 697 / 884 | 7 / 258 (3 %) | — | 612 / 614 | 50/50 | 0/50 | 50/50 / 1/50 | 442 |
| unsplit (sweep) | 8 | 809 / 1,146 (71 %) | 699 / 878 | 110 / 268 (41 %) | — | 614 / 614 | 20/20 | 0/20 | — | 337 |
| max_fanout 8 | 4 | 771 / 1,293 (60 %) | 637 / 888 | 92 / 265 (35 %) | 43 / 140 | 658 / 663 | 20/20 | 0/20 | 20/20 / 1/20 | 517 |
| max_fanout 8 | 8 | 884 / 1,293 (68 %) | 689 / 888 | 130 / 265 (49 %) | 65 / 140 | 663 / 663 | 20/20 | 0/20 | 20/20 / 1/20 | 403 |
| max_fanout 16 | 4 | 721 / 1,194 (60 %) | 604 / 888 | 101 / 258 (39 %) | 16 / 48 | 628 / 629 | 20/20 | 0/20 | 20/20 / 1/20 | 471 |
| max_fanout 16 | 8 | 847 / 1,194 (71 %) | 678 / 888 | 150 / 258 (58 %) | 20 / 48 | 627 / 629 | 20/20 | 0/20 | 20/20 / 1/20 | **346** |

**What the split buys.** The reset and cancel *outputs* go from unplaceable to ordinary: the
hub class rises from 7 of 258 edges at the H0 bound to 92–150, because a copy that inhibits
eight or sixteen latch members is a motif the antennal lobe and the gnathal ganglion have in
quantity (§What the wiring lacks). But every copy needs the original's four inputs — the reset
trigger and three relays — so those excitatory neurons must now reach 16 (or 8) inhibitory
hosts each, and the fan-out problem reappears one hop upstream as the tree-input class: 43 of
140 carried at `max_fanout = 8`, and only 1–2 of the 16 copies of `Q.reset_inh` receive all
their inputs at any setting. Splitting those relays in turn (the transform does it) moves the
problem one hop further up, to the register's completion neuron, which is where it ends. The
count that matters for a Profile 2 image is the last column, the edges a working adder still
needs from Profile 3: 442 at the H0 bound unsplit, 337 at `k_max = 8` unsplit, 346 split at
`max_fanout = 16` — the split and the looser bound each buy about a fifth, and they do not add.
The hard edges lose a little (697 → 678 at `k_max` 8, fan-out 16) because the four-restart
budget is shared with more neurons.

**Whether the placed adder now computes.** Condition B (carried edges only) computes no sum in
any configuration, as before: the missing edges are still spread through the logic (`edge → u`
41–50 of 76 missing, `reset_inh → u/v` 31–43 of 73 each, `cancel_inh → hop` 35–72 of 100). With
the logic completed and only the split resets left as placed (F), every configuration computes
the first sum and 1 of 20 when chained — a reset that reaches 58 % of its targets is not a
reset, since one latch left lit stalls the next word. So splitting is the right shape (it turns
the impossible neuron into possible ones) but not a sufficient one: the reset's *inputs* need a
different design — a reset that each register motif derives locally from a signal it already
receives (its own completion or the consumer's ACCEPT), instead of a tree that must be driven
from one point — and that is a circuit change, not a placement change.

## The placed adder inside the whole brain

Tool: `embed_image.full_graph_topology` and `bench/h1_fullgraph.py`; data: `docs/h1_fullgraph.json`
(compact) and Juno's `data/h1_fullgraph/*.json` (job 405257). The condition-A image — 704 carried
edges rescaled in place, 442 Profile 3 edges added, 17,845 parasitic edges among the hosts zeroed,
2 synthetic neurons — applied to the entire MCNS topology (166,702 neurons, 25.6 M edges), driven
and decoded exactly as in isolation, with the surround driven as Stage H0 drove it: *silent* (nothing
outside the circuit is driven), *Poisson* (every sensory neuron at 2 Hz, 17,922 of them), *burst*
(1,000 random cholinergic neurons together 5 ms after the operands load). Two controls: the hosts'
inputs from the rest of the brain zeroed (327,497 edges: the isolated case embedded), and the hosts'
outputs into the brain zeroed (302,263 edges). About 45–55 s of wall time per addition on a CPU core.

| surround | hosts' brain inputs | hosts' brain outputs | correct | how it fails | spikes entering the circuit (max per ms, exc / inh) | circuit spikes per addition (isolated: 14–16 k) |
|---|---|---|---|---|---|---|
| Poisson 2 Hz | **zeroed** (327,497) | live | **10 / 10** (7 spike-identical to isolation) | — | 0 / 0 | 14.7 k |
| silent | live | live | 0 / 30 | 19 faults, 6 timeouts, 5 incomplete; ACCEPT at 16 ms (the fault path) | 28,041 / 14,840 | 71 k |
| Poisson 2 Hz | live | live | 0 / 30 | 22 faults, 2 timeouts, 6 incomplete | 25,543 / 12,477 | 71 k |
| burst | live | live | 0 / 30 | 21 faults, 8 timeouts, 1 incomplete | 24,304 / 16,706 | 71 k |
| Poisson 2 Hz | live | **zeroed** (302,263) | 0 / 10 | 9 faults, 1 incomplete | 45,855 / 15,913 | 70 k |
| Poisson 2 Hz, chained | live | live | 0 / 20 | 19 faults, 1 no ACCEPT | 25,662 / 12,480 | 71 k |

**What it says.** The image is a circuit only when the rest of the brain cannot reach its hosts.
With the hosts' 327,497 anatomical inputs live, the adder computes nothing under any surround —
not even a silent one: the circuit's own spikes leave through the hosts' 302,263 output edges at
anatomical weight, 24,000 neurons of the surround fire 3.8 M spikes within the 1.2 s window, and
some 20,000–28,000 spikes per millisecond come back into the 614 hosts, whose output rails light
within 16 ms of loading (a fault, before any arithmetic). Silencing the outputs does not help: under
the 2 Hz sensory drive the brain itself fires ~1.9 M spikes from ~38,000 neurons in 1.2 s (the
inputs-zeroed run shows the same surround activity with the circuit sealed off), and 45,000
excitatory spikes per millisecond arrive at the hosts. H0's 15-neuron circuit survived this surround
because its hosts were chosen for a small isolation cost (`embed_h0.isolation_cost`); the placement
search never looked at it, and it put the adder on antennal-lobe and gnathal neurons with ~530
inputs from the brain each. So the Profile 2 image of the adder is, today: 61 % of its edges the
fly's own, 442 added, and **327,497 documented zeroings** of the hosts' inputs on top of the 17,845
among the hosts — computing all sums inside the whole brain with those zeroings, none without.

The next placement objective is therefore not coverage alone: the hosts' external in-degree (and
the surround's response to their outputs) must be part of the score, and the whole-brain LIF at
these weights is itself a storm under the 2 Hz drive — the "background-input envelope" a Profile 2
circuit must tolerate is set by that, not by the circuit.

## Placing for quiet hosts

Tool: `place_netlist(..., isolation_weight=w)` and `bench/h1_isolation.py`; data: `docs/h1_isolation.json`.
Every candidate ranking (domain order, lookahead, hub coverage, repair) now subtracts
`w · (log(1 + inputs) + 0.25 · log(1 + outputs))`, the host's synapses from and to the rest of the
brain; an unplaced neuron is charged the noisiest neuron's penalty so the weight never rewards
leaving one out. The 4-bit adder, `k_max = 4`, four restarts, seed 0:

| w | carried | hard | hub | placed | hosts' external input synapses: mean / p90 / max | external input edges | Profile 3 edges A needs | A | B |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 690 / 1,144 (60 %) | 659 / 878 | 32 / 268 | 613 | 5,459 / 10,793 / 47,712 | 315,326 | 453 | 20/20 | 0/20 |
| 0.5 | 587 (51 %) | 558 | 29 | 612 | 3,166 / 7,132 / 39,246 | 215,864 | 559 | 20/20 | 0/20 |
| 1 | 535 (47 %) | 526 | 9 | 610 | 3,216 / 8,382 / 50,448 | 192,076 | 609 | 20/20 | 0/20 |
| 2 | 469 (41 %) | 466 | 3 | 604 | 2,966 / 6,647 / 45,638 | 190,789 | 677 | 20/20 | 0/20 |
| 4 | 460 (40 %) | 452 | 8 | 607 | 3,218 / 7,604 / 45,918 | 206,277 | 686 | 20/20 | 0/20 |

The trade is poor. Between `w = 0` and `w = 2` the hosts' external input synapses fall by 45 % (a
mean of 5,459 to 2,966 per host — still thousands) while the carried edges fall from 60 % to 41 %
and the hubs are lost entirely; past `w = 2` nothing improves. The reason is structural: a neuron
that can carry a latch loop needs a reciprocal partner at ≥ 57 synapses, and in MCNS such neurons
are the brain's hubs — antennal-lobe local neurons, gnathal and nerve-cord interneurons with
thousands of inputs — so "strong enough to be a latch" and "quiet" pull in opposite directions.
The mapping saved as `docs/h1_placement_mapping_isolated.json` (`w = 2`) is the quietest that
still computes in isolation. In the whole brain (Juno 406188) it computes nothing either: silent
surround 0 / 20 (14 no ACCEPT, 5 timeouts, 1 fault), 2 Hz Poisson 0 / 20 — the flood halves
(12,800 excitatory spikes per ms at the hosts instead of 28,000) and the channel now stalls
instead of double-railing, but stalls all the same.

**The control that locates the cause.** With the hosts' *outputs* into the brain silenced (302,263
edges for the coverage placement, 188,031 for the quiet one) and nothing outside driven, both
mappings compute **10 / 10** — so in a silent brain the storm is the circuit's own doing: its
latches fire at 213 Hz, their spikes leave through the hosts' anatomical outputs at full weight,
24,000 neurons wake and ~20,000 spikes per ms come back. Silencing the outputs is a smaller and
cleaner documented edit than silencing the inputs (the circuit is forbidden to disturb the brain,
rather than deafened to it), and under it the adder is a Profile 2 circuit *in a quiet brain*. It
does not survive the 2 Hz sensory drive (0 / 10 in the first run's outputs-zeroed Poisson
condition): under that drive the simulated brain fires ~1.9 M spikes per 1.2 s on its own, and
45,000 spikes per ms reach hosts with thousands of inputs each. Whether a fly brain at these LIF
parameters is really that excitable is a question about the model's surround, not the circuit —
H0's envelope was measured for 15 quiet hosts, and it does not transfer to 614 hubs.

## One kernel cell

Tool: `drosophilos/bench/h1_cell.py`; data: `docs/h1_cell.json`. The next real target after the
adder is a resident kernel (`lib/kernel.py`, `build_pipeline`; `docs/a3_kernels.md` §1–4), and its
smallest instance is one cell: `c1 = input + 1` at 8 bits — the host-loaded input register, the
cell's operand gates, ALU, stage and master register, with their completion trees, resets and
watchdogs. It is five times the adder: 3,170 neurons, 5,768 synapse entries (5,760 distinct
pairs). Placed with `place_netlist(k_max=4, restarts=2, time_limit_s=1800, seed=0)` as built and
after `split_hubs`; the 4-bit cell (1,906 neurons) the same way, because it is the one size at
which the requested `split_hubs(max_fanout=16)` terminates (below). No simulation. Each placement
ran in a child process whose resident memory was polled and capped at 8 GB; the cap was never
approached.

| | adder | adder, split 16 | 4-bit cell | 4-bit cell, split 16 | **8-bit cell** | **8-bit cell, split 20** |
|---|---|---|---|---|---|---|
| neurons | 614 | 629 | 1,906 | 1,948 | 3,170 | 3,222 |
| synapse entries (distinct pairs) | 1,146 (1,144) | 1,194 | 3,412 (3,404) | 3,555 (3,547) | 5,768 (5,760) | 5,958 (5,950) |
| latches / relays / chains | 72 / 79 / 18 | — | 217 / 256 / 39 | 220 / 287 / 38 | 375 / 444 / 63 | 380 / 514 / 62 |
| hubs (> 20 partners) and fan-outs | 3: 115, 102, 41 | 0 | 9: 255, 78, 69, 47, 39, 38, 29, 24, 21 | 0 | 12: 455, 109, 94, 87, 79, 70, 45, 34, 33, 32, 29, 21 | 0 (52 copies, 190 duplicated inputs) |
| carried | 704 / 1,146 (61.4 %) | 721 / 1,194 (60.4 %) | 2,030 / 3,404 (59.6 %) | 2,059 / 3,547 (58.0 %) | **3,073 / 5,760 (53.3 %)** | **3,046 / 5,950 (51.2 %)** |
| hard (motif and motif-to-motif) | 697 / 884 (78.8 %) | 604 / 888 (68.0 %) | 1,947 / 2,812 (69.2 %) | 1,853 / 2,812 (65.9 %) | 2,953 / 4,680 (63.1 %) | 2,815 / 4,680 (60.2 %) |
| hub fan-out (after a split: the trees' outputs) | 7 / 258 (2.7 %) | 101 / 258 (39.1 %) | 83 / 580 (14.3 %) | 156 / 589 (26.5 %) | 120 / 1,068 (11.2 %) | 213 / 1,078 (19.8 %) |
| tree inputs (duplicated onto copies) | — | 16 / 48 | — | 50 / 134 (37.3 %) | — | 18 / 180 (10.0 %) |
| neurons placed | 612 / 614 | 628 / 629 | 1,894 / 1,906 | 1,938 / 1,948 | 3,097 / 3,170 | 3,122 / 3,222 |
| latches complete (both loop edges on a real mutual pair) | 58 / 72 | — | 158 / 217 | 157 / 220 | 194 / 375 | 306 / 380 |
| parasitic anatomical edges among the hosts | 17,844 | 14,043 | 87,575 | 91,749 | 199,131 | 181,654 |
| search seconds (restarts) | 65 (8) | 54 (4) | 66 (2) | 62 (2) | 167 (2) | 77 (2) |
| peak memory (child process, `ru_maxrss`) | not measured | not measured | 2.34 GB | 2.31 GB | 2.10 GB | 2.14 GB |

The adder columns are §Result (8 restarts) and §Broadcast neurons split into trees (`max_fanout 16`,
`k_max 4`, 4 restarts); the cell columns are this bench, 2 restarts each, best restart kept
(for the 8-bit cell restart 0, hubs last, unsplit — the hubs-first restart was 81 edges worse — and restart 1, hubs first, split, 17 edges better than restart 0). The
"hard" and "hub" classes are the placer's own (`_Design.hard` / `.soft`); after a split there are
no hubs, so "hub" counts the edges out of the former hubs' trees and "tree inputs" the inputs the
split duplicated onto the copies.

**`split_hubs(16)` does not terminate on the 8-bit cell.** The fault latch `c1.faultL.u` (32
outputs) drives the register's reset trigger `c1.Q.reset`, whose inhibitor `c1.Q.reset_inh` (455
outputs) inhibits the fault latch: splitting any of the three duplicates inputs onto copies, which
pushes the next one past 16, which pushes the first past 16 again, and the transform raises
`ValueError` (cycle through `c1.faultL.u`). The bench steps the bound up by 4 and 20 is the smallest
that terminates (52 copies: 22 of `c1.Q.reset_inh`, 5 of `c1.M.reset_inh`, 4 each of the cancel
and `IN.Q.reset_inh`, 3 of `c1.actd.d10`, one or two of the rest, and one each of the reset
trigger and its three relays). At 4 bits, 16 works (42 copies) because the trigger stays under
the bound after its copies are added. A bound of 16 on a kernel cell therefore needs the reset
redesign that §Broadcast neurons already asked for, not a larger transform.

**What changes at five and ten times the size.** The search's cost does not: setup (dense
masks, motif instances, two rounds of arc consistency) went from 0.7 s for the adder to 16 s at
1,906 neurons and 25 s at 3,170; one descent from 4–13 s to 22 s and 31 s; the whole run of two
restarts with repair took 167 s of the 1,800 s allowed, at 2.1 GB (the connectome's own sparse
matrices are about 1.5 GB of that; the 3,170 dense 166,700-byte rows add 0.5 GB and are freed
after setup). Time and memory scale close to linearly in designed neurons and neither is near a
limit at this size. What runs out is the connectome's supply of latches. The 750 latch members
of the 8-bit cell all draw from the same 771 real neurons — the entire membership of the 1,223
mutual cholinergic pairs at ≥ 57 synapses; the pairs share members heavily (the 2,446 of §What
the wiring lacks counts each pair's two ends), and a placement that gives every designed neuron
its own host can use at most a *disjoint* set of pairs: a maximum matching of that graph has
**343 pairs**. So 375 designed latches cannot all be complete at `k_max 4` whatever the search
does; the unsplit run completes 194 (52 %; 4 latches unplaced, 177 partial: one member on a
mutual-pair neuron whose partner was taken by another latch or relay), the split run 306 (81 %,
89 % of the 343 ceiling). Relays are the second wall: 888 designed relay neurons against 913
distinct candidates. The classes that lose the most edges are the ones that were already thin in
the adder, now at ten times the count: `edge → u` (relays igniting their latch) 335 of 491 missing,
`reset_inh → u/v` 594 of 684, `u → veto` 264 of 406, `edge_inh → edge` 203 of 513, the latch loops
themselves 283 of 764. The carried fraction falls from 61 % (adder) to 60 % (4-bit cell) to 53 %
(8-bit cell) as the latch demand goes from 72 to 217 to 375 against 343 disjoint pairs; the hard
edges fall from 79 % to 63 % for the same reason. And the parasitic edges grow faster than the
circuit: 199,131 anatomical edges among the 3,097 hosts (64 per host, against 46 for the 4-bit cell's 1,894 hosts and 29 for the
adder's 612), because the hosts are the brain's most strongly interconnected neurons and taking
more of them takes more of the edges among them.

The split run shows a limit of the search itself, distinct from the connectome's. With every
fan-out ≤ 20 no neuron is a hub, so the 1,078 former broadcast edges become *hard*, and the strict
layer's arc consistency then finds the design globally infeasible: 18 designed neurons lose their
last candidate to a neighbour that still had candidates — the `actd.d10` copies' relays
(`c1.aNrM.g.veto ← g.edge` at ≥ 29, 11 of them; the fan-out node of §What the wiring lacks, now
enforced), five input-register master latch members (`IN.M.bNrM.u`) with no ≥ 57 edge onto any of their
veto's 5–7 remaining candidates, `c1.sub.g.edge`,
and `IN.Q.reset ← IN.Q.reset_inh.c4` (the copy has 2 candidates and neither is driven by the
trigger's) — and the implementation then propagates *emptiness*: an empty domain gives its hard
neighbours no support, so they empty too, and 1,440 domains are empty after the first round and
2,239 of 3,222 (69 %) after the second, 484 of the 760 latch members among them. The search runs on
the loose layer (the motif's own instances, no context) for two thirds of the circuit. It still
carries 51 %, and *more* complete latches than the unsplit run (306 against 194) because a latch
placed from the loose layer is placed as a real mutual pair with no other demands — but the
chains, which are found by depth-first search inside the strict domains, collapse (11 complete of
62 against 39 of 63; the 92-hop watchdog chain is no longer even read as a chain, since each hop
now has a second hard input from a cancel copy, and 42 hops go unplaced) and the trees' inputs
carry 18 of 180. Arc consistency that empties a neighbour on the strength of an already-empty
domain is throwing away information the search then has to rebuild motif by motif; making an
empty domain fall back to its loose layer *before* it propagates is the one search change this
run asks for, and it is cheap. At 4 bits the same split leaves no domain empty (the trees are
small enough for consistent candidates to exist), and the split costs 1.6 points of coverage instead of 2.2.

**What it implies for a 65,000-neuron render kernel.** Scale the cell twenty times: about 5,300
latches (`capacity_doom.md` §5's count for the four-cell render kernel; at this cell's ratio of
750 latch members per 3,170 neurons, 65,000 neurons would hold nearer 7,700) against a supply of disjoint real pairs that is
**343 at `k_max 4`, 1,047 at `k_max 8`, 2,896 at `k_max 16`** (1,223 / 4,292 / 12,315 pairs on
771 / 2,432 / 6,979 neurons; strong-edge cores of 3,197 / 10,131 / 22,347 neurons). So at most
6.5 % of the kernel's latches can have both loop edges on real neurons at the H0 bound, 20 % at
`k_max 8`, 55 % at `k_max 16` — and those are upper bounds from a perfect matching that ignores
every other constraint on the same neurons (the 8-bit cell reaches 89 % of its ceiling only when
the split frees the latches from their context). The relay supply is of the same order (913
candidates for 888 relay neurons already at 3,170), so the arithmetic's relays are as short as its
registers. Then the search: its memory is one dense 166,700-byte row per designed neuron during
setup, 10.8 GB for 65,000 — above the 8 GB this laptop can give it — before any of the anatomy's
own masks, so the setup has to go sparse (index arrays from the start, or masks per motif kind
instead of per neuron) before a kernel can even be attempted here; its time, extrapolated
linearly from 25 s setup and 31 s per descent at 3,170 neurons, is 8–9 minutes of setup and about
11 minutes per descent, i.e. an hour for a few restarts, which is affordable. And the parasitic
count: 65,000 hosts is 39 % of the brain's neurons and, on the trend of 29 → 46 → 64 edges per host
from 612 to 1,894 to 3,097 hosts, several million anatomical edges to zero — the "Profile 2 image" would
be a documented deletion of a large part of the fly's wiring. The honest reading of the numbers:
the connectome carries about half of a kernel cell's edges and cannot carry more than about 6 %
of a render kernel's latches at the H0 bound (a fifth at `k_max 8`, half at 16); the register file of any kernel this
size is a Profile 3 object, and what Profile 2 can host is the small, latch-light logic between
registers.

Reproduce (about 8 minutes with the MCNS data present; no simulation):

```python
from drosophilos.bench.h1_cell import run_all, place_cell, latch_capacity
run_all()                                   # docs/h1_cell.json: n8 and n4, unsplit and split, plus the capacity table
place_cell(8, split=False, restarts=2)      # one configuration in-process (2.3 GB)
```

## What the fly's wiring contributes: shuffled controls

Tool: `bench/h1_shuffle.py`; data: `docs/h1_shuffle.json`. The plan's Stage H asks for matched
shuffled comparisons: is the adder's placement a property of the fly's actual wiring, or of any
graph with those degrees? Three controls, two seeds each, the same search (k_max 4, two restarts,
300 s), the real MCNS beside them under the same settings:

| substrate | what it keeps | mutual pairs ≥ 57 | disjoint pairs | strong core | carried | hard | hub |
|---|---|---|---|---|---|---|---|
| **MCNS, real** | everything | 1,223 | 343 | 3,197 | **683 / 1,144 (60 %)** | 648 / 884 | 35 / 256 |
| configuration shuffle | every neuron's in- and out-synapse counts, transmitters; the wiring randomised | 24 | 22–24 | ~6,350 | 570–579 (50 %) | 538–565 | 5–41 |
| weights shuffle | who connects to whom; the synapse counts permuted among the edges | 62 | 62 | ~1,450 | 506–510 (44 %) | 487–508 | 2–19 |
| transmitter shuffle | the wiring and its strengths; transmitter labels permuted within superclass | 2,482–2,888 | 698–715 | ~6,650 | **745–760 (65 %)** | 721–725 | 24–35 |

**The fly's wiring beats a degree-matched random graph by ten points**, and the difference is
reciprocity: a random graph with MCNS's degrees has 24 strong reciprocal pairs where the fly has
1,223, and those pairs are the latches (the configuration control still carries the relays, vetoes
and chains, which are paths, not loops). Which edges are *strong* matters more than who is wired to
whom: permuting the synapse counts over the real edge set leaves 62 strong pairs and a 1,400-neuron
core, and the placement falls to 44 %. And the fly's sign layout is a handicap for this circuit:
relabelling transmitters at random within each superclass more than doubles the excitatory
reciprocal pairs (2,500–2,900) and lifts the placement to 65 %, because many of the fly's strongest
reciprocal pairs are between *inhibitory* local neurons (the antennal lobe's GABAergic LNs), which a
two-neuron excitatory latch cannot use. So the measured statement is: the fly's reciprocal strong
motifs are real and specific (fifty times a random graph's), they are what carries the register
logic, and about a third of them are of the wrong sign for an excitatory latch — which is one more
reason an inhibition-based latch design would fit this substrate better than the current one.

## Placing flip-flops

Tool: `place_netlist` now reads a third pair motif off the netlist, `ffpair` — two neurons with a
non-zero `bias` that inhibit each other (`protocol.flipflop.add_flipflop`; the loop is 1.0× loop
quanta, ≥ 57 synapses at `k_max = 4`) — and places it as a unit on a real **mutual inhibitory pair**
(GABA or glutamate both ways at ≥ the loop threshold) whose members each have at least one
excitatory input ≥ that threshold from outside the pair: the driven pool `bench/h1_inhpairs.py`
counted (358 disjoint pairs at `k_max = 4`; `ff_driver=False` opens every mutual inhibitory pair).
The driver is not needed anatomically — the bias is a Profile 2 parameter edit — so the audit only
reports how many chosen hosts have one (`Placement.ffpair_drivers`). A flip-flop member is
inhibitory whatever else it drives: an edge relay reading `.u` puts two *excitatory* edges on a
GABA-designed neuron, and those are of the wrong sign for any host under Dale's law
("wrong sign (mixed-sign designed neuron)" in the audit), on this or any connectome. The image
(`build_image`) carries each designed bias onto its host as a `ParameterEdit("bias", 0 → mV)`,
counted as `biases` in the manifest (a synthetic neuron keeps its bias as part of its own
definition: `biases_synthetic`), and into the image topology's `bias`; `full_graph_topology` puts
each bias on the host's full-graph index. The excitatory latch motif is unchanged
(`tests/test_embed_netlist.py`, `tests/test_embed_image.py`).

**The toy.** 32 flip-flops in a line, each joined to the next by one edge relay read off the
previous pair's `u` and igniting the next pair's `u` (`chain_toy_netlist` in
`tests/test_embed_netlist.py`; 126 neurons, 188 edges), against 32 excitatory latches wired the
same way. Placement only — nothing here is simulated, and the flip-flop toy is not a working
circuit (a single ignite pulse never sets a flip-flop; the sign of its readout is the point below).
Real MCNS, `k_max = 4`, two restarts, seed 0, ~28 s each:

| netlist | objective | carried / carriable / designed | pairs complete | relays complete | hosts' external input synapses: mean / p90 / max | pair members: mean / median | hosts with a driver |
|---|---|---|---|---|---|---|---|
| 32 flip-flops | coverage (w = 0) | **126 / 126** / 188 (67 %) | **32 / 32** | 31 / 31 (E, I singles: `edge_inh -> edge` and `edge -> u` all carried) | 3,686 / 7,038 / 14,541 | 3,815 / 2,653 | 64 / 64 |
| 32 latches | coverage (w = 0) | 162 / 188 / 188 (86 %) | 30 / 32 (2 partial) | 24 / 31 (7 partial) | 6,678 / 13,259 / 49,211 | 5,922 / 5,044 | — |
| 32 flip-flops | quiet hosts (w = 2) | 102 / 126 / 188 | 20 / 32 (12 partial) | 31 / 31 | 2,017 / 3,607 / 13,004 | 1,374 / 1,232 | 60 / 64 |
| 32 latches | quiet hosts (w = 2) | 107 / 188 / 188 | 16 / 32 (16 partial) | 15 / 31 | 2,508 / 5,937 / 50,431 | 1,675 / 674 | — |

The 62 edges the flip-flop toy cannot carry are exactly the 31 × 2 readout edges `u -> edge`,
`u -> edge_inh` (wrong sign); every carriable edge is carried, in one descent with one backtrack
and no repair. The latch toy loses 26 edges to the strong-edge supply itself (10 `edge -> u`, 6
`edge_inh -> edge`, 2 loops, 1 neuron unplaced), after 95 backtracks and a 9-edge repair. So on
the fly's wiring the flip-flop's loop is the *easier* motif to place — 32 driven inhibitory pairs
under 31 relay targets, all complete — and its hosts take **42 % fewer input synapses from the
rest of the brain** than the latches' (3,686 against 6,678 per host; the pair members 3,815
against 5,922: nerve-cord interneurons and central-brain GABAergic cells instead of the latches'
`PFNv` and antennal-lobe `lLN` hubs), with a third of the parasitic edges to zero (327 against
1,042). Under the quiet-host objective both toys give up loops for quieter hosts (the repair
breaks 12 of the 32 flip-flop loops, and 4 members land on neurons with no driver at all) and
the flip-flops stay ahead: 2,017 against 2,508 mean, and a p90 of 3,607 against 5,937. These
are placement numbers only; the driven disjoint pool's own mean is 6,677 input synapses per member
(`docs/h1_inhpairs.json`, `k_max = 4`; `docs/capacity_doom.md` §5 quotes "66" for the same
figure, which does not match the JSON), so at w = 0 the search already lands on the quieter
half of that pool.

**What it says.** The flip-flop is placeable today as a *rail* — 32 of 32 pairs on real driven
inhibitory pairs, biases carried as parameter edits, Profile 2 without the readout edges — but
not as a *source*: the fly's inhibitory pair can only ever inhibit its readers, so the edge relay
that reads `.u` as an excitatory train (the primitive every consumer in the protocol uses) has no
host. The next design step is the inhibitory readout the capacity note asked for: a biased
excitatory neuron that `u` silences (fires while CLEAR, stops on SET) or a target `u` holds down
directly, and a relay reading *that* — then the 62 impossible edges become carriable ones, and the
comparison above can be rerun on a circuit that could also be simulated.

**With the proxy (`fftriple`).** `add_flipflop(proxy=True)` adds p, a biased *excitatory* neuron
whose only input is v's inhibition, and readers connect to p (`docs/a1_flipflop.md`). The placer
now reads that off the netlist as a third flip-flop motif, `fftriple`: the biased mutual inhibitory
pair plus a biased excitatory neuron whose only designed input is an inhibitory edge from one
member (a pair without a proxy stays `ffpair`; a pair with both p and q is a triple on p, q a
single joined by its ordinary hard edge). It is placed as a unit on a real **(driven mutual
inhibitory pair, excitatory neuron the v host inhibits ≥ the proxy's threshold)** — the loose
`proxy_readout` pool of `bench/h1_inhpairs.py` (700 of the 729 driven pairs have such a proxy at
`k_max = 4`; 358 disjoint triples by MILP); the strict variant's "proxy with an excitatory output"
is what p's own designed outputs impose through the degree masks. p's outgoing edges (to its edge
relay, a veto) are excitatory from an excitatory neuron: ordinary hard edges the search carries,
and the relay read off p is a real `relay` motif whose source is p. The loose layer keeps the
proxy requirement off the pair (a triple that cannot be completed still lands its pair on a real
driven pair and leaves p unplaced rather than faking it, `tests/test_embed_netlist.py`); the strict
layer restricts v to members that inhibit some excitatory neuron. The image needs nothing new:
every biased designed neuron's bias goes onto its host as before, p's included (six bias edits for
two proxied flip-flops, `tests/test_embed_image.py`). The audit adds `Placement.proxy_readout`
(proxies designed / placed, designed edges out of a proxy / carried) and `summary()` reports it.

The same 32-flip-flop toy, each relay now read off the previous flip-flop's p (`chain_toy_netlist(32,
"ffpair", proxy=True)`: 158 neurons, 220 edges, every one of a carriable sign), against the pair-only
toy above. Real MCNS, `k_max = 4`, two restarts, seed 0; the times are `place_netlist` alone (the
~28 s quoted above included loading the connectome):

| netlist | objective | carried / carriable / designed | pairs or triples complete | relays complete | proxy readout `p -> edge`, `p -> edge_inh` carried | hosts' external input synapses: mean / p90 / max | pair members: mean / median | proxies: mean / median | hosts with a driver | parasitic to zero | search |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 flip-flops, pair only | coverage (w = 0) | 126 / 126 / 188 (67 %) | 32 / 32 | 31 / 31 (E, I singles) | — (62 wrong sign) | 3,708 / 7,303 / 14,403 | 3,626 / 2,587 | — | 64 / 64 | 314 | 2.1 s |
| 32 flip-flops with proxies | coverage (w = 0) | **219 / 220 / 220 (99.5 %)** | **32 / 32** | 31 / 31 | **62 / 62** | 6,546 / 11,405 / 40,517 | 7,651 / 7,415 | 4,922 / 5,135 | 64 / 64 | 1,883 | 3.3 s |
| 32 flip-flops, pair only | quiet hosts (w = 2) | 104 / 126 / 188 | 19 / 32 (13 partial) | 31 / 31 | — | 2,061 / 3,760 / 13,004 | 1,412 / 1,201 | — | 59 / 64 | 335 | 2.5 s |
| 32 flip-flops with proxies | quiet hosts (w = 2) | 115 / 220 / 220 | 3 / 32 (29 partial) | 14 / 31 | 31 / 62 | 2,497 / 6,002 / 17,475 | 2,227 / 1,651 | 1,144 / 554 | 49 / 64 | 572 | 3.5 s |

The 62 edges the pair-only toy could never carry are carried: every one of the 32 triples lands
on a real driven pair with a cholinergic proxy the v host inhibits, all 31 relays read off those
proxies are complete, and the one missing edge in 220 is a single `edge -> u` (one relay's ignite
into the next pair's u; the hub-first restart found this placement after 31 backtracks, the
hub-last one stalled at 208 with one triple and two relays partial). Carried fraction 67 % → 99.5 %.

The price is exposure: the (pair, proxy) pool is a different, better-connected part of the wiring
than the driven pairs alone. Pair members take **7,651** input synapses from the rest of the brain
against 3,626 in the pair-only toy (2.1×; the driven disjoint pool's own mean is 6,677, and the
`proxy_readout` packing's pairs 6,712, so the pair-only search was landing on the *quiet* half of
the driven pool and the triple constraint takes that freedom away), the proxies 4,922 (the
packing's quietest-proxy re-assignment reaches 1,782 — the search takes the first proxy that
completes the triple in context, not the quietest), and the parasitic edges to zero go from 314
to 1,883. The hosts: nerve-cord interneurons for the pair (`IN13A001`, `IN19A005`, `IN19A001`)
and ascending / nerve-cord neurons for p (`INXXX466`, `AN19B009`, `IN03A004`). Under the
quiet-host objective (w = 2) both toys give up their loops, and the proxied one gives up more: the
repair breaks 26 of 32 `v -> p` edges and 22 of 31 `p -> edge` edges for proxies on sensory
neurons (`SNpp52`, `SNpp45`; median 554 input synapses) — at a weight of 2 a triple's three edges
are worth less than the log-exposure gap between a nerve-cord interneuron and a sensory afferent.
The weight was calibrated for the adder (above); a flip-flop circuit wants a smaller one, or the
quietest-proxy re-assignment the bench does after the pairs are fixed.

**What it says now.** The flip-flop is placeable as a *source* as well as a rail: 32 of 32 triples
on real (driven pair, proxy) instances, the readout edges carried, biases carried as parameter
edits, and every edge of the chain but one Profile 2 — a circuit that could also be simulated
(`tests/test_embed_netlist.py::test_two_proxied_flipflops_are_placed_completely_including_the_readout`
places two proxied flip-flops joined by an edge relay from p into the second's three-pulse set
chain, 15 of 15 edges, on a synthetic connectome with a planted triple). What it costs is a
noisier host set than the pair alone, which is the next thing to trade off.
