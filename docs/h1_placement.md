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
(reported only), neural simulation of the placed circuit, and the whole-cell (~1,300-latch) target.

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
