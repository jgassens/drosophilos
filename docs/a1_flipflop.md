# A1: the flip-flop latch (two inhibitory neurons, tonic bias, mutual inhibition)

`drosophilos/protocol/flipflop.py`; contract `docs/contracts/flipflop.yaml`; tests
`tests/test_flipflop.py`. Everything below is RefSim on circuits of 2–8 neurons, ≤ 2.1 s of
neural time, default parameters, no weight noise.

## Why

`docs/capacity_doom.md` §5: the fly has 358 / 1,105 / 3,717 disjoint strong mutually
inhibiting pairs with drivers, on quiet neurons the excitatory latch cannot use. A latch
made of such a pair would roughly double the latch supply. This note builds one and
measures whether the token protocol's primitives can use it.

## The design

Two inhibitory neurons `u` and `v`, each with a **bias** of 58 mV added to its resting
potential (so alone it fires tonically), each inhibiting the other with 1.0× loop
(3,621 quanta) per spike. Exactly one fires: `u` while SET, `v` while CLEAR.

- **Bias 58 mV gives the latch's rate exactly**: the time to climb the 7 mV gap after the
  2.2 ms refractory hold is τ_m·ln(b/(b−7)) = 2.5 ms, period 47 steps, 212.8 Hz — the
  excitatory latch's loop period step for step. The table of rate vs bias is in the contract
  (7.5 mV: 17.5 Hz … 80 mV: 250 Hz).
- **Bistability floor**: 0.8× loop holds both states; at 0.75× the silent member escapes and
  both fire at ~123 Hz. 1.0× is the default (a 25 % margin, above the campaigns' ±8 %
  weight noise). The parked member sits ~15 mV below threshold (mean; ~1 mV ripple).
- **Bias is a new per-neuron parameter**: `Netlist.neuron(role, bias=)` / `set_bias`,
  carried by `topology()` into `Topology.bias` (None when every entry is 0, so no existing
  topology changes) and read by a simulator through `topo.sim_bias()`; `summary()` counts
  `biased_neurons` for a Profile 2 image's parameter-edit list.

## What it costs to switch (all-phase minima)

The parked member is 15 mV below threshold and the latch's ignition pulse is 12.6 mV, so
**one standard ignite pulse never sets it** (0 of 47 phases). One spike of the newcomer
only delays the running member by a few ms, so switching needs either a train or a pulse
2–4× the standard size:

| switch | single pulse into u | train into u (53 steps apart) | pair (u and v at once) |
|---|---|---|---|
| SET (from CLEAR) | ≥ 2.25× ignite | 3 × 0.75 ignite, or 2 × 1.25, or 4 × 0.6 | 1.5× ignite into u + 1.5× loop into v |
| CLEAR (from SET) | ≥ 4.0× loop | 3 × 1.0 loop, or 2 × 1.5, or 4 × 0.75 (the standard reset train; 0.7 switches only 75 %) | 1.5× loop into u + 1.5× ignite into v |

Defaults (`set_pulse`, `clear_pulse`) carry a margin: SET = **four** standard ignite pulses
(`SET_TRAIN_PULSES`; three until 2026-09-16 — three switch every phase of the nominal pair
but lock a mix-B-perturbed one, §Lockstep) or one 3× pulse; CLEAR = three standard reset
pulses at −1.5× loop (or one 5× loop pulse). All switch at every one of the 47 phases.

Switching time: SET train — `u`'s first spike 6.0–6.5 ms after the first pulse (it is the
second pulse that fires it), `v`'s last spike ≤ 12.6 ms; SET single pulse — 1.4–1.7 ms and
≤ 6.8 ms; CLEAR — `u` stops at once (no spike after the first pulse lands), `v`'s first
spike 9.4–14.0 ms later (it climbs 15 mV with τ_m). No doublet after SET: first ISI
3.7–4.8 ms, 200–220 Hz over the first 50 ms, 212.8 Hz from then on.

Hold: 2 s with no input in both states (425 spikes each, 212.5 Hz).

## Noise margins (compared with the excitatory latch)

| stray pulse | flip-flop | excitatory latch |
|---|---|---|
| excitatory, into the silent member / latch | never switches to SET below 2.0× ignite (2.0× flips 33 % of phases) — but from 1.15× up it drops the pair into **lockstep**, see §Register below | ignites at 1.1× need (2,845 quanta, ~7.7 mV) |
| inhibitory, into the firing member / one latch member | survives 2.25× loop (8,147 quanta); 3.0× flips 71 % | survives 2.0× loop; 3.0× kills 79 % |

The flip-flop is ~3× harder to set by a stray excitatory pulse than the latch is to ignite,
and about as hard to knock down.

## Power-on and lockstep — the part that does not take care of itself

Both members start at rest, so with nothing else they reach threshold together at 2.5 ms,
inhibit each other together, recover together, and keep firing **in lockstep** (~105 Hz
each at 1.0× loop) for ever. This is a stable third state (§Lockstep, below, maps it). Giving `v` a larger bias does
not break it: +10 mV still locks at 1.0× loop, and the +15 mV that does resolve it leaves
`v` unparked in SET (its rate is 240 Hz and its park depth ~0). So:

- **Power-on needs a pulse**: one 1.0× loop inhibitory pulse into `u` at step 0
  (`power_on_pulse`) starts it CLEAR with `u` never firing. Without it an edge relay on
  `u` fires spuriously at 4.9 ms.
- A clear train into `u` resolves lockstep to CLEAR whenever it happens.

## Compatibility with the existing primitives (measured)

| primitive | works? | measured |
|---|---|---|
| `add_edge_relay(source=ff.u)` | **yes** | fires exactly once per SET (3 of 3 cycles, ~4.5 ms after `u`'s first spike), never on the tonic train; spurious once at power-on without the power-on pulse |
| `add_veto_relay(vetoes=[ff.u])` | **yes** | relay and target held while SET; a re-drive 50 ms after CLEAR fires the relay and ignites the target, 40 ms does not — the 55 ms rule holds unchanged (`u` stops at once on CLEAR, like a killed latch) |
| `add_veto_relay(target=ff)` and `celement._ignite_from` | **no** | their single ignite pulse into `.u` never sets it (0 of 47). Fix: `add_set_chain` — a chain of a trigger and three relays that turns one ignite pulse into four (measured end to end: latch train → edge relay → chain → SET) |
| `add_reset` on `ff.members` | **no** | a train into **both** members (4 × 0.75 or 3 × 1.5 loop) leaves the flip-flop where it was (SET stays SET, 12 of 12 phases). Fix: `add_clear_chain` — the same controller aimed at `u` only (measured: trigger → 3 pulses → CLEAR) |
| kill train into `u` only | **yes** | 3 × 1.0 loop at every phase; the standard 4 × 0.75 loop too, with ~7 % margin |
| a single ignition pulse | **no** | see above |

So the flip-flop is a drop-in as a **rail** in a simulation (anything that reads `.u` as a
213 Hz train: edge relays, veto neurons, rate-mode gates see the same statistics) — on the
connectome the rail has to be the excitatory proxy `p`, §below — but not as a **target**:
every ignition or reset path that touches `.u` or `.members` directly must go through the
two adapters, at 4 neurons per flip-flop for SET (3 until the lockstep fix, §Lockstep) and
4 shared neurons for CLEAR.

## Lockstep — what it is, when a SET train causes it, and the four-pulse fix (2026-09-16)

Everything here is the CPU simulator (`TorchSim`, float64, trace-identical to `RefSim`) on
a single proxied flip-flop with its set and clear chains, batched as the nodes of one run;
"mix B" is the register campaign's perturbation (`tests/test_ffregister.py`, `MIX_B`: 4 %
log-normal weight noise on every synapse, the chains' included; 0.2 mV threshold and bias
drift per neuron on top of the 58 mV; 5 Hz × 150 quanta stray input into every neuron),
applied with `run_ff_campaign`'s recipe. Numbers are in the contract under `lockstep`.

**What it is.** Both members spike within the synaptic delay (1.8 ms) of each other, so
neither inhibition arrives before the other's spike. Both are then inhibited together,
recover together, and repeat: period ~96 steps, ~105 Hz each, `p` and `q` running at the
same rate. The synchronous pair is an *attractor* — a member that spikes late receives the
other's inhibition earlier in its own cycle, more of it inside its 2.2 ms refractory hold,
so it recovers sooner and catches up — with a basin of roughly ±18 steps (the delay) of
relative phase. Once in it, the pair stays: from power-on without a pulse it is still locked
after 1 s at every (`u→v`, `v→u`) in 0.7–1.5× loop within ~0.3× of each other, with and
without stray input; a bias difference of up to +12 mV on `v` does nothing; and 500 of 500
mix-B-perturbed copies were still locked at 2 s. Only an asymmetry of ≥ 0.4× loop between
the two inhibitions resolves it (to the side with the stronger inhibition). Noise does not
break it toward a state. (A lock *started by a stray* — 1.5× ignite into a parked `u` at
nominal weights — also holds for 1 s, but it carries a phase offset and any asymmetry ≥ 0.1×
loop between the two inhibitions resolves it to the stronger side.)

**How a SET train causes it.** `u`'s park is ~15 mV × (`v→u` / loop) and one ignite pulse
climbs 12.6 mV, so at nominal weights the *second* pulse fires `u` (6.0–6.4 ms after the
first), `v` is inhibited before its next spike, and `u` leads by more than the delay from
then on. Once `v→u` is ~12 % strong, the park is deeper than two pulses reach and `u` first
fires on the *third* — the last — pulse. Its next tonic spike comes ~5.4 ms later (delayed
by `v`'s inhibition landing on it), `v` recovers from `u`'s single inhibition at about the
same moment, the two spikes fall within 18 steps of each other, and the pair is in the
basin with no pulse left to push `u` ahead. The maps:

| sweep (47 phases each, 3 × 1.0 ignite train) | locks |
|---|---|
| `u→v` × `v→u`, 0.70–1.30 in 0.05 steps (169 pairs, 7,943 runs) | only `v→u` ≥ 1.15× (28 % of phases at 1.125, 100 % at 1.175–1.2; ≥ 1.325 fails to SET instead), and below the bistability floor (`u→v` ≤ 0.75, `v→u` ≤ 0.95: the parked member escapes). `u→v` > `v→u` helps by 2–3× at a given `v→u`, no more |
| `u` bias × `v` bias, 57.0–59.0 mV in 0.2 mV steps | 0 locks, 0 failures anywhere |
| `u` threshold −45 to −43.5 mV, `u` bias 57–59 mV, one at a time | 0 locks, 0 failures for any train |
| train spacing 47 / 53 / 60 / 70 steps | the cliff stays at `v→u` = 1.10–1.125×: the pair's 4.7 ms rhythm is not what the train has to match |
| stray input on / off | no difference |

So the lever is the *length* (or strength) of the train against the park depth. The
largest `v→u` at which all 47 phases SET with no lock and no failure:

| SET train | clean up to `v→u` = | cost |
|---|---|---|
| 3 × 1.0 ignite (the old default) | 1.10× | — |
| 3 × 1.25 | 1.20× | fatter synapses on `u` |
| 3 × 1.5 | 1.30× | 1.5× ignite synapses on `u` (436 anatomical synapses per relay edge instead of 291) |
| **4 × 1.0 (the new default)** | **1.25×** | **one more relay neuron per set chain** |
| 4 × 1.25 | 1.40× | both |
| 5 × 1.0 | 1.375× | two more relays |

The CLEAR train's own margin, measured the same way against `u→v`: 3 × 1.5 loop (the
default) 1.40×, 4 × 1.5 loop (the register's reset train) 1.50×, the top of the sweep. The
clear side had a 40 % margin where the set side had 10 %; four pulses bring the set side to
25 %, five to parity.

**Pair-level candidates, rejected.** Stronger symmetric inhibition deepens the park as fast
as it deepens the hold: 1.2× loop locks half of all mix-B SETs, 1.5× fails to SET at all.
An asymmetry `u→v` = 1.2× moves the SET cliff only from 1.10 to 1.125× and drops the CLEAR
margin to 1.175× (0 SET locks of 8,000 at mix B, but 1 CLEAR lock). Self-inhibition of 0.3×
loop on both members locks half of all switches — it slows both alike and breaks nothing.
Bias asymmetry has no effect on the attractor.

**Before and after under mix B** (500 perturbations × 16 random phases of `v`, SET and
CLEAR twice each through the real chains, 5 Hz stray; the same perturbed copies for every
train):

| | SETs that lock | CLEARs that lock | spontaneous, 2 s hold |
|---|---|---|---|
| 3 × 1.0 (before) | 9 / 8,000, 15 / 8,000, 74 / 16,000 in three draws (0.1–0.5 %: a handful of pairs with `v→u` ~3σ strong lock at a third to all of their phases) | 0 / 16,000 | 0 / 500 in SET, 0 / 500 in CLEAR |
| **4 × 1.0 (after)** | **0 / 16,000** | 0 / 16,000 | 0 / 500, 0 / 500 |
| 4 × 1.0 at 1.5× mix B (6 % weights, 0.3 mV) | 0 / 16,000 (3 pulses: 589) | 0 / 16,000 (3 pulses: 1) | — |
| 4 × 1.0 at 2× mix B | 52 lock + 63 fail / 16,000 (3 pulses: 1,169 + 247; 5 pulses: 4 + 31) | ~100 / 16,000 for every SET train — the clear train's own 2× limit, and 4 clear pulses do not move it | — |

**What changed and what did not.** `SET_TRAIN_PULSES = 4` is the default of `set_pulse` and
`add_set_chain`; the set chain is a trigger and three relays (4 neurons per flip-flop, one
more than before); the clear train stays 3 × 1.5 loop. Measured at all 47 phases, the
contract is unchanged step for step: `u`'s first spike 6.0–6.4 ms after the first pulse
(the second pulse still fires it), `v`'s last ≤ 12.6 ms, `p` up 18.4–23.1 ms after the
first pulse and 10.5–14.0 after `v`'s last, `p`'s last spike ≤ 18.2 ms after the first CLEAR
pulse, the edge relay on `p` once per SET at 22.6–27.3 ms over the 47 phases (the old
"24.1–24.6" was three cycles' worth of phases), the rate 200–220 Hz in the first 50 ms (200
with three pulses: the fourth pulse adds one early spike at some phases) and 212.8 Hz from
then on, hold, noise margins and the 4 × 0.6 ignite all-phase minimum. The register's
`Q.b{i}r{r}` set chains grow by one relay each (+8 neurons on 225); its campaign line is not
re-run here. Test: `test_500_mix_b_flipflops_set_and_clear_without_lockstep` — 500
perturbations × 4 phases, bound 2 locks in 4,000 SETs (measured 0; the three-pulse train
locks 3 of the same 4,000, and every one of those is resolved by the next clear train).

## The excitatory proxy: reading a flip-flop on the fly (`add_flipflop(proxy=True)`, 2026-09-16)

**The problem.** `u` is an inhibitory neuron. Every reader in the protocol — an edge relay
(`u → edge`, `u → edge_inh`), a veto neuron (`u → veto`) — takes its rail through an
*excitatory* synapse, and under Dale's law an inhibitory host has no excitatory outputs. The
placement toy in `docs/h1_placement.md` (§Placing flip-flops) carries every flip-flop loop and
none of the 62 readout edges: on the connectome the flip-flop can be built but not read.

**The design.** A third neuron `p`, **excitatory**, on the same 58 mV bias as the pair, with one
synapse `v → p` of −1.0× loop and no outputs of its own. While CLEAR, `v`'s 213 Hz train parks
`p` ~15 mV below threshold exactly as it parks `u`; while SET, `v` is silent and `p` fires its
own tonic train, 212.9 Hz, period 47 steps, step for step with `u`. So `p` is `u` with the
right sign, and every reader connects to `ff.p` (`ff.rail`) instead of `ff.u` with no other
change. Writers are untouched: the set chain still pulses `u`, the clear chain still inhibits
`u` only. `p` feeds nothing back, so nothing that happens to `p` can move the pair's state.
`clear_proxy=True` adds `q`, the same neuron inhibited by `u`, as the CLEAR rail.

Sweep of the `v → p` strength (contract `proxy_readout.design.v_to_p_quanta_sweep`): 0.75×
loop leaks (`p` fires through `v`'s train), 0.8× is the floor, and above 1.0× the only effect
is a deeper park and a slower restart (1.5×: 20–23 ms after `v` stops; 2.0×: 26–29 ms). 1.0×
is the default: the pair's own strength, 25 % above the leak floor, the same margin the
loop itself has.

**What it costs: one park recovery each way.** A parked neuron does not fire the instant its
inhibitor stops; it climbs 15 mV with τ_m, which is why `v`'s first spike comes 9.5–14 ms after
a CLEAR. `p` pays that same delay on every edge (24 phases measured, every 2nd of 47):

| | through `u` | through `p` |
|---|---|---|
| rail's first spike after the first SET pulse (train / single 3× pulse) | 6.0–6.4 / 1.4–1.6 ms | 18.5–23.1 / 13.2–17.8 ms (10.5–14.0 after `v`'s last spike) |
| rail's last spike after the first CLEAR pulse | at once | 9.8–17.8 ms (`v` restarts at 9.5–14.0; `p` crosses once or twice before the park) |
| `add_edge_relay(source=rail)` | once per SET, ~11 ms after the pulse | **once per SET (3 of 3)**, 4.2 ms after `p`'s first spike, 24.1–24.6 ms after the pulse; never on the train |
| `add_veto_relay(vetoes=[rail])`: SET train must lead the driver's rise by | ≥ 10 ms | **≥ 22 ms** (`p` is up ~12 ms later) |
| re-drive after the first CLEAR pulse that fires the vetoed relay | 45 ms yes, 40 no | **60 ms yes, 55 no** — measured from the rail's last spike the 55 ms recovery rule is unchanged |
| set chain (latch → edge relay → chain) and clear chain | work | work; `p` up 34 ms after the source ignites (`u`: 19.5), down 25 ms after the clear trigger (`u`: 6.4) |
| kill train 4 × 0.75 loop into `u` | clears | clears; `p`'s last spike 22.9 ms after the first kill pulse |
| power-on without a pulse into the rail | `u` never fires (its pulse) | `p` fires once at 2.5 ms — before `v`'s first inhibition lands at 4.3 ms — and an edge relay on it fires at 6.7 ms. **`power_on_pulse` and `power_on_events` now pulse `p` as well as `u`**; `q` is left to start with `v` |
| hold, 2 s | 425 spikes | `p` 426 spikes SET / 0 CLEAR; `q` the reverse |
| resources | 2 neurons, 2 synapses, 2 biases | 3 / 3 / 3 (4 / 4 / 4 with `q`); all synapses inhibitory |

**Margins, from the reader's side.** A stray excitatory pulse into `p` while CLEAR: 1.1×
ignite (5,120 quanta) never fires it (24/24 phases); 1.15× fires `p` exactly once and the
edge relay on it once (1.8×: two `p` spikes), and the pair is untouched. That is the same
1.1× number as `u`'s own lockstep threshold (§Register), with a milder failure: one spurious
relay pulse instead of a permanent lockstep. A stray inhibitory pulse into `p` while SET only
pauses it — 7–9 ms at 1.0× loop, 11–16 ms at 2.25× (the firing member's own margin), 14–19 ms
at 3.0× — and it resumes at every phase; an edge relay on `p` never re-fires on the pause
(0/24 at 1.0–3.0×), and a veto relay vetoed by `p` stays held when its driver rises 0–27 ms
after a 3.0× stray (0/10). In the pair's lockstep third state `p` and `q` both run at ~105 Hz,
as `u` and `v` do: a reader sees the same fault either way.

**Supply on the connectome** (`bench/h1_inhpairs.py`, `docs/h1_inhpairs.json`,
`proxy_readout`). A proxy host is an excitatory neuron outside the pair that one member
inhibits with ≥ the loop threshold; the *strict* host also has an excitatory output ≥ the
threshold to a neuron outside the pair, so it can drive a relay. Counted over the driven
inh-inh pairs, then packed: the largest set of (pair, proxy) triples with no neuron in two
pairs and no proxy shared (a 3-set packing, solved exactly by MILP at every k_max; ties
broken towards the quietest proxies by a min-weight bipartite re-assignment):

| k_max | threshold | driven inh-inh pairs | with a proxy, loose / strict | candidate triples | proxies per pair | disjoint triples, loose / strict (disjoint-pair bound) | proxies' mean external input synapses (pair members') | proxy superclasses, strict |
|---|---|---|---|---|---|---|---|---|
| 4 | 57 | 729 | 700 / 694 | 16,962 / 12,256 | 24.2 / 17.7 | **358 / 353** (358 / 353) | 1,782.1 / 2,007.1 (6,758.7) | cb_intrinsic 156, vnc_intrinsic 120, ascending_neuron 23, descending_neuron 18 |
| 8 | 29 | 3,970 | 3,448 / 3,402 | 139,276 / 122,097 | 40.4 / 35.9 | **1,030 / 997** (1,083 / 1,056) | 956.7 / 1,028.0 (4,794.5) | cb_intrinsic 431, vnc_intrinsic 295, ol_intrinsic 100, visual_projection 61 |
| 16 | 15 | 12,666 | 12,630 / 12,619 | 727,871 / 710,356 | 57.6 / 56.3 | **3,712 / 3,703** (3,786 / 3,778) | 497.9 / 506.6 (2,218.3) | ol_intrinsic 1704, cb_intrinsic 918, vnc_intrinsic 532, visual_projection 224 |

Proxies are not the bottleneck: a driven pair has 18–58 candidates on average, and the
packing reaches the disjoint-pair bound at k_max 4 (358 = the pool the placer already uses)
and 94–98 % of it at 8 and 16. Against the earlier readout-by-silence count (358 / 1,105 /
3,717 disjoint driven pairs with an inhibitory output each) the strict proxy supply is
353 / 997 / 3,703. The chosen proxies take 3–5× fewer input synapses from the rest of the
brain than the pair members (they are picked for that among equals), so the readout neuron
is the quiet one of the three. Runtime: ~10 min for the three thresholds, 2 GB, almost all
of it the two k_max 16 MILPs (`MILP_TIME_LIMIT_S`).

**Not done here.** Placing the proxy: `place_netlist`'s `ffpair` motif is two biased
inhibitory neurons in mutual inhibition; `p` is a biased excitatory neuron with one inhibitory
input, which the placer will treat as a single. A `ffproxy` motif (pair + proxy as a unit on a
packed triple) is the next placement step. The register reads through `p` since the same day
(§Reading through the proxies below), and its mix-B campaign is the first weight-noise
measurement through `p`.

## Not done

- Weight-noise campaigns (the margins above are at nominal weights; the bistability floor
  is 20–25 % below the default inhibition).
- A rate-mode gate reading `ff.u` was not run; the rail's ISI is the latch's (47 steps),
  so nothing is expected to differ.
- The simulators still take `bias` from the caller: `RefSim(topo, params,
  bias=topo.sim_bias())`. Making them read `topo.bias` themselves is a two-line change in
  `ref64.py` and `lif_torch.py` (`bias=None` → `topo.sim_bias(bias)`), outside this task's
  owned paths.

## The register on flip-flops (`build_channel(storage="flipflop")`, 2026-09-16)

`drosophilos/protocol/handshake.py`; contract `docs/contracts/ffregister.yaml`; tests
`tests/test_ffregister.py`. The consumer register's eight rail latches become flip-flops;
everything else is the frozen 4-bit channel (watchdog 55 hops). RefSim, no weight noise,
the contract's six words unless stated.

### How it is wired

- **SET**: the producer's data edge relay fires one ignite pulse into `Q.b{i}r{r}.set`, the
  trigger of a per-rail `add_set_chain` (trigger + 2 relays at the time of this measurement;
  trigger + 3 relays since the lockstep fix, §Lockstep), which puts three (now four) ignite
  pulses 5.3 ms apart into `u`. `Register.rail_inputs[i][r]` names that trigger; for a latch register
  it is the rail's own `u`, so `build_channel` wires `relay -> rail_inputs` in both cases.
- **CLEAR**: no second controller. The register's reset controller `inh` (which already fires
  once per reset relay, 4 times) gets one extra synapse per flip-flop, −1.5× loop into `u`
  only (`flipflop.connect_clear`): 4 × 1.5× loop, above the 3 × 1.0× all-phase minimum, and
  nothing into `v`. Clears at all 47 phases; the SET rails' last `u` spike is 5.9 ms (max
  6.3) after the reset trigger, against 8.5 ms (max 11.6) for the latch register's rails.
- **Reading**: every reader of a rail — the valid ORs, the fault ANDs, the decode taps
  (`Register.rail_taps`) — takes the rail's excitatory proxy `p` (`celement.rail_of`), never
  `u`; §Reading through the proxies below. (The first build, kept as `rail_proxy=False`, read
  `.u`: a 213 Hz rail is a 213 Hz rail in a simulation, but not on a host.) The completion tree
  reads the valid latches, which are latches.
- **Flags stay latches** (valid, tree, completion, fault). Measured with them as flip-flops
  too (`flag_storage="flipflop"`, every gate→state path through a set chain): 249 neurons
  instead of 225, accept 154.3 ms mean / 159.0 max over 50 random words against 154.1 /
  155.6 — 6 more neurons per bit for no gain in latency or margin, and the supply argument
  (§Why) is about the rail storage, not the flags. `flag_storage` is kept as an option.
- **Power-on**: `Channel.power_on_events()` (or `flipflop.power_on_events(net, drive)`) is
  the list of (step, neuron, quanta) a harness injects with the image — one −1.0× loop pulse
  into every flip-flop's `u`. The tests do it through `RefSim.add_events` before
  `run_transactions`; a campaign does it through its per-node event list.
- The default build is untouched: `build_channel`, the staged register and the ALU build
  neuron-for-neuron and synapse-for-synapse identical netlists to the previous commit
  (compared: roles, src, dst, quanta, delay; `Topology.bias` stays `None`).

### Side by side (4-bit channel, consumer register through the four-phase handshake)

The flip-flop column is the first build, read at `u` (`rail_proxy=False`, the top of
`ffregister.yaml`); the shipped build reads through `p` and is measured against it in
§Reading through the proxies below.

| | latch register (`channel_4bit.yaml`) | flip-flop register, read at `u` (`ffregister.yaml`) |
|---|---|---|
| neurons, whole channel | 201 | 225 (+6 per bit) |
| rail storage per bit | 4 (two 2-neuron latches) | 10 (two flip-flops + two 3-neuron set chains) |
| consumer excl. reset/READY chains | 62 (15.5 per bit) | 86 (21.5 per bit) |
| clear path | reset `inh` → both members, 4 × 0.75× loop | the same `inh` → `u` only, 4 × 1.5× loop; 0 extra neurons |
| accept latency, mean / max | 155.5 / 156.8 ms | 152.3 / 155.1 ms |
| initiation interval, mean / max | 333.5 / 334.8 ms | 330.3 / 333.1 ms |
| reset trigger → last rail `u` spike, mean / max (50 random words) | 8.5 / 11.6 ms | 6.3 / 7.0 ms (5.9 / 6.3 on the contract's six) |
| READY after reset trigger | ~85 ms (15-hop chain) | ~85 ms (same chain) |
| spikes per transaction | 1,951 | 2,103 (+8 %: the CLEAR-side `v` members run whenever a rail is not set) |
| idle cost | 0 | 8 × 213 Hz (one member of every pair always fires) |
| retention | (latch: 2 s in `flipflop.yaml`'s comparison) | 2 s, decoded valid, 212.5 Hz, period 47 steps |
| arrival jitter | bits up to 40 ms apart | bits up to 100 ms apart, 5/5 correct, accept after the last bit |
| stray excitatory pulse, silent rail | ignites at 1.1× need (2,845 q) | `u`: unchanged to 1.1× ignite (5,120 q); set trigger: fires at 0.6× ignite (2,793 q) — see below |
| stray inhibitory pulse, set rail | survives 2.0× loop | survives 2.25× loop; 2.5× clears 4/12 phases |
| errors | 0 in 1e5 mix-B (1 late_activity) | 0 in 200 (upper 95 % 1.5 %); 48 fail-stop, 0 wrong in 10,000 at mix B |

The flip-flop register is 3 ms faster to accept, not slower. Traced on one bit (word 2 of
the run, times after the load): the data relay fires at 6.6 ms in both builds; the latch's
`u` first spikes at 11.6 ms but its train ramps up (first ISIs 9.3, 6.5, 5.6, 5.2 ms), while
the flip-flop's `u` first spikes at 19.5 ms (trigger 10.8 ms, then the set train) and runs at
full rate from the first spike (ISIs 4.6, 3.9, 4.3, 4.5 ms). The rate-mode OR feeding the
valid latch therefore fires at 37.9 ms instead of 41.1, and the 3 ms carry through the two
tree levels (c0 at 101.4 vs 104.5 ms, c1 = ACCEPT at 153.6 vs 156.7). The 150 ms is the
three rate-mode gate levels either way.

### Reading through the proxies (`rail_proxy=True`, the default, 2026-09-16)

`build_channel(storage="flipflop")` now builds every rail flip-flop with `proxy=True` and
every reader of a rail takes `p`: the valid ORs and fault ANDs (`add_register`), the decode
taps (`Register.rail_taps`, `celement.rail_of`), and the `hold_from` of a flag's ignition relay
were a flag ever a proxied flip-flop. The completion tree reads the valid latches, which stay
latches, so the register's only read stage through `p` is rails → valid OR. Writers are
unchanged: the set chains pulse `u`, the reset controller's `inh` clears through `u` (an
inhibitory synapse into an inhibitory neuron: sign-correct), and the power-on pulse goes into
`u` and `p`. No veto in the channel reads a consumer rail — the watchdog reads the
*producer's* latch rails and READY has no veto — so `p`'s 22 ms veto lead is not exercised
and no delay chain was widened. The default latch build is unchanged neuron for neuron and
synapse for synapse (compared: roles, src, dst, quanta, delay, bias, groups).

| | read at `u` (first build) | read at `p` (shipped) |
|---|---|---|
| neurons, whole channel / consumer excl. reset and READY / rail storage per bit | 225 / 86 (21.5 per bit) / 10 | 233 / 94 (23.5 per bit) / 12 (+1 per rail) |
| synapses (inhibitory) / biased neurons | 427 (153) / 16 | 435 (161) / 24 |
| accept latency, mean / max, contract's six words | 152.3 / 155.1 ms | 166.4 / 168.1 ms (**+14.1**) |
| accept latency, mean / max, 200 random words (fast test) | 154.2 / 156.8 ms | 167.7 / 170.0 ms (+13.5) |
| initiation interval, mean / max, six words (200 words) | 330.3 / 333.1 (332.2 / 334.8) ms | 344.4 / 346.1 (345.7 / 348.0) ms (+14.1) |
| rail's first spike after the set trigger's pulse | `u` 6.0–6.5 ms | `p` 18.5–23.1 ms |
| reset trigger → last `u` spike / → last spike the readers see, mean / max | 5.9 / 6.3 ms, the same | 4.9 / 6.2 ms; `p` 22.1 / 23.3 ms (200 words: 23.8 / 27.2) |
| READY after the reset trigger | 84.8 ms | 84.8 ms (`p` silent ~60 ms before it) |
| spikes per transaction | 2,103 | 2,383 (+13 %: four `p` trains per word, and the longer cycle) |
| stray excitatory pulse into a parked rail's `p`, one proxy | — | 1.1× ignite never fires `p`; 1.15× / 1.8× / 4.0× fire it 1 / 2 / 3 times, pair untouched, and **nothing downstream fires** (no OR spike, no valid / tree / completion latch, no fault gate): one `p` spike is 766 q into an OR with a 2,586 q single need |
| the same stray into both proxies of a bit at once | — | 1.15× nothing; 1.8× one OR spike → a false *valid* on that bit; a false *completion* needs it on every bit at once (1.8× into all eight does complete) |
| 200 random transfers | 200 / 200, 0 errors | 200 / 200, 0 errors (upper 95 % 1.5 %) |
| mix B, 10,000 transfers | 48 non-ok (43 timeout, 4 no ACCEPT, 1 no READY), **0 wrong** | 37 non-ok (21 timeout, 10 no ACCEPT, 1 no CLEARED, 3 no READY, **2 wrong after a fail-stop**), 0 wrong on a clean transfer; accept 171.2 mean / p99 194.8 / max 440.0 ms (u: 154.3 / 170.9 / 389.5) |

The +14 ms is one park recovery — `p` climbs 15 mV with τ_m after `v` stops — paid once,
at the rails → valid OR stage; every later stage reads a latch. On the CLEAR side `p` fires
for 22–27 ms after the reset trigger instead of 6 ms, well inside the 85 ms to READY.

**The campaign, and the two wrong values.** The fail-stop rate fell from 0.48 % to 0.37 %
(different random draws: the 8 extra neurons shift every perturbation sample, so the two runs
are not node for node). Every non-ok in chunk 0 (30 of 4,000, all in four nodes) was traced
to one mechanism, present in both builds: a perturbed pair lands in **lockstep** on its SET
train (`u`, `v` and `p` all at ~106 Hz), its valid OR is marginal at that rate (766 q per
spike at 106 Hz ≈ 1.0× `rate_need`) and ignites 200+ ms late, and the watchdog times out.
That is a fail-stop in both builds, and after it READY comes ~490 ms after the load, past
the harness's next load at its fixed 450 ms period (the protocol's upstream would wait). The
u-readout build recovered from every one of its 43 (the SET trains 56–62 ms after the reset
trigger still take). The two wrong values are the transfer after such a timeout, in two
nodes, by two routes:

- **node 197, a double reset, proxy timing.** The late completion reaches the producer's
  reset trigger 54 ms after the timeout's FAULT-ACCEPT (through `u` the same event came at
  46 ms in the u-build's node 188 and was swallowed), past the trigger's ~50 ms edge re-arm,
  so producer and consumer **reset twice**; the second consumer reset train lands ~456 ms
  after the load, on top of the next load, the SET trains fail against it, the producer's
  rails are never reset (its data edge relays fire only on a rise), and a stale partial word
  completes three transfers later.
- **node 175, a second lockstep event, timing-independent.** The next transfer (loaded before
  READY) put the same pair into lockstep again; it completed at 440 ms with no watchdog
  timeout, its producer reset came after the following load, and the stale rails made the
  word. Through `u` the completion would have been ~427 ms: the same outcome.

`run_ff_campaign` now files a wrong value whose node's previous transfer was non-ok as
`wrong_value_after_failstop` and the slow test asserts only the fresh count is zero. So the
direction: fewer fail-stops, and the timeout class now has a tail that a fixed-period harness
turns into two wrong words in 10,000 (one of them from the proxy's 12 ms). The fix is
upstream of the readout — keep a perturbed pair out of lockstep on SET (done: the four-pulse
train, §Lockstep; ±4 % weights and ±0.2 mV bias were enough to get there at some phases with
three) and lengthen the reset trigger's re-arm past the proxy's lag — and neither is a proxy
change. **Re-run on the four-pulse train (Juno 408918, the same 10,000 mix-B transfers, seed 0):
0 non-ok and 0 wrong** — no timeout, no missing ACCEPT, CLEARED or READY, no cascade (upper
95 % limit 3.0 × 10⁻⁴ on each). Every one of the 37 fail-stops was the lockstep entry, and the
reset trigger's re-arm was never reached without one. The contract keeps both figures.

`lib/contracts.measure_contract` classes a `Q.b*.p` role as control, not latch (its rule is
`.u`/`.v`), so the by-class spike split in the contract moves the proxies' spikes from latch to
control; the totals are right.

### What the measurement changed about the noise-margin claim above

The table in §Noise margins says the silent member "survives 1.8× ignite". It does not switch
to SET, which is what that sweep counted — but a single excitatory pulse of **≥ 1.15× ignite**
into a parked member drops the pair into **lockstep** (both members at ~105 Hz, the stable
third state of §Power-on) at every one of 47 phases, and it stays there. To every reader of
`.u` that is a 105 Hz train on a rail that should be silent: on a register, a fault (both
rails of the bit active) at the next decode. 1.1× ignite (5,120 quanta) leaves it CLEAR at
every phase; 2.0× SETs it at about half the phases and locks the rest. The next reset heals
it (the `u`-only train resolves lockstep to CLEAR, tested), so a stray costs one transaction.

The register's real stray margin is the **set trigger**, an ordinary neuron: 0.55× ignite
(2,560 quanta, ~1.0× single need) never fires it, 0.6× fires the chain and SETs the rail at
every phase. That is the latch's own margin (its `u` ignites at 1.1× need). So the register
is no more stray-proof than the latch register at its input, ~1.8× more at the storage
itself, and at the campaigns' 150-quanta strays (0.03× ignite) neither margin is touched.

### Harness limits (outside this task's paths)

- `protocol/run.py`'s fault specs (`corrupt`, `duplicate`, `late_opposite`, `stale`) inject
  one ignite pulse into `Q.rails[i][r].u`: on a flip-flop register they do nothing (the pulse
  is below 1.15×). A corruption test must target `Q.rail_inputs[i][r]`.
- `lib/campaign.run_campaign` hands `TorchSim` its own bias drift, which replaces the
  topology's flip-flop biases (`Topology.sim_bias` returns the caller's array when given), and
  files every consumer spike after READY as `late_activity`, which the `v` trains always are.
  `tests/test_ffregister.py::run_ff_campaign` is that loop with the drift added on top of
  `topo.bias`, the `v` members excluded from the late-activity set and the power-on pulses
  in each node's events; the 20-transfer smoke of it runs in the fast suite.
- `run_transactions` rebuilds the spike trace once per transaction (quadratic in the run
  length), and the `v` trains double the trace: 200 transfers in one run took 60 s, four
  runs of 50 take 23 s, which is how the fast test is split.
- The campaign harness loads on a fixed 450 ms period whatever the channel says; after a
  timeout the next load lands before READY. Under the protocol that load waits. This is what
  turns the proxied build's double reset into wrong values (§Reading through the proxies).
- The producer stays a latch register: every harness loads it with one pulse into
  `P.rails[i][r].u`. `add_register(storage="flipflop")` builds a flip-flop producer too
  (`build_channel(producer_storage=...)`), loaded through `P.rail_inputs`; not measured.
