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

Defaults (`set_pulse`, `clear_pulse`) carry a margin: SET = three standard ignite pulses
(or one 3× pulse); CLEAR = three standard reset pulses at −1.5× loop (or one 5× loop
pulse). All switch at every one of the 47 phases.

Switching time: SET train — `u`'s first spike 6.0–6.5 ms after the first pulse (it is the
second pulse that fires it), `v`'s last spike ≤ 12.6 ms; SET single pulse — 1.4–1.7 ms and
≤ 6.8 ms; CLEAR — `u` stops at once (no spike after the first pulse lands), `v`'s first
spike 9.4–14.0 ms later (it climbs 15 mV with τ_m). No doublet after SET: first ISI
3.7–4.8 ms, 200–220 Hz over the first 50 ms, 212.8 Hz from then on.

Hold: 2 s with no input in both states (425 spikes each, 212.5 Hz).

## Noise margins (compared with the excitatory latch)

| stray pulse | flip-flop | excitatory latch |
|---|---|---|
| excitatory, into the silent member / latch | survives 1.8× ignite (8,379 quanta, ~22.7 mV); 2.0× flips 33 % of phases | ignites at 1.1× need (2,845 quanta, ~7.7 mV) |
| inhibitory, into the firing member / one latch member | survives 2.25× loop (8,147 quanta); 3.0× flips 71 % | survives 2.0× loop; 3.0× kills 79 % |

The flip-flop is ~3× harder to set by a stray excitatory pulse than the latch is to ignite,
and about as hard to knock down.

## Power-on and lockstep — the part that does not take care of itself

Both members start at rest, so with nothing else they reach threshold together at 2.5 ms,
inhibit each other together, recover together, and keep firing **in lockstep** (~100 Hz
each at 1.0× loop) for ever. This is a stable third state. Giving `v` a larger bias does
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
| `add_veto_relay(target=ff)` and `celement._ignite_from` | **no** | their single ignite pulse into `.u` never sets it (0 of 47). Fix: `add_set_chain` — a 3-relay chain that turns one ignite pulse into three (measured end to end: latch train → edge relay → chain → SET) |
| `add_reset` on `ff.members` | **no** | a train into **both** members (4 × 0.75 or 3 × 1.5 loop) leaves the flip-flop where it was (SET stays SET, 12 of 12 phases). Fix: `add_clear_chain` — the same controller aimed at `u` only (measured: trigger → 3 pulses → CLEAR) |
| kill train into `u` only | **yes** | 3 × 1.0 loop at every phase; the standard 4 × 0.75 loop too, with ~7 % margin |
| a single ignition pulse | **no** | see above |

So the flip-flop is a drop-in as a **rail** (anything that reads `.u` as a 213 Hz train:
edge relays, veto neurons, rate-mode gates see the same statistics) but not as a **target**:
every ignition or reset path that touches `.u` or `.members` directly must go through the
two adapters, at 3 neurons per flip-flop for SET and 4 shared neurons for CLEAR.

## Not done

- Weight-noise campaigns (the margins above are at nominal weights; the bistability floor
  is 20–25 % below the default inhibition).
- A rate-mode gate reading `ff.u` was not run; the rail's ISI is the latch's (47 steps),
  so nothing is expected to differ.
- The simulators still take `bias` from the caller: `RefSim(topo, params,
  bias=topo.sim_bias())`. Making them read `topo.bias` themselves is a two-line change in
  `ref64.py` and `lif_torch.py` (`bias=None` → `topo.sim_bias(bias)`), outside this task's
  owned paths.
