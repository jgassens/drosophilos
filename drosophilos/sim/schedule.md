# Simulation model and within-step order (normative)

This file is the contract for every simulator backend in DrosophilOS. `ref64.py` is the
executable form of it; every other backend is certified against `ref64.py`. Anything not
written here is not part of the model.

## 1. State per neuron *i* (per node)

| symbol | meaning | unit | type |
|---|---|---|---|
| `V_i` | membrane voltage | mV | float64 |
| `g_i` | synaptic drive (exponential kernel), summed over all inputs | mV | float64 |
| `r_i` | remaining steps of integration hold (refractory) | steps | int32 |
| `b_i` | bias (tonic) drive | mV | float64, parameter |
| `silenced_i` | never integrates, never spikes, delivers nothing | – | bool, parameter |

## 2. Dynamics between events

For a non-refractory neuron, between spike arrivals:

```
tau_m · dV/dt = (E_L + b_i − V) + g
tau_s · dg/dt = −g
```

`g` jumps by `w_ij` (mV) at the moment a spike from *j* is delivered to *i* (section 4).
All parameters may vary per neuron (Profile 2/3 overrides); defaults are the Shiu et al.
2024 values: `E_L = −52 mV`, `V_th = −45 mV`, `V_reset = −52 mV`, `tau_m = 20 ms`,
`tau_s = 5 ms`, `t_ref = 2.2 ms`, weight per anatomical synapse `w_syn = 0.275 mV`,
default delay `1.8 ms`.

## 3. Time discretisation

`dt = 0.1 ms`. Step `s` covers `[s·dt, (s+1)·dt)`. Within a step **no spike arrives**:
all deliveries happen at step boundaries (section 5). Therefore the linear system above is
integrated **exactly** over one step, per neuron, with precomputed constants

```
a = exp(−dt/tau_m)
c = exp(−dt/tau_s)
k = tau_s/(tau_s − tau_m) · (c − a)          if tau_s ≠ tau_m
k = (dt/tau_m) · a                            if tau_s = tau_m
```

and the update

```
V' = E_L + b + (V − E_L − b)·a + g·k
g' = g·c
```

This is the same closed-form solution Brian2 uses for these equations with its default
`exact` (linear) method, which is why spike traces can be compared one-to-one.

Timestep refinement (`dt → dt/5`) must leave **decoded transactions** unchanged for every
certified primitive; a primitive that only works at one `dt` is exploiting the integrator.

**Implication for circuit design (measured, default parameters).** `k ≈ 0.00494`, so a jump
of `g` moves `V` by only `0.49 %` of `g` in the first step; the membrane excursion peaks
`≈ 9 ms` after arrival at `≈ 0.157·g`. Crossing the `7 mV` threshold gap needs
`g ≳ 45 mV` (`≈ 160` anatomical synapses arriving together) with `≈ 5–9 ms` of latency, or
`g ≳ 1,420 mV` (`≈ 5,150` synapses, `82,400` quanta) to fire on the very next step.
Gate latencies in this substrate are therefore milliseconds, not steps, unless Profile 2/3
weight edits raise the drive. The latency budget of the whole machine follows from this.

## 4. Synaptic weights are integers

A synapse carries an integer number of **quanta**, `q_ij ∈ int32`. One anatomical synapse
is `QUANTA_PER_SYNAPSE = 16` quanta; the voltage jump is

```
w_ij = q_ij · w_unit,   w_unit = w_syn / QUANTA_PER_SYNAPSE = 0.0171875 mV
```

Sign comes from the presynaptic neurotransmitter (GABA, glutamate → negative; all others
positive), applied when the connectome is loaded, and may be further changed only as the
active profile permits. Delivery sums quanta in **int64** per target per step, then converts
once: `g_i += w_unit · Σ q`. Integer sums do not depend on reduction order, so spike-level
results are bit-identical across CPU threads, GPU atomics, and batch sizes. Fractional
(bounded) weight edits are expressed as quanta; `1/16` of a synapse is the resolution.

## 5. Within-step order (fixed; equals Brian2's default slot order)

For step `s`, in this order, for every node in the batch:

1. **integrate** — for each neuron with `r_i == 0` and not silenced: apply section 3.
   For each neuron with `r_i > 0`: `V_i := V_reset` (held), `g_i := g_i·c`, `r_i −= 1`.
   Silenced neurons: `V_i := E_L`, `g_i := 0`.
2. **threshold** — `spike_i := (r_i == 0) and not silenced_i and (V_i > V_th)`
   (strict `>`, as in the Shiu model). Record `(s, i)` for every spiking neuron.
3. **deliver** — (a) for every neuron *j* that spiked in this step and every outgoing synapse
   `(j → i, q_ij, d_ij)`, enqueue `q_ij` for target *i* at step `s + d_ij` (`d_ij ≥ 0`
   integer steps, `d_ij ≤ D_MAX = 100`); (b) for every external port event `(step = s,
   target i, quanta q)`, enqueue at step `s`; (c) apply everything enqueued **for step s**:
   `g_i += w_unit · Σ q` (int64 sum). A delivery for step `s` therefore influences the
   integration of step `s + 1`, never step `s`. A zero-delay synapse from a neuron spiking
   at `s` is applied at `s` (after threshold) and felt at `s + 1`.
4. **reset** — for every neuron that spiked in this step: `V_i := V_reset`,
   `r_i := n_ref − 1` where `n_ref = round(t_ref/dt)` (22 for 2.2 ms). Integration is
   suspended for steps `s+1 … s+n_ref−1` and resumes at step `s + n_ref`, which is exactly
   when Brian2's `(unless refractory)` with `refractory = t_ref` resumes.
5. **observe** — optional recording of `V`, `g`, `r` for selected neurons, after reset.

Silenced neurons are removed from step 3(a) as sources and from steps 1–2 as targets of
integration; deliveries to them are discarded.

## 6. External ports

An **input port** is a set of neurons that may receive external events `(step, neuron,
quanta)`; the events are the only channel from the host into the substrate. An **output
port** is a set of neurons whose spikes the host may observe. Both sets are declared in the
program image manifest; a backend must refuse events to undeclared neurons and must not
expose spikes of undeclared neurons to the transport (recording for diagnostics is separate
and labeled).

## 7. Spike-event trace (the comparison object)

A run produces the sequence of `(step, neuron_id)` pairs sorted by `(step, neuron_id)`.
Two backends **agree** at the spike level when these sequences are identical. Trajectory
agreement (`V`, `g` within a tolerance) and transaction agreement (decoded architectural
values identical) are the other two levels and are reported separately.

## 8. Snapshots

A snapshot holds, per node: `V`, `g`, `r`, the delay queue (all pending `(step, target,
quanta)` deliveries), pending external events, RNG state of any perturbation source,
weights/overrides in force, and the current step. Restoring and continuing must reproduce
the spike trace of an uninterrupted run exactly. Snapshots are a debugging and
resume-the-HPC-job tool; they are **not** architectural checkpoints and support no
recovery claim.

## 9. Batching

Nodes form a batch dimension. Topology (`CSR` of `(j → i, d_ij)` and base quanta) is shared
by all nodes; per-node arrays hold state, parameter overrides (`V_th`, `b`, `silenced`, quanta
overrides), and Profile 3 added synapses. Results for a node must not depend on which other
nodes share the batch (tested).
