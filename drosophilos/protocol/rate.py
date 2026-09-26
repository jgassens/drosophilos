"""Condition rate-mode inputs before adding their currents.

A fast latch must not count as more than one logical input. A strong feed-forward
readout runs close to the model's refractory limit across the measured latch-rate
distribution. Share that readout by source, then calibrate every gate to its rate.
This is a neural circuit, with ordinary noisy synapses, rather than simulator-side
clipping or knowledge of which rail is logically live.
"""

from ..lib.netlist import Drive, Netlist


def rate_tap(net: Netlist, drive: Drive, source: int) -> int:
    """Shared feed-forward saturator. It has no feedback into the storage latch."""
    if not drive.rate_robust:
        return source
    if source not in net.rate_readouts:
        readout = net.neuron(f"{net.roles[source]}.rate")
        net.synapse(source, readout, 8 * drive.loop)
        net.rate_readouts[source] = readout
    return net.rate_readouts[source]


def condition_rate_gate(net: Netlist, drive: Drive, gate: int, *, gain: float = 2.0) -> None:
    """Condition a gate's weak train inputs; retain pulse inputs and inhibitory vetoes.

    Call after installing the gate's positive inputs, before wiring its output. The
    shared ignition helper also calls this for arithmetic majority gates. All rate
    inputs in these primitives are <= or_in; single-pulse inputs are >= single_need/2.
    Biased one-shot qualifiers use gain=1: V_reset is fixed by the simulator, so
    doubling their pulse and gap together would not preserve single-spike behavior.
    """
    if not drive.rate_robust or gate in net.rate_gates:
        return
    edges = [k for k, (target, q) in enumerate(zip(net.dst, net.quanta))
             if target == gate and 0 < q <= drive.or_in]
    if not edges:
        return
    # Eight loop pulses per source spike saturate the readout near n_ref + 3 steps.
    # The measured range, including -10% input weights, is checked in test_fast_latch.
    period = net.params.n_ref + 3
    for k in edges:
        net.src[k] = rate_tap(net, drive, net.src[k])
        net.quanta[k] = int(round(net.quanta[k] * period / drive.loop_period_steps))
    # Preserve signal fractions, but halve threshold/bias drift and stray input relative
    # to the gap. scale_inputs also scales resets/vetoes that are attached afterwards.
    if gain != 1.0:
        net.scale_inputs(gate, gain)
    net.rate_gates.add(gate)
