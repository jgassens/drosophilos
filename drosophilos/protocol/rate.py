"""Condition rate-mode inputs before adding their currents.

A fast latch must not count as more than one logical input. A strong feed-forward
readout runs close to the model's refractory limit across the measured latch-rate
distribution. Share that readout by source, then calibrate every gate to its rate.
Its source's inhibitory inputs also clear the readout's stored current, including
kill/reset trains installed later. This is a neural circuit with ordinary noisy
synapses, not simulator-side clipping or knowledge of which rail is logically live.
"""

from ..lib.netlist import Drive, Netlist


def rate_tap(net: Netlist, drive: Drive, source: int) -> int:
    """Shared saturator in the source's reset domain, without feedback into storage."""
    if not drive.rate_robust:
        return source
    if source not in net.rate_readouts:
        readout = net.neuron(f"{net.roles[source]}.rate")
        net.synapse(source, readout, 8 * drive.loop)
        net.rate_readouts[source] = readout
        # Storage has two members and stops when either misses its next spike; the
        # saturated reader can still have several spikes' worth of current then.
        # Twice its input gain drains that reserve within the measured 5.6 ms fall
        # budget, including in-flight spikes and the independent +/-3-sigma corners
        # in test_fast_latch. These extra edges never inhibit the source itself.
        net.mirror_inhibition(source, readout, 16)
    return net.rate_readouts[source]


def condition_rate_gate(net: Netlist, drive: Drive, gate: int, *, gain: float = 2.0,
                        clear_with_inputs: bool = False) -> None:
    """Condition a gate's weak train inputs; retain pulse inputs and inhibitory vetoes.

    Call after installing the gate's positive inputs, before wiring its output. The
    shared ignition helper also calls this for arithmetic majority gates. All rate
    inputs in these primitives are <= or_in; single-pulse inputs are >= single_need/2.
    Biased one-shot qualifiers use gain=1: V_reset is fixed by the simulator, so
    doubling their pulse and gap together would not preserve single-spike behavior.
    Coincidence qualifiers use clear_with_inputs: a consumed required rail must also
    discharge the qualifier's own current/voltage, not just stop its readout train.
    """
    if not drive.rate_robust or gate in net.rate_gates:
        return
    edges = [k for k in net.incoming[gate] if 0 < net.quanta[k] <= drive.or_in]
    if not edges:
        return
    # Eight loop pulses per source spike saturate the readout near n_ref + 3 steps.
    # The measured range, including -10% input weights, is checked in test_fast_latch.
    period = net.params.n_ref + 3
    sources = [net.src[k] for k in edges]
    for k in edges:
        net.src[k] = rate_tap(net, drive, net.src[k])
        net.quanta[k] = int(round(net.quanta[k] * period / drive.loop_period_steps))
    # Preserve signal fractions, but halve threshold/bias drift and stray input relative
    # to the gap. scale_inputs also scales resets/vetoes that are attached afterwards.
    if gain != 1.0:
        net.scale_inputs(gate, gain)
    if clear_with_inputs:
        for source in dict.fromkeys(sources):
            net.mirror_inhibition(source, gate, 1.0)
    net.rate_gates.add(gate)
