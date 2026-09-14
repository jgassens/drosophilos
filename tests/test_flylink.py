"""Stage C, Demo 1 shape: node A reads an input, computes, branches, and stores a result to
its output port; the transport carries the port word's rail rises to node B's input port,
whose completion interrupts B; B's handler loads it, computes and emits the pixel."""

from drosophilos.cluster.flylink import run_linked, word_link
from drosophilos.lib.control import build_machine, load_image, reference_run
from drosophilos.protocol.token import decode_at
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim

PARAMS = Params()


def test_two_nodes_message_and_pixel():
    # node A: y = x + 3; if (y & 1) y = y ^ 6; port_out <- y   (port_out = word 7, input x at word 0)
    progA = [("LOAD", 0), ("ADD", 3), ("STORE", 1), ("AND", 1), ("JZ", 6), ("LOAD", 1), ("XOR", 6), ("STORE", 1), ("LOAD", 1), ("STORE", 7), ("HALT", 10)]
    # node B: idle loop 0<->1; handler at 4: pixel = port_in + 1 -> port_out (word 7)
    progB = [("MOV", 0), ("JMP", 2), ("JMP", 1), ("HALT", 3), ("LOAD", 6), ("ADD", 1), ("STORE", 7), ("IRET", 0)]
    x = 5
    refA = reference_run(progA, 4, 4, {0: x}, max_steps=40)
    sent = refA["dmem"][7]
    pixel = (sent + 1) & 15
    mA = build_machine(PARAMS, n=4, n_prog=16, n_data=8, port_out_word=7)
    mB = build_machine(PARAMS, n=4, n_prog=8, n_data=8, handler_pc=4, port_in_word=6, port_out_word=7)
    simA, simB = RefSim(mA.net.topology(), PARAMS), RefSim(mB.net.topology(), PARAMS)
    load_image(simA, mA, progA, {0: x})
    load_image(simB, mB, progB, {})
    link = word_link(mA, mB, delay_steps=200)  # 20 ms modeled transport delay
    run_linked([simA, simB], [link], int(16000 / PARAMS.dt))
    # A sent the right word; B received it and emitted the pixel
    outA = mA.dmem.words[7]
    ev = simA.trace.events
    stepsA = ev["step"][ev["neuron"] == outA.completion.u]
    assert len(stepsA) and decode_at(simA.trace, outA.rail_taps, int(stepsA[-1]), 94)[0] == sent
    assert len(link.log) == 4, link.log  # one event per bit rail
    inB, outB = mB.dmem.words[6], mB.dmem.words[7]
    evB = simB.trace.events
    s_in = evB["step"][evB["neuron"] == inB.completion.u]
    s_out = evB["step"][evB["neuron"] == outB.completion.u]
    assert len(s_in) and decode_at(simB.trace, inB.rail_taps, int(s_in[-1]), 94)[0] == sent
    assert len(s_out) and decode_at(simB.trace, outB.rail_taps, int(s_out[-1]), 94)[0] == pixel, pixel
    print(f"demo1: A sent {sent} at {stepsA[0]/10:.0f} ms, B received at {s_in[0]/10:.0f} ms, pixel {pixel} at {s_out[0]/10:.0f} ms;",
          mA.net.n, "+", mB.net.n, "neurons")
