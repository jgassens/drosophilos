# Stage H autapse census

An autapse is a neuron-to-itself anatomical connection; Dale's law means all outputs of one designed neuron use one transmitter sign.

| row | count |
| --- | ---: |
| anatomical self-edges | 101 |
| transmitter class: inhibitory / excitatory / modulatory / unknown | 27 / 53 / 1 / 20 |
| placement sign: inhibitory / excitatory | 27 / 74 |
| configuration-F / non-F flip-flop adder neurons | 740 / 664 |
| F latch / one-shot self-edge demand (quanta) | 140 (-724) / 8 (-15932) |
| latch: standalone inhibitory loops with >= 12 synapses / full-rule carriable | 2 / 0 |
| one-shot: standalone inhibitory loops with >= 249 synapses / full-rule carriable | 0 / 0 |

The real F build directly supplies the demand, split by its netlist self-edge quanta rather than a hand-matched circuit pattern.
The two standalone latch candidates are bodyIds 10540, 801562; neither could also host the excitatory loop to which its autapse belongs.
Both full-rule counts are zero because every relevant designed source votes excitatory while its self-edge is inhibitory.
Thus inhibitory self-feedback on an excitatory neuron needs a separate inhibitory neuron, making a two-step loop through an interneuron anatomically possible.
The optional cross-check against the driven mutual-inhibitory-pair pool was not done.
