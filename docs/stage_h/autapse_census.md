# Stage H autapse census

An autapse is a neuron-to-itself anatomical connection.

| row | self-edges |
| --- | ---: |
| supply total | 101 |
| inhibitory supply | 27 |
| excitatory supply | 74 |
| carriable latch autapse at k_max 4 | 2 |
| carriable one-shot autapse at k_max 4 | 0 |
| demand on the 4-bit adder (latch / transaction) | 124 / 8 |

Profile 2 allows a designed edge only when the presynaptic host has its sign and `0 <= |q| <= k_max * count * QUANTA_PER_SYNAPSE`; `Policy.req_count` supplies the count conversion.
The latch feedback is -724 quanta and needs 12 synapses at k_max 4; the transaction feedback is -15932 quanta and needs 249.
The raw table has 123 self rows, of which 22 have a body outside the MCNS neuron table, leaving 101.
Cholinergic (excitatory) self-edges are shown as supply but cannot carry these inhibitory designed edges.
