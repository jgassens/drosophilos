# Stage C, first milestone: two nodes over FlyLink (Demo 1 shape)

`cluster/flylink.py`, `lib/control.py` (`port_out_word`, `port_in_word`), `tests/test_flylink.py`.

## What the transport is

The host's role is transport only. After every simulation step it reads the spikes of a
node's export neurons and schedules each as an event on the destination node's mapped neuron
after the modeled link delay, with the mapped drive. A packet is (source node, source neuron,
step); the mapping B_ji is fixed when the link is made. Nothing in a packet is a host-computed
value: the destination word is rebuilt from rail events and validated by its own completion
tree. Positive modeled delays give conservative ordering (an event can never arrive before
the sender's own clock has passed it).

Export: an output port word (a data-memory master) gets one edge relay per rail
(`LINK.out.b{i}r{r}`), which fires once per rail rise, i.e. once per STORE to that word.
Import: the input port word's completion feeds the machine's interrupt-pending rail
through an edge relay: a message's arrival is an interrupt, and the handler LOADs the word.

## Demo 1 shape, measured (clean model, two 4-bit machines, 6,082 + 4,524 neurons)

Node A: `LOAD x; ADD 3; STORE y; AND 1; JZ; LOAD y; XOR 6; STORE y; LOAD y; STORE port` (reads
an input, computes, branches, stores neurally, sends). Node B: `MOV 0` then an idle loop
`JMP 2 / JMP 1` (the FSM must keep passing safe points), handler at word 4:
`LOAD port_in; ADD 1; STORE port_out; IRET`.

| event | time |
|---|---|
| A's port word complete (value 6 for x = 5) | 9,114 ms |
| B's input word complete (4 rail events, 20 ms modeled delay) | 9,135 ms |
| B's pixel word complete (7) | 13,212 ms |

The pixel equals (A's result + 1) as the two references compute it. This is the shape of the
plan's Demo 1 minus the H0 circuit beside it: input-dependent neural storage, arithmetic, a
neural branch, a validated result sent to a second node, and a neurally computed pixel.

## What it is not yet (the rest of Stage C)

One message per input word. The exact-channel machinery is not built: sequence and epoch
fields, credits (the receiver must clear its port word before the next message), ACCEPT back
to the sender, timeout and retransmit, duplicate rejection, corruption detection, reordering,
backpressure and recovery. The port codec is a bare rail-rise export, not the FlyLink packet
of `docs/spec.md` §3.4 with its provenance rule.

## A rule found on the way

The accumulator is dark at power-up, and a dark operand lights both A rails in the operand
gate. MOV and LOAD ignore A, but JMP, JZ, JNZ and STORE run `OR 0`, which then double-rails
the result and is refused (node B's idle loop halted on its first JMP). The lowering now
emits `MOV 0` as the prologue; hand-written programs must write the accumulator first.
