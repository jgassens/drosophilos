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
INTP has a one-entry shadow queue, INTQ. INTS ("INTP settled") records whether INTP is
stably empty: its r1 rail is loaded at power-up and reasserted from `IT.d` after eight hops
(~42 ms), while an INTP.r1 rise asserts INTS.r0. Arrival and timeout queue relays veto on
INTS.r1, not INTP.r0. Consequently a request against a settled-empty INTP goes directly to
INTP, one against a pending INTP enters INTQ, and one in INTP's post-clear paralysis window
also enters INTQ instead of disappearing. Concrete request sources retry INTS.r0 after 12
hops so a request racing a fresh INTS.r1 kill train still leaves the state marked pending;
INTQ promotion asserts INTS.r0 directly as it re-ignites INTP.r1. Taking the active request
clears INTP and, after 20 relay hops (~106 ms), promotes INTQ, beyond INTP's ~80 ms
paralysis; three more hops clear INTQ. The isolated timing probe previously lost requests
at 5--30 ms after `INTP.clear`; it now loses none at 0--45 ms. A depth of one is sufficient
for this milestone's single message in flight: the only overlap is its arrival and timeout.

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

## The exact channel (second milestone, in test)

`tests/test_exact_channel.py`: the alternating-bit protocol on two machines. A sends
`seq | payload` (bit 3 the sequence bit, bits 0-2 the payload) by storing to its port word; B's
arrival handler acknowledges at once on the reverse link (stores the sequence bit to its own
port word), compares the sequence bit with the one it expects, and on a match consumes the
payload, emits it as a pixel, counts the delivery and flips its expected bit; a duplicate
(same bit) is acknowledged but not delivered. A's handler tells a reply from a timeout by
loading the send timer's status word: on a timeout it counts the retry and re-sends; on a
reply it checks the ACK's sequence bit and records "acked". Both handlers empty their input
port word (`CLR`) before returning, which is the receive credit.

Found on the first clean-link run: A's status word was *empty* on the reply path (the timer
had never written it), a LOAD of an empty word reads as both rails, and the machine halted as
a fault, so `acked` stayed 0 although B had delivered once. The rule it settles: **every
word a handler may read on any path must hold a value on every path**. The status word now
starts at 0 in the image, the timer writes 1 through a write port of its own, and the handler
stores 0 back instead of emptying it (`docs/a2_ram_control.md` §2.2).

A late ACK exposed a second ordering rule. If the original send timer expires first, the
timeout and arrival cannot be merged into the same INTP state: the timeout handler must run
first (status = 1), re-send, and leave the held ACK for the queued arrival handler (status =
0). Nor is arrival completion a receive credit. The timer is cancelled by any evidence of an
ACK, twice over: by the arrival's completion train (the ACK is here, so the send was not
lost — cancelling only on consumption made the clean link retry once, since the handler
consumes the word seconds after a 1,900-hop timer expires), and again when the handler
consumes the input word: its `CLR` finishes the word's reset and raises READY, which
launches a four-pulse cancellation train across the timer chain. The second cancel is the
one that matters for a retransmission: a second ACK that lands while the first is still held
never completes, so consuming the first must cancel the re-send's timer. The slow late-ACK
regression delays B→A so that `TIMER.h1899 < LINK.in.arrive < INTP.clear`, and checks two
handler entries, one retry, one delivery and `acked = 1`; the clean link checks no retry
(Juno job 405152: clean, dropped-event and late-ACK scenarios all pass).

Still not built: epochs, credits beyond one message in flight, corruption detection at the
receiver (a corrupted rail event is a double rail, refused by the completion tree as a fault,
but nothing yet turns that refusal into a retransmit request), reordering, and the FlyLink
packet format of `docs/spec.md` §3.4 with its provenance rule; the port codec is a bare
rail-rise export.

## A rule found on the way

The accumulator is dark at power-up, and a dark operand lights both A rails in the operand
gate. MOV and LOAD ignore A, but JMP, JZ, JNZ and STORE run `OR 0`, which then double-rails
the result and is refused (node B's idle loop halted on its first JMP). The lowering now
emits `MOV 0` as the prologue; hand-written programs must write the accumulator first.
