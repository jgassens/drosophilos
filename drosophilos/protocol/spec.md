# Token protocol (normative) — bounded-delay, self-timed, four-phase, dual-rail

This file defines the only way a word moves between two circuits in DrosophilOS.
`machine.py` is its executable form; `explore.py` enumerates every reachable state of the
abstract machine and checks the properties in §5. Neural implementations (`latch.py`,
`celement.py`, `handshake.py`) are certified against the same event sequences.

## 1. Vocabulary

- **Rail**: a neural channel. A bit has two rails; a spike on rail 1 means "valid one", on
  rail 0 "valid zero". Both rails active in one transaction is a **FAULT**. Silence is
  "not here yet", never zero.
- **Token**: a spike (or spike train) carrying one of: a DATA bit, ACCEPT, CLEARED, READY.
- **Latch**: storage that holds a rail value from the moment it arrives until RESET.
- **Producer P**: holds the word to send in its own latches (its output register).
- **Consumer Q**: captures the word in its input latches, detects completion, hands the
  value on, and resets.

## 2. The four phases (one transaction, word number `n`)

```
phase 1  DATA(n)     P's output latches drive Q's input rails; each bit arrives on exactly
                     one rail, in any order, at any time.
phase 2  ACCEPT(n)   Q's completion element sees every bit valid, holds "complete", the
                     value is consumed exactly once, and Q sends ACCEPT to P.
phase 3  CLEARED(n)  P resets its output latches on ACCEPT and, when none of its bits is
                     valid any more, sends CLEARED.
phase 4  READY(n+1)  Q resets its input latches and completion element on CLEARED and,
                     when none of its bits is valid any more, sends READY. Only now may P
                     load and send word n+1.
```

Each phase is one token crossing the channel: DATA (2·width rails), ACCEPT (1),
CLEARED (1), READY (1). The order DATA → ACCEPT → CLEARED → READY is total; nothing
overlaps within one channel.

## 3. Component state machines

```
P:  IDLE --load(w)--> LOADED --send--> SENT --ACCEPT--> CLEARING --(no bit valid)--> AWAIT_READY --READY--> IDLE
Q:  EMPTY --DATA bit--> PARTIAL ... --all bits valid--> COMPLETE --consume, send ACCEPT--> AWAIT_CLEARED
    --CLEARED--> RESETTING --(no bit valid)--> send READY --> EMPTY
```

Latch value per bit: `E` (empty), `0`, `1`, `F` (fault: both rails seen).

Rules:
- A DATA bit on the rail a latch already holds is a **same-rail duplicate**: absorbed, no
  state change.
- A DATA bit on the opposite rail of a held latch, at any time before RESET, makes the
  latch `F`. A word containing an `F` bit is never consumed; Q raises FAULT, counts it,
  and proceeds directly to RESETTING (a fault transaction still completes the four
  phases so the channel does not deadlock: ACCEPT is replaced by FAULT-ACCEPT).
- DATA arriving while Q is RESETTING is dropped (the latches are inhibited during reset).
- DATA arriving while Q is EMPTY but P has not been given READY is a **stale token**
  (it can only belong to an earlier word). Without a sequence mechanism it is
  indistinguishable from real data and is accepted: this is the failure the checker
  demonstrates (§5, property 3). With **phase rails** (even words use rail set A, odd
  words rail set B, and Q enables only the current set) it lands on a disabled rail and
  is dropped. With a **timing bound** (READY is delayed after reset by more than the
  longest possible stale spike), it cannot occur. Each neural implementation states which
  it uses; the contract records the measured bound.
- A lost token (ACCEPT, CLEARED, READY, or a DATA bit) stops the transaction: that is a
  deadlock unless a timeout re-sends it. Producers retransmit DATA after a timeout; a
  retransmitted bit on the same rail is a same-rail duplicate and is absorbed, so
  retransmission is idempotent. ACCEPT / CLEARED / READY are likewise re-sent on timeout
  and are idempotent because each is enabled in exactly one state of the peer.

## 4. Timing assumptions (this is why it is "bounded-delay", not delay-insensitive)

- Inside a latch, the circulating spike regenerates within a bounded period (`P_loop`).
- The reset inhibition must outlast one loop period on every latch member it hits.
- READY / CLEARED are emitted a fixed delay after the reset pulse and are vetoed by any
  still-valid bit; the delay must exceed the reset settling time.
- A stale spike cannot arrive later than `T_stale` after RESET; `T_stale` is measured and
  READY is delayed by more than it (timing-bound variant).

## 5. Properties checked exhaustively on the abstract machine (`explore.py`)

1. **No deadlock**: from every reachable state, in the absence of lost tokens, both
   components return to (IDLE, EMPTY) with the word consumed or a FAULT counted.
2. **Exactly-once consume**: each word is consumed at most once, and exactly once when no
   fault or loss occurs.
3. **No stale acceptance**: a DATA bit tagged with word `n` never contributes to the value
   consumed as word `m ≠ n`. Demonstrated to FAIL without phase rails or the timing bound
   and to hold with either.
4. **No illegal transition**: every event is enabled only in the states listed in §3.
5. **Idempotent retransmission**: any token delivered twice leaves the outcome unchanged.
