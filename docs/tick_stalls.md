# Tick stalls: event diagnosis and campaign accounting

This implements the replay-and-localize part of `docs/perf_campaign.md` §6 without
pretending that one perturbed copy can be replayed independently. Campaign stray input is
drawn on the device for the whole batch, so copy N's stream depends on B. The diagnostic is
therefore read-only: it rebuilds the campaign kernel, validates the dump's roles, and analyzes
the recorded events. A separately constructed RefSim corner is the regression for a localized
mechanism.

## `bench/stall_diag.py`

The tool accepts either the seed-107 compact format (`step`, `neuron`, `uniq`, `role_of`) or a
new role-filtered replay (`step`, `neuron`, `role`). It rebuilds `examples/tick2.c` with
`compile_c`, `compile_kernel(loop_body(prog), "i")`, and `build_pipeline` using the campaign's
datapath and request-repair options. Every recorded role is asserted against the rebuilt
netlist. The old artifact predates Track B's role-preserving reorder of generic ALU
construction, so any needed moved datapath ID is resolved through a unique role name; the
handshake IDs involved in this failure are unchanged.

The analysis makes one vectorized selection pass over the spike arrays. It does not iterate
23 million spikes in Python. For each cell and transaction it reconstructs:

- every source request's false and true latch trains, receipt, DONE-driven clear, and repair;
- the go chain, START, ACT^d, stage completion, commit request/commit, and DONE;
- IDLE, fault/refusal circuitry, and input-register completion/READY.

It classifies the earliest phase which remains blocked beyond the workload threshold as a
stuck request, blocked go, datapath non-completion, consumer-held commit, missing DONE, or
input producer/watchdog failure. Roles omitted by the replay filter are treated as unavailable,
not silent. The Markdown report contains the blocked cell, its producers and consumers, the
neuron IDs and event steps, and per-copy campaign accounting.

Example:

```sh
python -m drosophilos.bench.stall_diag \
  data/a2/tick_s107_node18_compact.npz \
  --campaign data/a2/kc_tick_B90s107b.json \
  --node 18 \
  --out docs/a2/tick_s107_node18_stall_diag.md
```

## Seed 107, copy 18: precedent reproduced

The tool independently selects `c2_sel`'s second transaction as the first blocked START. Its
classification paragraph is:

> The first blocked cell is **`c2_sel`**, at its **2nd START**. This is a **request / repair /
> clear** failure, not a completion failure: false request latch **11828,
> `c2_sel.req.c0_addr0.u`** remains lit after the next `c0_add` request's clear. The preceding
> repair pulse **11970, `c2_sel.relight.c0_add.edge`**, at step **37,674** reached an
> already-live false rail. Both request rails are consequently live; false vetoes the go
> chain, so START, ACT^d, stage completion, commit, and DONE do not occur.

The relevant tool-generated timeline is:

| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |
|---|---|---:|---:|---:|---:|---:|
| `c0_add` / 1 | c4_sel 25; input 11,834 | 13,731 | 14,292 | 18,131 | 19,691 | 23,881 |
| `c0_add` / 2 | c4_sel 66,991; input 21,511 | 68,450 | 69,014 | 73,420 | 74,981 | 79,134 |
| `c1_and` / 1 | c0_add 23,962 | 24,668 | 25,221 | 30,026 | 30,820 | 34,924 |
| `c1_and` / 2 | c0_add 79,215 | 79,926 | 80,482 | 85,425 | 86,219 | 90,332 |
| `c2_sel` / 1 | c0_add 23,964; c1_and 35,005 | 36,896 | 37,467 | 39,805 | 41,329 | 45,432 |
| **`c2_sel` / 2 (blocked)** | c0_add 79,218; c1_and 90,414 | **—** | — | — | — | — |
| `c3_xor` / 1 | c2_sel 45,512 | 46,232 | 46,803 | 50,362 | 51,146 | 55,408 |
| `c4_sel` / 1 | c2_sel 45,518; c3_xor 55,491 | 57,374 | 57,961 | 60,404 | 62,672 | 66,905 |

The live-false evidence is equally specific. Neuron 11828's false-u train has intervals
23–24,101 and 36,936–299,993. Repair fires at 37,674. When the next `c0_add` receipt arrives at
79,216, clear neuron 11836 fires at 79,310, 79,354, and 79,398, but the false train survives.
Its true partner is also live from 79,218. `c1_and` becomes true at 90,414; the false rail then
continuously vetoes go, even though IDLE is true. Every transaction which actually STARTed in
the dump completed and committed.

`tests/test_stall_diag.py` constructs this mechanism directly in a small one-cell/two-source
kernel with two RefSim copies. The fixed copy includes the live-false repair veto and completes
twice. The comparison copy zeros only that edge; its live false train accelerates, survives the
second clear, leaves both request rails lit, and never produces the second START or DONE. These
are bounded synthetic parameters, not reconstructed campaign weights or stray events.

## Seed 108 replay

When Juno 413471 returns `data/a2/tick_s108_node27.npz`, run exactly:

```sh
python -m drosophilos.bench.stall_diag \
  data/a2/tick_s108_node27.npz \
  --campaign data/a2/kc_tick_B90s108.json \
  --node 27 \
  --out docs/a2/tick_s108_node27_analysis.md
```

The queued role filter contains request, relight, START, ACT^d, IDLE, commit, DONE, kill,
stage-valid/completion, fault, READY, and go roles, so absence of one of those captured signals
is meaningful. The tool will validate the role strings before classification. A seed-108
mechanism regression must be added only after this dump names the interaction; until then it
is deliberately scaffolded, not guessed.

## Reporting discipline and stall thresholds

Never collapse throughput and correctness into one pass/fail count. Every campaign report must
state these separately:

| field | meaning |
|---|---|
| requested transactions | input tokens the workload asked the host to deliver |
| completed transactions/outputs | observed protocol completions (also state output arity) |
| wrong values | completed, decodable values unequal to the reference |
| duplicates | completions beyond the one owed per output and token |
| detected refusals | fault/watchdog/timeout signals which explicitly refused work |
| unfinished | requested outputs with no completion at observation end |

For the 100-copy, eight-token, two-output seed-108 campaign, the known accounting is 800
requested transactions, 1,600 requested outputs, 1,569 completed outputs, zero wrong values,
and 31 unfinished outputs in copies 21, 22, 27, 47, and 53. Duplicate and detected-refusal
counts must still be emitted from the campaign artifact rather than inferred from those
numbers.

A run stopped at `--max-ms`, a wall limit, scheduler limit, or capture limit while work remains
is **truncated**. It is not automatically stalled. The whole 90 s seed-108 run is therefore a
resource-truncated campaign; an individual copy may additionally be called stalled only when
its event record shows a particular handshake phase remaining blocked beyond a justified
threshold.

The diagnostic derives that threshold from the tested protocol and workload:

- a latch is considered continuously lit while spike gaps are no more than three nominal loop
  periods (141 steps for this build);
- the phase-stall threshold is twice the longest completed START→DONE duration in the same
  dump, with a 20,000-step (2 s) floor;
- missing output count alone is never the stall criterion;
- the campaign's 90 s resource ceiling is an observation limit, not a handshake deadline.

On the precedent the longest healthy cell duration is 10,684 steps, hence the tool's threshold
is 21,368 steps (2.1368 s). `c2_sel` remains blocked for far longer and has a concrete
request/repair/clear cause. This threshold belongs to the `tick2.c` workload and the current
handshake contract; it must not be generalized to `doom4`, timestep-refinement validation, or
CUDA transaction-level certification.

## Faster campaign backend

`make_perturbed_sim(..., backend="torch" | "torch-fast")` now constructs either `TorchSim` or
`FastSim`; `torch` remains the default. Weight noise, threshold drift, bias drift, and the stray
seed are drawn before backend construction in the same order. Both simulators draw one seeded
`torch.rand((B, n))` stray mask at the same place in each step. A short CPU regression compares
the per-copy perturbations and exact `(step, value)` kernel outputs. Cluster campaigns can use:

```sh
python -m drosophilos.bench.kernel_campaign tick \
  --copies 100 --mix B --seed 108 --max-ms 90000 \
  --device cuda --backend torch-fast \
  --out data/a2/kc_tick_B90s108.json
```

## Seed 108 on the current build (2026-09-20, Juno 413471 / 413581 / 413582)

The recorded failure set does not transfer. The campaign report's seed-108 run (Juno 409236:
1,569 ok / 0 wrong / 31 missing, copies 21, 22, 27, 47, 53) and today's run of the same
command on `main` differ in every copy's output timing (100 of 100), because the merge of
Track B re-ordered neuron and edge indices and the per-seed noise is drawn by index: same
mix, same seed, a different realization. Today's build is deterministic — two reruns agree
with the replay in every copy, step for step (100 / 100) — and its realization is:

| | ok | wrong | missing | failing copies |
|---|---|---|---|---|
| Juno 409236 (report) | 1,569 | 0 | 31 | 21, 22, 27, 47, 53 |
| today (413471 = 413581) | 1,566 | **5** | 29 | 3 (1 missing), 8 (12), **77 (5 wrong, 2 missing)**, 98 (14) |

So the "0 wrong values in 4,000 outputs" of the report was one realization of the noise; the
present one shows five silent wrong values on one copy in 1,600 outputs. Copy 27's captured
dump (`tick_s108_node27.npz`) is of a copy that completed and is not useful. Handshake
captures of copy 77 (the wrong values) and copy 8 (12 missing) are queued (Juno 413584,
413585); `stall_diag` reads them with:

```sh
python -m drosophilos.bench.stall_diag data/a2/tick_s108_node77.npz --campaign docs/a2/kc_tick_B90s108_current.json --node 77 --out docs/a2/tick_s108_node77_stall_diag.md
```

### Copy 77: one refused input word, resent by nobody (Juno 413584)

The "five wrong values" are one dropped token. Copy 77's outputs are exactly what the kernel
computes for the seven-token stream `[5, 5, 250, 0, 40, 40, 40]` — token 4 (`vel = 3`) is
missing and everything after it is right for the stream it actually received (`px`: 25, 30,
24, 24, 64, 104, 0; `mx`: 88, 86, 84, 82, 80, 82, 80). Scoring by position turns one missing
token into 5 wrong + 2 unfinished.

The capture (13.5 M spikes, `IN.*` handshake roles) shows the drop, step by step:

| step | event |
|---:|---|
| 130,481 | stage DONE for token 3; stage reset; `IN.Q.ready` at 131,366 (READY #3) |
| 131,852–131,903 | host loads token 4 (`3` = bits 0 and 1 true): valid latches rise on bits 1–7 — **bit 0 never latches** (`IN.Q.valid0` has no rise) |
| — | the stage's completion tree never fires; no commit request; `IN.creq`, `IN.commit` silent |
| 136,443 | 4,590 steps (459 ms) after the load, the producer's watchdog (`add_liveness`, 60 + 4·8 = 92 hops) times out and resets producer and stage together (`IN.P.ready_delay0` and `IN.Q.ready_delay0` both at 136,44x); the valid latches of bits 1–7 die at 136,47–136,55 |
| 137,243 | `IN.Q.ready` fires again (READY #4): the reset's READY, not a consumption |
| 137,732–137,789 | the host, which counts READY rises, loads **token 5** (`0`); all eight valid latches rise, completion at 139,341, commit at 139,386 — a clean word, one token late |

No fault gate fired (`IN.Q.fault*` silent all run) — the kernel refused the half-latched word
cleanly, as designed (fail-stop), and the runner even counted it (`timeouts 1` in the log).
What was missing is the producer's half of the protocol: the host never resent the refused
word. Which element lost the single ignition pulse (the `IN.P.b0r1` rail, its data relay or
the stage rail) is not in this capture (rails are not in the role filter); it does not
change the classification: **request** phase, producer side — a detected refusal with no
retry. The `torch-fast` realization did not reproduce it because a K-step-shifted load meets
different noise.

**Fix (2026-09-20, `lib/kernel.py`, both runners):** the runner now counts every rise of the
input watchdog's TIMEOUT latch per node and stream as a *detected refusal*
(`stats["refusals"]`, `stats["refused"]`), and re-sends the refused word on the READY the
reset raises (`retry_refused=True`, `max_retries=3`; a word refused more often blocks its
node, `stats["blocked_nodes"]`, so the copy stays unfinished rather than skipping ahead).
`kernel_campaign` records `refusals`, `retries`, `per_node_refusals`. Regression:
`tests/test_input_refusal.py` constructs the refusal directly (one rail's ignition dropped
from one load of a 4-bit MOV kernel, `rail_filter`): with the retry the four tokens come
out whole (refusals 1, retries 1, faults 0); without it the outputs are `[1, 3, 5]`, the
seed-108 shape; a permanently dark rail blocks the node after three retries. Capture
`wd\.timeout` in `--dump-roles` so `stall_diag` can count refusals from a dump.

**Review (openai-sol, 2026-09-20) and what it changed.** Two findings held. (1) A resend on
every TIMEOUT rise would replay a word the kernel had already accepted if a stray spike lit the
TIMEOUT latch later — a duplicate output and a shifted remainder. The runner now watches each
input register's stage-to-master commit as well: a word is resent only if no commit followed
its load, and the *next* word goes in only once the previous one has committed (READY alone
ran the host a word ahead whenever a stray reset re-raised it — the copy-77 shape from another
direction). A READY rise over a word that neither committed nor timed out means the stage
was reset under it: that word is resent too (`reason: "reset"`). (2) Constructing the stray
case (`tests/test_input_refusal.py`, a TIMEOUT doublet in the ~800-step window between the
stage's reset and the next load — while the stage is complete its completion train holds the
watchdog's cancel on the latch) exposed a kernel gap: the stray resets the stage but not the
producer, whose reset trigger is still depressed by the completion train that just ended, so
the TIMEOUT latch stays lit; the spoiled next word then sits in the producer with the
watchdog's final pulse landing on an already-lit latch — no TIMEOUT rise, no fail-stop, a
stuck stream. The stage's reset now clears the TIMEOUT latch too (`protocol/handshake.py
add_liveness`); the chain re-lights it ~460 ms after the spoiled load, both registers reset,
and the word is resent (three loads of the same word, one output, in the test). The
runner-side rules above are what keep this from duplicating anything.

A second review (claude-fable, of the same commit) confirmed the retry logic's ordering on
both runners and added: a repeated TIMEOUT rise for a word already queued is a duplicate,
not a block; `retries` counts retry loads that went in, not queue entries; a blocked node
finishes as a fail-stop once the words it did take in have produced their outputs, so one
blocked copy no longer holds a 100-copy batch to `max_ms`; and the retry path now has tests
for two nodes, a refused last token, and the `torch-fast` block path. Not done: a two-stream
kernel test, and the reload margin of a kill pair at the commit gate's ~50 ms (the kernel
tests are the check for that).

Restated headline: seed 108, current build, 1,600 outputs — 0 silent wrong values; 1
detected refusal (copy 77, now resent); 29 unfinished in three stalled copies (3, 8, 98). The
report's "0 wrong in 4,000" stands as a statement about the kernel; the host runner is what
dropped the word.

### Copy 8: a fast false rail slips through the kill train (Juno 413616)

`stall_diag` on the copy-8 capture (`docs/a2/tick_s108_node8_stall_diag.md`): first blocked
cell `c4_sel`, third START; **stuck request**: the false rail of `c4_sel.req.c2_sel`
(neuron 14087) stays lit after `c2_sel`'s third DONE delivers the request (true rail rises
at 164,436, `received` at 164,439, the clear train fires at 164,535 / 164,581 / 164,624) —
and no repair pulse is involved this time, unlike the seed-107 precedent. Both rails live,
the false rail vetoes go, and the cell never starts again; the two earlier clears of the same
latch (45,769; 101,036) had worked.

What differs at the third clear is the latch's speed. Its loop period wanders between 33 and
41 steps over the run (nominal 47; threshold and bias drift plus a +weight draw on the loop
edges), and at the failed clear it was **34 steps** against 38–39 at the two that worked; the
three 0.75×-loop pulses, 46 steps apart, slowed it (gaps 42, 65, 72) and it recovered.

Measured on one latch in RefSim (`tests/test_kill_margin.py`; twelve kill phases; loop weight
scale × threshold offset):

| kill train | +8 % / −0.8 mV (40 steps) | +20 % / −0.8 mV (38) | +40 % / −1.2 mV (31) | +60 % / −1.2 mV (29) |
|---|---:|---:|---:|---:|
| 3 × 0.75 (until 2026-09-20) | survives 2–10 / 12 phases | 12 / 12 | 12 / 12 | 12 / 12 |
| 4 × 1.0 | 0 | 0 | 11–12 / 12 | 12 / 12 |
| 4 × 1.25 | 0 | 0 | 0 | 4–12 / 12 |
| **3 × 1.5** (register reset strength, same three pulses) | 0 | 0 | 0 (+30 % / −1.2 mV, 33 steps: 0) | 10–12 / 12 |
| 4 × 1.5 | 0 | 0 | 0 | 0 — but stops the control machine |

With kill weights themselves 8 % low (the same noise) the old train fails from +4 % / −0.8 mV.
A killed rail reloads ~150 ms after the kill with any of these trains in isolation (a reload
100 ms after a kill fails for all; the handshake's earliest relight, START after a request,
is ~190 ms).

**Neither stronger train survives contact with the kernel.** 4 × 1.5 leaves the control
machine's ring stuck after its first commit (`tests/test_compiler.py`, cluster test run
413662). 3 × 1.5 runs the machine and every local test, but in the 100-copy mix-B tick
campaign it **stalled 85 of 100 copies**, most after the first token (Juno 413672: 676 ok,
910 missing) — under noise the rails' relights ~190 ms after a kill no longer clear the
extra after-hyperpolarisation. The train therefore stays **3 × 0.75**; the margin problem is
recorded, not fixed (`tests/test_kill_margin.py`: the default train lets the +20 % latch
through at every phase, a 3 × 1.5 train kills it in isolation, a kill pair reloads at 150 ms
for nominal and slow loops). A kill that beats a fast latch without slowing the relight — a
longer train at the old strength, or a relight drive that scales with the kill — is a
separate experiment. Campaign records carry `kill_train`, and `stall_diag` rebuilds older
dumps with the train they were run with.
Validation on the cluster: seed 108–110 campaigns on the new build against the old build's
stall counts (a netlist change is a new noise realization, so the comparison is per-seed
totals, not per copy). Old build (3 × 0.75, commit `128d982`), 100 copies each:

| seed | ok | wrong | missing | failing copies |
|---|---:|---:|---:|---|
| 108 (413471) | 1,566 | 5 (= 1 refused word, copy 77) | 29 | 3, 8, 77, 98 |
| 109 (413666) | 1,567 | **8** (copy 61) | 25 | 8, 61, 78, 98 |
| 110 (413667) | 1,574 | 0 | 26 | 0, 45 |

Fixed build (`6bac959`: resend of refused and lost words, next word after the previous commit,
stage reset clears TIMEOUT; kill train unchanged; the two added synapses make it a new noise
realization, so the comparison is per-seed totals):

| seed | ok | wrong | missing | refusals / retries | failing copies |
|---|---:|---:|---:|---|---|
| 108 (413703) | 1,596 | 0 | 4 | 0 / 0 | 24 (stalls after token 6) |
| 109 (413704) | 1,546 | 0 | 54 | 0 / 0 | 36, 49, 81, 87 (stalls) |
| 110 (413705) | 1,532 | **2** | 66 | 0 / 0 | 20, 42, 43, 57, 69, 74, 76 (stalls), **55** |

Build with both kernel fixes as well (`b4e6905`: power-up veto, commit idle rail re-ignition;
28,669 neurons, again a new realization):

| seed | ok | wrong | missing | failing copies |
|---|---:|---:|---:|---|
| 108 (413812) | 1,584 | 0 | 16 | 67, 72 (stalls) |
| 109 (413813) | 1,548 | 0 | 52 | 29, 41, 50, 77 (stalls) |
| 110 (413814) | 1,541 | 0 | 59 | 18, 20, 34, 67, 73, 86 (stalls; 73 produced no output at all) |

**300 copies, 4,800 outputs, 0 silent wrong values, 0 refusals, 0 faults; 12 copies stalled
(4 %, 127 outputs unfinished).** The two wrong-value mechanisms and the refusal class are
gone from three realizations; what remains is stalls, whose one localized cause (a fast
latch slipping through the kill train, copy 8) is unfixed and whose other instances are
unread. Each needs its own capture (~20 min of H200 per copy). Copy 73 of seed 110, which
never produced its first output (Juno 413875, `docs/a2/tick_s110_node73_stall_diag.md`), is
the kill-margin class again: the image-lit false rail of `c2_sel`'s request from `c1_and`
(neuron 11845, loop period 41 steps against 47 nominal) survived the clear train at the very
first request (35,103 / 35,153 / 35,197) and vetoed `c2_sel`'s first START for the whole run.
Two of the three stalls read so far are this one mechanism. Since the stronger trains failed
in the kernel (above), the next trial is four pulses at the original 0.75 strength (+1 relay
per kill train, 3.0 loop units of charge against 3 × 1.5's 4.5): in isolation it kills every
latch down to 39 steps at every phase and 38 steps at 5 of 12 phases, and it runs the control
machine; seeds 108–110 on the cluster decide. **Seed 108 decided against it (Juno 413917):
no stall at all in 100 copies, but 6 silent wrong values in 3 copies** — `c9_sel` twice
selecting `mx − 2` where `dist & 128` said `mx + 2` (copies 19, 88: 76, 74 for 80, 78) and
`c4_sel` once keeping `px = 147` where `px & 128` should have zeroed it (copy 37) — a SEL
starting on a stale condition. The fourth pulse changes when the cleared request rail can be
relit, and the request/clear ordering the cells rely on slipped. Reverted: a stall is a
fail-stop, a wrong value is not. What is left to try is a *conditional* second clear —
re-kill only if the false rail is still lit ~50 ms after the true rail's rise — which adds no
charge to the nominal case. **Tried (`retry_clear`, Juno 413959): the same trade.** A gate
that needs the delayed receipt pulse and both rails' trains fires a 1.5 × train ~64 ms after a
DONE only in the stuck state; it cures the constructed stall
(`tests/test_request_retry_clear.py`) and never fires on a healthy cell, but seed 108 gave 3
silent wrong values in 2 copies (again a SEL on a stale condition: `px = 147` kept, `mx − 2`
for `mx + 2`) and 5 stalls. Off by default, kept as an experiment. Three attempts have now
turned stalls into wrong values by changing when a request's false rail can be relit; the
SEL's condition request evidently tolerates less reordering than the go chain's guards
assume, and that ordering — not the kill train — is the thing to read next (a capture of one
of those wrong copies on the trial build).

### The ordering, read (seed-108 copy 5 on the retry build, Juno 413989)

`c1_and` (`px & 128`) ran nine transactions for eight tokens. At 260,754 `c0_add`'s DONE
lit `c1_and`'s request; the retry gate fired at 261,527 — both rails still live, the first
clear train not yet through — and `c1_and` STARTed at 261,544, seventeen steps later, which
re-lit the false rail ("consumed"). The retry's train then landed on that freshly re-lit rail
and killed it; START's own kill had taken the true rail; both dark. The guards read "not
false" as pending, so after this transaction's DONE and IDLE the go fired again (274,302): an
extra `c1_and` transaction and DONE (284,882) on the same `px`, which `c2_sel` consumed as
token 6's condition — and every condition after was one token late, until token 8, where
`px & 128` of 107 (instead of 147) left `px = 147` unzeroed.

**Nothing ordered a DONE's clear of a request's false rail against START's re-light of the
same rail.** With a three-pulse train and a cell that starts within a few ms of the request
(all other requests and IDLE already true) the train's ~15 ms tail ends before the go chain
gets to START; a fourth pulse, a stronger pulse, or a retry reaches past it. That is why
each of the four trials produced the same wrong value. Fix (`build_pipeline`,
`start_relight_hops=5`, recorded): START re-lights the false rails through a 5-hop delay
(~27 ms), after any clear train has ended; in between the pair is dark on both rails, which
the go chain cannot act on (IDLE was killed at START) and which delays a producer's commit
gate by the same ~27 ms. On top of it the kill train is **4 × 0.75** (Juno 413917 showed
four pulses remove every stall on seed 108; 40-step loops die at every phase). Seeds 108–110
on this build (`3cc733b`, Juno 414205–7):

| seed | ok | wrong | missing | failing copies | wall |
|---|---:|---:|---:|---|---:|
| 108 (414205) | **1,600** | **0** | **0** | none | 572 s (nothing waited for the cap) |
| 109 (414206) | **1,600** | **0** | **0** | none | 568 s |
| 110 (414207) | **1,600** | **0** | **0** | none | 568 s |

**300 copies, 4,800 outputs: 0 wrong, 0 missing, 0 faults, 0 refusals.** The stall rate went
from 4 % of copies to none on three realizations once the clear-train tail could no longer
reach START's re-light, and the fourth pulse then took care of the fast latches. Every
100-copy campaign now finishes in ~570 s instead of the 90 s cap's ~990 s, because no copy
waits it out.

Over 300 copies of the first fixed build: 10 failing on the old build, 13 on the fixed one — the stall rate is set by
kernel mechanisms the host fixes do not touch, and it swings with the realization (1, 4 and
8 copies for three seeds of the same build). No refusal occurred in the three fixed runs, so
the resend path was not exercised on the cluster; it is exercised by the tests. Seed 110's
copy 55 is the copy-61 family again, mid-run: at token 7 `mx` moves +4 instead of +2 (78 →
82, then 80 at token 8, consistent with `mx` = 82), i.e. one extra `mx` transaction — with
`px` right throughout and no fault or timeout. Its capture (Juno 413725) shows a third
mechanism, and it starts at power-up: `c7_add` (`mx + 2`) commits its initial-value result
at 19,077; the commit pulse kills the commit request's "pending" rail and re-lights its
"nothing to commit" rail — **54 ms after the autocommit's kill train dropped that rail**,
inside its after-hyperpolarisation, and the single ignition produces one spike and no train
(`c7_add.creqr0.u`: one spike at 19,081, then dark until 93,146). The pair is dark on both
rails. The commit guard (`guarded_pulse`) fires on the reader's free-rail rise vetoed by the
"nothing to commit" rail — it reads "not false" as true — so when `c9_sel` STARTs its first
tick (91,942) and its request rail goes free, a **second commit pulse** goes out (93,107) with
nothing new in the stage; its DONE (97,131) re-lights `c9_sel`'s request with the old value,
`c9_sel`'s next START consumes that, and `c7_add`'s genuine value (autocommit 108,136) waits
for the START after — from then on `c7_add` runs one token behind `c8_sub`, invisible while
`c9_sel` selects `c8_sub` and wrong (82 for 80) the first time it selects `c7_add`, at token 7.
**Fixed (`lib/kernel.py gate_commit`, `commit_reignite`):** a second ignition of the
"nothing to commit" rail ~85 ms after the commit pulse (16 hops), past the recovery;
`tests/test_commit_request_reignition.py` removes the immediate ignition on one copy and
both on another and reads the rail's trains: nominal and fixed copies relight after every
commit, the unfixed copy shows a lone spike (the two-cell kernels absorb the duplicate commit
itself; the tick kernel's loop did not). Both fixes are recorded in the campaign record
(`powerup_veto`, `commit_reignite`) and `stall_diag` rebuilds older dumps without them.

Seed 109's copy 61 is a **new silent wrong-value mechanism**, not a refusal: `c9_sel` commits
its initial value (90, `mx` before any tick) at 2.8 s, long before the first tick's output at
6.7 s, and from then on `mx` steps every other tick (88, 88, 86, 86, 84, 84, 86) while `px`
is right throughout — the state feedback lags a token. No fault, no timeout. A handshake
capture of that copy (Juno 413685, reproduced exactly) shows the mechanism's shape but not
its trigger: at step 27,850 — 2.8 s in, with `c9_sel` idle and nothing committed
(`creqr0` "nothing to commit" lit throughout, no START, no commit) — `c9_sel.done.edge`
fires: its master's completion latch re-rose. The DONE is a real one to the consumers:
`c7_add` and `c8_sub` take new requests at 27,932 and START extra transactions at 29,366
and 29,395 on the master's unchanged initial value (90), `c10_xor` and `c5_sub` likewise,
and the runner's decoder, which reads a master completion rise as an output, records the
90. From then on the `mx` loop carries one extra transaction and the state lags a token
(88, 88, 86, 86, …). **Classification: a spurious DONE — a state cell's master completion
re-rising without a commit, before the first tick.** The completion latch itself
(`c9_sel.M.comp.c3_0.L.u`) and the master's reset are not in that capture's role filter; a
second capture with `M\.comp|M\.reset` in the filter (Juno 413719) shows the cause. The
image lights a state cell's initial value in its master but keeps the master's completion
**root** dark on purpose (a complete master would fire DONE at power-up): the leaves and the
inner levels of the completion tree are lit from step 0, the root AND gate has one lit input
(the data half of the tree) and one dark (the flag half), and so sits at 65 % of threshold
for the whole run. Its first spike ever came at 27,728 — one stray coincidence — and the
ignition relay lit the root latch at 27,808; from a lit latch there is no way back. **Fixed
(`lib/kernel.py build_pipeline`, `powerup_veto`):** a veto latch, lit by the image and killed
by the master's first reset (its first commit), holds the root's ignition relay down until
the first real value has landed. `tests/test_state_master_powerup.py` fires the root gate
once at step 2,000 on a state kernel: before the fix the completion rose before the first
token; after it no completion precedes the first load and the outputs are right.

### `--backend torch-fast` is not a step-identical substitute in the campaign (Juno 413583)

Same seed, same perturbed copies: `FastSim` reproduced the `TorchSim` run's `(step, value)`
outputs on 71 of 100 copies and its values on 96, with a different failing set (8, 39, 83, 98:
0 wrong, 42 missing; `TorchSim`: 3, 8, 77, 98: 5 wrong, 29 missing) in 665 s of wall time
against 969 s. The cause is the K-step observation lag (`observe_every` = 94 by default):
under noise a token loaded up to 9.4 ms later meets a different stray-input sample, and which
copies stall is sensitive to that. `observe_every=1` restores `TorchSim`'s schedule exactly
(the parity tests use it) at the cost of the per-step host copy. Rule for the reliability
campaigns: compare runs on one backend, or run `FastSim` with `--observe-every 1`; a change
of backend is a change of realization, not a re-run. Copy 77's wrong values did not occur on
the `FastSim` realization, so the mechanism is timing-sensitive — the capture on `TorchSim`
(413584) is the one to read.
