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

The report's headline must be restated as "0 wrong in 4,000 outputs on one realization;
5 wrong in 1,600 on another" until the copy-77 mechanism is found and fixed.
