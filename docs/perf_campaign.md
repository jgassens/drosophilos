# Performance campaign — orders and ledger

Adopted 2026-09-17 from the owner's campaign instructions (written against the 17 September
campaign report and the repository audit). The instructions are reproduced below unchanged;
the ledger at the top records what has been done against them and where the evidence is.

## Ledger

| order | status | evidence |
|---|---|---|
| §2 backend selection, timing, reproducibility record, profiling regions | done 2026-09-17 (openai-sol, cleanup openai-terra) | `fe2f07a`, `f95bcdc`; `bench/render_doom.py`, `bench/repro.py`, `sim/profile.py` |
| §3 fixed comparison suite (`bench/perf_campaign.py`) | built 2026-09-17 (openai-terra); sized for the cluster 2026-09-18 (claude-sonnet, openai-luna) after a default run consumed 14 h of laptop CPU; primitive baseline measured on Juno (412136, 1 h 16 min on one H200; `docs/perf/juno-h200-primitive.md`): wall/neural 5.1–5.3× at 3k neurons, 5.9× at 28k, 6.2–8.0× at 47k–110k; 8 copies cost the same wall time as 1 below ~50k neurons; the pipelined multiplier is 2.1× faster per token in isolation. Small level: 412137 crashed after a correct 2.5 h doom2 render on a duplicates-counter bug (fixed, `2cb03db`); rerun 412442 running (3/4 configurations) | `docs/perf_campaign_suite.md`; results land in `data/perf/` when the jobs run |
| §4 Track A (simulator) | **done 2026-09-19**: profile (Juno 412204) showed a launch- and sync-bound step, flat 0.36–0.46 ms from 3k to 880k neurons; FastSim (static-shape step, sparse-product delivery, Observer, K-step CUDA graphs) merged `70f7267`; H200 A/B (413020): 5.3× at 3k neurons, 3.6× at 28k, 2.2× at 110k, all outputs identical; the profiled twin (413021) shows the graph replay at 0.09–0.23 ms/step at every size, the rest being the primitive level's own all-neuron spike capture (`--no-spike-count` added); reviews: Kimi (A–B) and openai-sol (C–F), five findings fixed `bcb1f10`, `dd82449`; frame-scale twin running (413097 vs 412442) | `docs/track_a.md`, `docs/perf/juno-h200-trackA-ab.md`, `docs/perf/juno-h200-fast-profile.md` |
| §4 Track B (specialized cells) | **B1+B2 merged and measured 2026-09-20** (`f1806a0`; Juno 413149): −31–35 % neurons per AND/XOR/MOV cell, −14–18 % per kernel, latency −8–16 ms, wall time unchanged (FastSim's step does not scale with neurons at these sizes); STORE stays generic. Specialized-datapath campaign, seeds 108–110 × 100 (Juno 424774/424775/424777): **299/300 copies, 0 wrong**, 25,113 neurons versus 29,073 generic | `docs/track_b.md`, `docs/perf/juno-h200-trackB-prim.md`, `docs/tick_stalls.md` |
| §5 multiplier diagnostic | **done 2026-09-20** (`docs/mul_diag.md`): the pixel pass runs one pixel at a time — early values read by late cells hold their producers' commits (operand retention), so the period is the path latency (array 26.2 s, pipelined 42.4 s per pixel; the pipelined rows overlap 0 of 40 times inside the pass against 15 of 16 standalone); latency, not throughput, is the cost. Reproducer `tests/test_pipeline_retention.py` (relaying the value: 2.3× throughput). Fix = relaying long-lived values (the 'retained intermediates' experiment), not a multiplier change | `docs/perf/muldiag_*.json` |
| §6 tick-failure diagnosis, 1,000-tick Stage D exit | tool merged 2026-09-20 (`bench/stall_diag.py`, openai-sol): rebuilds the campaign kernel, validates a dump's roles, reconstructs every cell's request/repair/clear/go/START/ACT^d/stage/commit/DONE per token and names the first blocked phase; reproduces the seed-107 precedent (c2_sel, latch 11828, relight 11970 @ 37,674) with a passing RefSim regression of the mechanism; campaigns take `--backend torch-fast` with identical perturbation draws. **Seed-108 replay does not reproduce the recorded copies** (Track B's index reorder changed the noise realization; the build itself is deterministic, 100/100 copies identical across reruns) and shows **5 wrong values on copy 77** in 1,600 outputs — the report's '0 wrong in 4,000' was one realization; **copy 77 read (413584): one refused input word, not five wrong values** — bit 0 of token 4 never latched, the producer watchdog refused the word after 459 ms, READY re-rose and the host loaded the next token; outputs match the 7-token reference exactly. Fixed on the host side 2026-09-20: both runners count refusals per TIMEOUT rise and resend the refused word (`retry_refused`, `max_retries=3`, blocked nodes), `tests/test_input_refusal.py`; **copy 8 read (413616): a request's false rail running at 34 steps (mix-B drift + weight draw) survived its 3 × 0.75 kill train** — the train has no margin against a fast latch; measured in RefSim, stronger kill trains measured and **rejected** (4 × 1.5 stops the control machine; 3 × 1.5 stalls 85/100 mix-B copies, Juno 413672) — train stays 3 × 0.75, the margin problem is recorded in `tests/test_kill_margin.py` as an open item, `tests/test_kill_margin.py`; review (sol) → resend gated on the word's commit, next word only after the previous commit, READY over an uncommitted word = lost word (resent), and the stage reset now clears the producer's TIMEOUT latch (`protocol/handshake.py`); old-build baseline seeds 109/110 = 4 + 2 failing copies incl. **a new silent wrong-value mechanism (seed 109 copy 61: `c9_sel` commits its initial value early, state lags a token; capture 413685 queued)**; fixed build seeds 108–110: 1 / 4 / 8 failing copies (old build 4 / 4 / 2; realization-dependent; 0 refusals occurred), both wrong-value mechanisms read and fixed: **copy 61 = a state master's completion root gate at 65 % for the whole run, one stray coincidence lights it for good (power-up veto latch, `tests/test_state_master_powerup.py`)**; **copy 55 = the commit request's idle rail re-lit 54 ms after its kill fails to latch, the guard reads the dark pair as free and issues a second commit, the producer runs one token behind (delayed second ignition, `tests/test_commit_request_reignition.py`)**; **validated: seeds 108–110 with both fixes = 4,800 outputs, 0 wrong, 0 refusals, 12 stalled copies (4 %)**; stalls remain: copies 8 and 73 are the kill-margin class; three attempts to widen it (4 × 1.5, 3 × 1.5, 4 × 0.75 trains; a conditional 1.5 × second clear) each traded stalls for silent wrong values (a SEL on a stale condition) and are off; read (s108 copy 5 capture): a clear train reaching START's re-light of the same rail; with the re-light delayed 5 hops and a 4 × 0.75 train everywhere seeds 108–110 ran **300/300 copies clean** — but that build double-STARTs multiplier rows (CI) and the fourth pulse breaks the control machine (CI), then (2026-09-23/24) guards that require the true rail (sol → review → astra), the re-light delay and four pulses on the kernel's kill trains with the machine's at three: **seeds 108–110: 299/300 copies, 4,799/4,800 outputs, 0 wrong**, all suites passing, Kimi-reviewed, merged (`docs/tick_stalls.md`); 1,000-tick exit: see the Stage D rows below | `docs/tick_stalls.md`, `tests/test_stall_diag.py` |
| §6 Stage D scoring and reporting discipline | scoring fixed (`c9f5a83`, sol): per-field commit order with count and causal lower-bound checks; the earlier six-tick false failure treated input loads as output boundaries in a pipelined kernel. Inputs are not paced. Reports keep completed/matched, wrong, faults, timeouts and stalled copies separate | `docs/tick_stalls.md` |
| §6 Stage D 1,000-tick exit | nominal `tick2.c` passed 1,000 ticks (Juno 425032). Old-build mix B × 100 (Juno 425033): 91 copies matched all 1,000 ticks, 9 stalled, 5 run-level faults, 0 wrong in 93,917 ticks. Head-to-head mix B, seed 108, 100 copies × 1,000 ticks, `ad4046d`, same scoring: default (Juno 426357) 96 matched, 4 stalled at copy/tick 18/1, 34/31, 74/323, 17/728; 97,083 matched ticks, 0 wrong, 1 fault; `rate_robust` (Juno 426358) 95 matched, 5 stalled at 43/450, 28/535, 34/599, 24/654, 11/864; 98,102 matched ticks, 0 wrong, 1 fault. `rate_robust` does not reduce Stage D stalls; most stalls have no fault and the mechanism is unidentified. **Stage D mix-B exit not met** | `docs/tick_stalls.md` |
| §6 tick-failure diagnosis | Stage D exact-prefix replay tooling merged (`5c49025`): `stage_d --dump-copies`; compact atomic dumps bounded around each copy's last commit (`09f2a01`); `stall_diag`/`fault_diag` accept Stage D records. First replay (Juno 428569) reproduced stalls (98/100 done) but exhausted home quota writing a per-spike-role dump; rerun (Juno 428908–428910) was lost in the Juno filesystem outage at about 16:20 2026-09-30 (all logs froze). Replays queued: Juno 429332–429334, G2 3979025 | `docs/tick_stalls.md`, `tests/test_stall_diag.py` |
| §7 report recommendation and speed wording | done 2026-09-17; Stage D reporting discipline updated with the scoring fix and separate completion, wrong-value, fault, timeout and stall counts | `docs/campaign_report_2026-09-17.md` §6, §3; `docs/tick_stalls.md` |
| §8 four-configuration final comparison | **done** — primitive level all four configurations (413148/413419/413149); render level Q1, Q2 and Q4 (Juno 413470, 10 h 25 min: specialized datapath −10–12 % neurons, 1.07–1.13× wall over Q2; doom4 single copy finished every pixel), Q3 not run by design | `docs/perf_final_comparison.md`, `docs/perf/juno-h200-q4-small-fast-specialized.md` |
| §6 Stage D mix-B exit, Campaign D | `tick2.c`, 1,000 ticks × 100 copies, mix B, `--rate-robust --experimental-autapse`; initial generic seeds 108 (Juno 443026) and 109 (Juno 443063), `f71538c`/merged `28caa66`, each 100/100 and 100,000/100,000 matched, 0 wrong/faults/timeouts/stalls/refusals/retries; integrator `stage_d --recheck` verified both records and exactly-once/count evidence. 30,643 neurons, +6,430 autapse synapses versus rate-reader build; nominal per tick ~5,645 ms. After seeds 108 and 109 (superseded by the 500-copy totals below), zero stalls/200 gives exact one-sided 95% per-copy 1,000-tick-stall upper bound 1.4867%; one-sided Fisher 0/200 versus pooled seed-108 no-autapse baselines 15/300 p=0.00040724, or versus 16/400 including campaign C p=0.00137452. Additional commit `63c41f8`: generic seed 110 (Juno 449070) **exit not met**: 99 matched, copy 82 stalled after 352 completed ticks, 1 run-level fault, 99,352 matched ticks, 0 wrong, 648 missing, 0 refusals/retries; generic seed 111 (Juno 449071) **exit met**, 100/100, 0 wrong/faults/timeouts/stalls; specialized-datapath seed 108 (Juno 449072) **exit met**, 100/100, 0 wrong/faults/timeouts/stalls, 26,683 neurons. Across five 100-copy campaigns: 4/5 exits met; 1/500 stalled (0.2%), 0 wrong in 499,352 matched ticks. One-sided 95% Clopper-Pearson upper bound for per-copy 1,000-tick stall probability: 0.9452282208%; one-sided Fisher exact 1/500 versus pooled no-autapse seed-108 baselines 15/300: p=0.00000344134 and versus 16/400 including campaign C: p=0.00001965498. Autapse reduces the observed stall rate about 25×, not to zero; seed-110 copy-82 spike-dump replay is queued. Different seeds, and topology changes that redraw noise by index, compare rates rather than paired copies. Nominal: `tick2.c`, 1 copy, mix none, seed 1, 1,000 ticks, `--rate-robust --experimental-autapse`, Juno 448740, commit `474a8ee`: exit met; 1,000/1,000 ticks matched, 0 wrong, faults, timeouts, missing, refusals or retries; 30,643 neurons, 5,736.6 s neural, 3 h 10 min wall on one H200. The configuration met the mix-B exit on generic seeds 108, 109 and 111 and specialized seed 108, but not generic seed 110. Autapse remains experimental and not default; isolated 44-step floor and reload-margin requirements remain failed; toy tick does not change. Keep the reviewed kernel-construction-order restriction and experimental-autapse version key | `docs/stage_d.md`, `docs/tick_stalls.md`, `docs/stable_latch.md` |
| §6 Stage D mix-B exit, Campaign E | `tick2.c`, 1,000 ticks × 100 copies, mix B, `--rate-robust --experimental-autapse --zero-once --robust-request-clear --experimental-register-reset`, commit `831016a`: Juno H100 seed 110 (449853) and seed 112 (449854) each **met exit**, 100/100, 0 faults/timeouts/wrong/missing, not truncated; seed 110 build: 30,487 neurons. Total 2/2 campaigns, 0 stalled/200 copies. |
| §6 Stage D mix-B exit, Campaign F | `tick2.c`, 1,000 ticks × 100 copies, mix B, `--rate-robust --experimental-autapse --ordered-xor-oneshot`, commit `c2464d7` (merged `da15590`): Juno H200 seed 110 (451082; 31,363 neurons) and seed 113 (451083) each **met exit**, 100/100, 0 faults/timeouts/wrong/missing, not truncated. Total 2/2 campaigns, 0 stalled/200 copies. Exact one-sided 95% binomial upper bounds are `1 - 0.05^(1/200) = 0.0148670392` (1.4867039%) for F and `1 - 0.05^(1/400) = 0.0074613555` (0.7461356%) for E+F pooled; E and F are different builds. One-sided Fisher exact F 0/200 versus pooled no-autapse baselines 15/300: p=0.000407242. Nominal: `tick2.c`, 1 copy, mix none, seed 1, 1,000 ticks, `--rate-robust --experimental-autapse --ordered-xor-oneshot`, Juno H100 451587, commit `ded224b`: exit met; 1,000/1,000 ticks matched, 0 wrong, faults, timeouts, missing, refusals or retries; 31,363 neurons, 4 h 23 min wall on one H100. Stage D exit is met with configuration F: nominal plus 100-copy mix B on seeds 110 and 113. F is the leanest configuration that met exit on every seed tried: opt-ins `experimental_autapse` and `ordered_xor_oneshot`, plus `rate_robust`; it remains experimental/non-default, has two seeds per configuration, is toy tick only, and the default-build decision is the user's. Added 2026-10-12: configuration F, same flags, records built at commit `caf043e`. Generic seed 114, Juno H200 (451848), **met exit**, 100/100 matched, 100,000 matched ticks, 0 faults/timeouts/wrong/missing/duplicates/refusals/retries, not truncated; 31,363 neurons / 64,238 synapses, 5,824.4 s neural, 50,160.7 s wall. Generic seed 115, Juno H100 (451849), **met exit**, same zero counts; 5,819.2 s neural, 56,744.5 s wall. Specialized-datapath seed 113, Juno H200 (451850), **met exit**, same zero counts; 27,083 neurons / 54,966 synapses, 5,803.4 s neural, 44,332.1 s wall. Generic F totals 4/4 campaigns, 0 stalled / 400 copies; one-sided 95% upper bound `1 - 0.05^(1/400) = 0.0074613555` (0.7461356%); one-sided Fisher exact 0/400 vs the 15/300 no-autapse pool p=2.4628e-6. Generic plus specialized totals 0/500: bound 0.5973552%, Fisher p=3.2607e-7; these pool two builds. |
| §6 configuration F render campaign | `examples/render.c` column kernel, `drosophilos.bench.kernel_campaign render`, flags `--rate-robust --experimental-autapse --ordered-xor-oneshot`, torch-fast on H200, commit `d66d3f8`; 11,768 neurons / 24,228 synapses, 1,000 passes × 8 outputs per copy. Calibration Juno (451886; commit `98c39ba`; mix B seed 110, 100 copies, 2 passes): 1,600/1,600, 0 faults/timeouts/bad outputs, 22.18 s neural, 99 s wall. Nominal Juno (451992; 1 copy): 8,000/8,000 ok, 0 wrong/missing/extra/faults/timeouts/refusals/retries, 9,254.8 s neural, 16,342 s wall. Mix B seed 110, 100 copies, Juno (451993): 800,000/800,000 ok, same zero counts, one-sided 95% upper bound 3.7447e-6 per output, 9,323.4 s neural, 38,654 s wall. Mix B seed 113, 100 copies, Juno (451994): 800,000/800,000 ok, same zero counts, 9,341.0 s neural, 38,586 s wall. Both seeds pooled: 0 non-ok / 1,600,000 outputs, bound `1 - 0.05^(1/1600000) = 1.8724e-6`. `kernel_campaign` applies no input arrival jitter; this is a column/ROM composition campaign, not a full game, RAM or paced-kernel campaign; widths above 8 and array MUL above 2 bits stay unqualified. |
| §6 configuration F array-MUL kernel campaign (mul8) | Configuration F (`--rate-robust --experimental-autapse --ordered-xor-oneshot`), merged `9aec10c`; `drosophilos/bench/kernel_campaign.py` `mul8`, 8-bit wrap-around kernel `c1 = MUL(input, 5)`, `c2 = ADD(c1, 3)`, `c3 = MUL(c2, input)`, tokens [0, 1, 7, 13, 51, 128, 200, 255], 16,674 neurons / 35,703 synapses, torch-fast on H200. Width-8 primitive qualification at `07e28ed`: Juno 452953 nominal 512 trials and 452954 mix B 256 trials, all counters 0; Juno 453007 width-4 and width-8 kernel nominal tests passed at `3f4f8ba`. Calibration Juno 453152 (mix B seed 110, 100 copies, 2 passes, `--max-ms 60000`) was truncated by the 60 s neural cap: 1,299 ok, 301 missing because unfinished, 0 wrong, 0 faults, 0 timeouts; about 4.4-4.6 s neural per output. Long runs therefore used 250 passes (2,000 outputs per copy), not 1,000, with `--max-ms 14000000`. Nominal Juno 453175 (1 copy, 250 passes): 2,000/2,000 ok, 0 wrong/missing/extra/faults/timeouts/refusals/retries, 8,786.7 s neural, 16,010 s wall. Mix B seed 110, 100 copies, Juno 453176: 200,000/200,000 ok, same zero counts; one-sided 95% upper bound 1.4979e-5 per output; 8,816.3 s neural, 45,146 s wall. Mix B seed 113, 100 copies, Juno 453177: 200,000/200,000 ok, same zero counts; 8,803.4 s neural, 44,412 s wall. Both seeds pooled: 0 non-ok / 400,000 outputs, bound `1 - 0.05^(1/400000) = 7.4893e-6`. `kernel_campaign` applies no input arrival jitter; this is one small 8-bit kernel; widths 3, 5-7 and above 8 remain refused; 16-bit kernels are unqualified. |
| §6 configuration F width-16 primitive surveys | 44/44 cases passed with zero failure counters at commit `a27f31f` on Juno CPU (jobs 453991-454038). Largest cases: ADD/SUB 4,096 transactions each condition; RAM 256 writes / 1,024 reads with COPY-after-clear 561-728 steps; MULP 32 products (512 row transactions); array MUL 512 nominal / 256 mix B. Longest jobs: array MUL nominal 4 h 59 min, MULP nominal 3 h 21 min. F gate unchanged at width 8; see `docs/stable_latch.md`. |
| §6 configuration F 16-bit perspective campaigns and repairs | `render2.c`, 2 passes × 8 outputs per copy, 100 copies per seed, mix B. F (`rate_robust + experimental_autapse + ordered_xor_oneshot + f_width16_experimental`): 8 frozen / 500 copies (1.6%), 0 wrong/faults/timeouts. F + `master_ignition` v2 (merged `c841ba2`): 5 frozen / 900 (0.56%), 0 wrong/faults/timeouts. Both repairs (`master_ignition` v2 + `operand_xor_oneshot` v3, merged `cd28765`): 0 frozen / 1,200, all 19,200 outputs correct; one watchdog timeout in seed 121 copy 99, resent and completed correctly (1 refusal, 1 retry); one-sided 95% upper bound 0.2493%; one-sided Fisher exact vs F p=5.38e-5 and vs F + `master_ignition` p=0.0144. Default without F: 2 frozen / 200 plus 1 fault. Freeze diagnoses: seed 110 copy 75 master partner ignition failure; copies 85 and 57 operand-XOR double rail; copies 43 and 51 stalled at the same datapath phase, mechanism not traced. Limits: 2-pass calibration-length runs, not 1,000-pass campaigns; 11/12 both-repairs seeds ran fp32 on RTX 3090 (seed 115 fp64 H200); changing build options redraws every copy's noise, so comparisons are rate-based, not paired; both options remain opt-in and experimental, and width 16 still requires `f_width16_experimental`. See the campaign table in `docs/stage_d_f16_stall.md`. |

---

# DrosophilOS Performance Campaign — Specific Instructions for the Next Campaign

**Source note:** These recommendations are based on the 17 September 2026 DrosophilOS campaign report and the repository audit discussed in this conversation.

Replace the current “write it up, then build E2” recommendation with a performance campaign that has two implementation tracks: faster simulation of the existing circuits, and smaller exact circuits. Keep E2 as a measured follow-on rather than assuming it is the only route to acceptable speed.

## 1. Change the immediate objective and limit the scope

Use this as the next campaign’s objective:

> Reduce the wall-clock cost of the existing neural world-update and rendering workloads while preserving their computed results. Measure simulator throughput separately from neural circuit latency. Establish which changes improve complete-frame performance, which only reduce circuit size, and which introduce correctness or reliability regressions.

Keep the current neuron model, timestep, arithmetic semantics, scenes, and reference outputs unchanged for the initial comparisons. Keep the current array multiplier and request-repair mechanism as the baseline.

Defer additional gameplay features, higher-resolution demonstrations, multi-GPU expansion, and self-hosted compilation. Those remain roadmap items, but they should not consume the primary performance effort. Continue the feasibility write-up and tick-failure diagnosis alongside this work. Stage D’s 1,000-tick requirement and the missing C/F/F2 components remain incomplete; a speed improvement does not complete them.

## 2. First implementation: repair the benchmark and record the missing timing evidence

**Files:** `drosophilos/bench/render_doom.py` and the runners in `drosophilos/lib/kernel.py`.

### Make backend selection explicit

Add a backend choice independent of copy count: reference simulator or Torch simulator. A Torch run with one copy must still use Torch on the requested device. Reject unsupported backend/device combinations rather than silently falling back. Record the effective simulator class, device, and dtype in both the console output and saved report.

This fixes a concrete defect: the current one-copy branch calls `run_pipeline` without passing the requested device or dtype, while its summary can still print the requested device.

### Separate timing into distinct measurements

Record:

- compilation and circuit construction;
- image loading and startup;
- time to the first completed frame;
- subsequent frame-completion intervals;
- total run time;
- for each runtime input, when it was injected;
- when the resulting state committed;
- when the first frame using that state completed.

Do not use the periodic progress callback to timestamp frame completion. Add an observer callback that fires when the runner has collected the last required valid pixel. Record the neural completion step and the wall time at which the result becomes available to the host separately.

Report any final tick after the last displayed frame as a separate tail, rather than treating it as rendering latency.

### Add a reproducibility record

Each result must identify:

- source and commit hashes;
- compiled circuit hash;
- inputs and initial state;
- pacing mode;
- multiplier;
- copy count;
- resolution;
- neuron and edge counts;
- model parameters;
- perturbation configuration;
- seed;
- effective backend and dtype;
- hardware and software versions.

Save requested and completed frame counts separately.

Preserve these outcomes separately:

- wrong values;
- missing values;
- duplicates;
- invalid output words;
- neural faults;
- host-detected stalls;
- externally truncated runs.

Missing counters are unknown, not zero.

### Use profiling without turning every benchmark into a trace dump

Add named profiling regions for:

- numerical integration;
- spike delivery;
- device-to-host observation;
- host scheduling;
- output decoding;
- image writing.

Profile short representative intervals. Obtain headline timings from separate unprofiled runs.

Use synchronized measurement boundaries or CUDA events for GPU timings rather than timing asynchronous submissions alone.

### Required output

Produce a reproducible baseline that shows where wall time is spent, with actual frame-completion timestamps — not another aggregate JSON containing only total run time.

## 3. Replace repeated long renders with a small, fixed comparison suite

Add a campaign driver, for example:

`drosophilos/bench/perf_campaign.py`

It should run explicitly selected configurations and write one comparison report.

Use three levels:

| Test level | Proposed workload | Purpose |
|---|---|---|
| Primitive and composition | Fixed-operation cells, a fan-out kernel, a multiplier kernel, and the existing tick kernel; enough inputs to observe repeated operation | Measure latency, sustained throughput, reset behavior, and backpressure |
| Small complete render | Start with three-frame, 8 × 5 versions of `doom2.c` and `doom4.c`, using a fixed input sequence | Exercise complete phase transitions and expose end-to-end regressions without immediately repeating the largest runs |
| Historical-size confirmation | The existing 40 × 25 textured workload and 24 × 15 sprite workload | Confirm whether a promising local improvement survives the original workload scale |

The proposed small samples must first pass a coverage check:

- the selected inputs must change state;
- later frames must reflect those changes;
- the sprite test must actually contain visible sprite pixels.

Adjust the sampling or input sequence when it does not.

Within each comparison, change only one implementation choice. Keep the following identical:

- source;
- inputs;
- copy count;
- precision;
- pacing;
- sampled pixels.

Do not compare a smaller render or lower-precision run against the old configuration and call the difference an implementation speedup.

Start with one and eight copies on the same effective backend. Add more copies only to answer a specific scaling question.

Repeat the short unprofiled measurements and report their spread. Reserve historical-size runs for configurations that have already passed the smaller tests.

### Required output

Provide separate comparisons for:

1. unchanged-circuit simulator speed;
2. changed-circuit performance;
3. their combined effect.

## 4. Run two optimization tracks after establishing that baseline

# Track A: remove avoidable host interaction from the simulator

**Files:** `drosophilos/sim/lif_torch.py`, a separate optimized backend under `drosophilos/sim/`, and `run_pipeline_batched`.

The current timestep extracts spikes with `torch.nonzero`, copies spike indices to NumPy, reads a GPU-derived scalar into Python, and constructs variable-sized delivery arrays. These are concrete optimization targets, although their contribution to total runtime still needs measurement.

Make the changes in this order.

### A1. Separate simulation from observation

Introduce an explicit observation interface instead of requiring runners to inspect:

- `_spk_step`;
- `_spk_node`;
- `_spk_neuron`.

Retain full tracing for reference/debug runs.

Add a filtered mode that retains only the spikes needed for:

- READY detection;
- output completion;
- output-rail decoding;
- fault/timeout observation;
- any requested diagnostic capture.

Do not disable internal spike propagation. Only reduce what is copied and retained for observation.

The current runner relies on those lists for both control and decoding, so deleting the copies without replacing that interface would break execution.

### A2. Prototype device-resident delivery and compact observation buffers

Avoid extracting variable-sized spike lists and scalar counts to Python every step.

Use preallocated device-side storage with explicit overflow detection.

An overflow must produce either:

- a reported failure; or
- a controlled fallback.

It must never silently discard spikes.

### A3. Combine numerical operations into fewer GPU kernels

Preserve the existing:

- integration order;
- threshold order;
- synaptic-delivery order;
- reset order;
- delayed events;
- integer synaptic accumulation;
- refractory behavior;
- silencing;
- perturbation semantics.

Keep the existing simulators available as comparison backends.

### A4. Only then consider multi-step execution or CUDA graphs

Multi-step execution must preserve:

- external-input timing;
- observation semantics;
- READY behavior;
- completion behavior.

Do not simply poll READY every hundred steps and describe the result as the same execution schedule.

CUDA graphs are not a wrapper that automatically fixes the present loop. CPU synchronization and dynamic shapes must first be addressed.

### Validation for Track A

For this track, the circuit must remain identical.

Compare:

- voltage/current trajectories where applicable;
- spike events under the applicable backend contract;
- decoded transactions;
- complete outputs.

Any floating-point or random-stream differences must be documented rather than described as bit-identical.

### Expected outcome

Lower wall-time cost for the same neural work.

If profiling shows that memory traffic or active-synapse delivery dominates instead, optimize that measured component rather than continuing a Python-overhead hypothesis.

# Track B: specialize fixed-operation cells before redesigning the protocol

**Files:** `drosophilos/lib/kernel.py`, particularly `add_alu_logic_tokens`, `_unit_rails`, their callers, and `load_pipeline_image`.

The current ALU builder constructs the adder, overflow logic, AND, OR, XOR, and pass-through paths, then selects a result — even though a resident cell’s operation is fixed.

STORE’s pass-through path also calls this general builder.

Add an experimental generic-versus-specialized datapath option and retain the generic implementation for comparison.

### B1. Start with AND, OR, XOR, and MOV

Instantiate only:

- the required operation;
- the required flag outputs.

Remove:

- unused datapaths;
- unnecessary unit-selection hardware.

Update image initialization so it does not assume removed unit-select neurons still exist.

Keep the external:

- request;
- sampling;
- staged commit;
- completion;
- reset;
- fault behavior

unchanged in the first version.

In particular, generate valid flags on every transaction. Do not replace a per-transaction zero flag with a constant level.

### B2. Then test direct LOAD/STORE data paths

Do not shorten STORE completion until a neural dependency guarantees the RAM write has finished.

The current implementation explicitly relies on the longer cell completion occurring after the write. Removing a general ALU could otherwise remove timing that currently protects the memory operation.

### B3. Test specialized cells rigorously

Test:

- exhaustive small-width values;
- wider boundary cases;
- wider random cases;
- repeated identical values;
- alternating values;
- stalled consumers;
- fan-out;
- reset/restart behavior.

Follow these with the existing noisy composition campaign.

Measure separately:

- neuron count;
- edge count;
- spike traffic;
- neural latency;
- initiation interval;
- wall time.

A smaller circuit that leaves neural latency unchanged can still be useful, but report that result accurately.

Only after this comparison should the campaign attempt:

- range-proven width reduction;
- retained column intermediates;
- neural predication;
- larger exact blocks with fewer handshakes.

Those are separate experiments. Do not combine them into the first specialization patch.

### Expected outcome

Fewer simulated components without initially changing the transaction protocol.

The magnitude of any frame-time improvement remains unmeasured until the matched campaign is run.

## 5. Turn the multiplier regression into a bounded diagnostic task

Keep the array multiplier as the default.

The current results show a standalone throughput improvement from the pipelined multiplier but worse full-workload behavior. The cause is not identified.

Run array and pipelined variants on an identical:

- input stream;
- surrounding kernel;
- source;
- backend;
- copy count;
- dtype;
- pacing mode.

Capture selected events from:

- the multiplier;
- its immediate producers;
- its immediate consumers.

Capture:

- request arrival;
- START;
- operand sampling;
- stage completion;
- commit;
- DONE;
- consumer readiness.

Separate four quantities:

1. time to the first result;
2. intervals between subsequent results;
3. time waiting for downstream consumers;
4. time draining the phase.

Use enough tokens to distinguish pipeline fill from sustained throughput.

Do not change timing constants while collecting this comparison.

First determine whether the loss comes from:

- producer starvation;
- downstream backpressure;
- operand retention;
- phase boundaries;
- the multiplier itself.

Correct the campaign report’s reference to inter-commit intervals in `doom4_24na_h200.json`: that saved file contains aggregate counters, not the event history needed for this diagnosis.

### Required output

Produce:

- an event-level explanation;
- a minimal reproducer.

A further long run that merely confirms “pipelined is slower” does not answer the open question.

## 6. Preserve the reliability campaign, but distinguish correctness from throughput

Run tick-failure diagnosis in parallel with the performance work.

Replay a recorded failing seed/copy and capture only the relevant:

- request;
- repair;
- clear;
- completion circuitry.

Turn the localized failure into a regression test before changing its mechanism.

Keep the established 100-copy mix-B comparison for handshake changes.

Add independent seeds after the original reproducer passes.

Report separately:

- requested transactions;
- completed transactions;
- wrong values;
- duplicates;
- detected refusals;
- unfinished transactions.

A faster run that silently drops work is not faster execution.

Classify a run stopped by its resource limit as **truncated**, not automatically stalled.

Establish stall thresholds from the tested contract and workload.

Complete the 1,000-tick canonical-state comparison on the explicitly named world-update program.

A toy tick result must remain labeled as such rather than being generalized to `doom4`.

Keep timestep refinement and CUDA transaction-level certification as explicit outstanding validation work.

## 7. Revise the campaign recommendation and final report

Replace Section 6’s recommendation with:

> Write up the demonstrated feasibility result now. In parallel, establish an instrumented performance baseline, improve simulator execution without changing the neural computation, and test operation-specific exact datapaths. Diagnose the multiplier regression and remaining tick failures with bounded traces. Use the resulting complete-frame measurements to determine the scope of E2 and any later cluster expansion.

Change the speed discussion from:

> wall time is GPU throughput

to a distinction between:

- **measured neural workload cost**; and
- **not-yet-profiled simulator cost**.

The current aggregate timings do not establish their detailed attribution.

Keep the E2 estimate of 10–50× clearly labeled as a proposed gain, not a measured result.

Begin E2 with one bounded primitive and include:

- encoding;
- settling;
- decoding;
- uncertainty detection;
- neural fallback

in its benchmark.

Do not replace the exact renderer until the complete-path comparison supports doing so.

## 8. Required final comparison

The campaign’s final comparison should contain four configurations:

| Configuration | Purpose |
|---|---|
| Original simulator + original circuits | Baseline |
| Optimized simulator + original circuits | Measures simulator-only improvement |
| Original simulator + specialized circuits | Measures circuit-only improvement |
| Optimized simulator + specialized circuits | Measures combined end-to-end improvement |

For each configuration, report:

- first-frame latency;
- subsequent frame intervals;
- neural time;
- wall time;
- wall/neural ratio;
- circuit neuron count;
- circuit edge count;
- spike count or bounded spike-traffic metric;
- wrong outputs;
- missing outputs;
- duplicates;
- invalid outputs;
- faults;
- timeouts;
- truncation status.

## 9. Immediate implementation order

The first code changes should be:

1. **Fix benchmark/backend selection.**
2. **Add instrumentation and frame-completion timing.**
3. **Add a fixed small performance campaign.**
4. **Profile the current Torch simulator.**
5. **Implement the observation abstraction.**
6. **Prototype device-resident spike delivery/observation.**
7. **Implement specialized fixed-operation ALU datapaths.**
8. **Diagnose the pipelined multiplier with event-level timing.**
9. **Re-run the matched small complete-render suite.**
10. **Only then run the historical-size confirmation workloads.**
11. **Use those results to decide the scope and priority of E2.**

The next result should be an attributable before-and-after speed measurement — not another larger image or a new gameplay feature.
