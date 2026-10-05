# Stage D completion stalls: seed 108, base 17/74 and rate-robust 11/24

**Current status:** the register-reset opt-in is withdrawn after review. See
[the qualification frontier](#register-reset-qualification-and-opt-in-2026-10-04)
for the biased recovery failure, revised experiments and release decision.

**2026-10-04 update:** an isolated reconstruction from each copy's static draws
names the failing primitive for base 74, base 17 and RR 24: an old master rail
survives the four-tap master reset, entrained like copy 28's request latch. It
then blocks its valid latch's re-ignition. RR 11 is not reproduced. See
"Isolated reconstruction from static draws" below; the text before that section
is the original capture-only analysis.

All four copies issue a **commit command**, clear their pending commit request, and
produce master READY, but never produce DONE for that transaction. Their captured
DONE inhibitors stop firing during the master rewrite and never resume. This points
to a missing renewed master completion, rather than a DONE inhibitor that remains
lit. **The failing storage/copy/completion primitive is not established:** the dumps
omit COPY, copy arms, master rails, master validity and completion, and register reset
spikes. The exact recaptures below are needed before recommending a fix.

The four copies share this observable stopping point. They are **not yet demonstrated
to share one initiating mechanism**. Base copy 74 additionally acquires persistent
master-bit-3 fault-gate activity; the other three do not. That distinction requires
rail evidence, not an assumption that the fault gate proves a double rail.

## Artifacts, rebuild and evidence limits

Read alongside [tick stall methods](tick_stalls.md),
[Stage D section 7](stage_d.md#7-diagnosing-a-stalled-copy),
[the datapath stalls](stage_d_datapath_stall.md), and
[the seed-109 false-fault recapture](a2/datapath_fault_s109_n73.md).

Inputs were read by absolute path under
`/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:

| Build / copy | Campaign JSON; dump suffix | Inclusive step window | Spikes | Captured IDs / firing IDs |
|---|---|---|---:|---:|
| base 17 | `mixB_s108_c100_replay740.json`; `_copy17.npz` | 43,114,738–45,271,503 | 35,274,883 | 5,674 / 5,188 |
| base 74 | same JSON; `_copy74.npz` | 18,799,802–20,956,617 | 36,844,315 | 5,674 / 5,159 |
| rr 11 | `mixB_s108_c100_rr_replay870.json`; `_copy11.npz` | 48,881,321–51,005,503 | 40,906,866 | 6,037 / 5,560 |
| rr 24 | same rr JSON; `_copy24.npz` | 36,915,114–39,039,303 | 41,717,091 | 6,037 / 5,542 |

The corresponding input reports are `reports/base_copy17.md`, `base_copy74.md`,
`rr_copy11.md` and `rr_copy24.md`. The named dump suffixes attach to the JSON stem,
not to `.json`.

`stall_diag.build_tick_pipeline` rebuilt the recorded options, and `load_dump`
validated every observed role at its recorded neuron ID: **zero ID drift** in all
four. Base is **29,375 neurons / 52,808 synapses**, the supplied default 09f2a01
netlist equivalent to ad4046d. RR is **30,643 / 55,936**, with
`RATE_ROBUST_VERSION=2`. Both have the generic datapath, true guards v2,
`copy_requires_rail=True`, delayed request repair, commit-request reignition,
four-pulse kernel/request kills at 0.75, ACT 11 hops and IDLE 20 hops.

Steps are 0.1 ms. Latch trains below use gaps ≤141 steps; gate spikes are inspected
individually. The reports' phase-stall thresholds are respectively 22,542, 23,166,
21,236 and 23,738 steps. The missing DONE persists **186.3089, 184.9461, 188.1368
and 184.4292 seconds after the commit command**, far beyond those thresholds.
The windows include 30 neural seconds before the copy's last observed output commit
and 10 seconds after stall detection, not merely 40 seconds around this cell.

The actual filter is:

```text
\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)
```

`capture_ids`, not just the list of firing neurons, establishes which silent roles
were recorded. In particular, `M.ready*`, `M.fault*` and both `done.*` neurons are
captured, but **`M.comp*`, `M.valid*`, Q/M data rails, Q/M resets, `guardd`, `grant`,
`copy` and `cp*` are unavailable**. A missing spike from these omitted roles is not
evidence of silence. Validity OR indicators do not prove exclusive or correct data.

## What DONE actually waits on

The diagnostic's `commit` column is **`cell.commit_pulse`**, the command emitted
by the reader-free commit guard in `lib/kernel.py:gate_commit`. It is not proof
that a complete new word reached the architectural master. The completion path in
`lib/staged.py:add_staged_commit` is:

```text
Q complete -> autocommit -> creq[1] -> reader-free guard -> commit_pulse
    -> commit_in -> 12-hop guardd -> grant -> M.reset -> 15-hop READY chain
    -> M.ready -> COPY -> selected-stage-rail copy arms -> M data rails
    -> M bit-valid OR latches -> M completion tree -> W_M
    -> done.edge -> Q.reset -> Q.ready
                 -> 20-hop idled -> idle[1]
```

The commit pulse also lights `creq[0]` (nothing pending), kills `creq[1]`, and
re-ignites `creq[0]` after 16 hops. **None of these is an input to DONE.** Stage
clear, stage READY and IDLE re-arm are consequences of DONE, not conditions it
waits for. Master READY is a timed pulse from `M.reset`, not an emptiness check.
Its observed chain therefore supports reset initiation without proving successful
clear of every master latch. The observed normal command-to-master-READY timing
is also evidence against a completely lost grant/reset command.

These are the exact nominal critical-path IDs; u/v pairs are latch members:

| Role | base 17: c4_sel | base 74: c5_sub | rr 11: c10_xor | rr 24: c2_sel |
|---|---:|---:|---:|---:|
| `commit_pulse` | 28143 | 28341 | 30107 | 29175 |
| `commit_in` | 3524 | 4101 | 7986 | 2698 |
| `guardd.d11` (unavailable) | 3536 | 4113 | 7998 | 2710 |
| `grant.edge` (unavailable) | 3539 | 4116 | 8001 | 2713 |
| `grant.L` u/v (unavailable) | 3537/3538 | 4114/4115 | 7999/8000 | 2711/2712 |
| `M.reset` / `reset_inh` / `reset_edge` (unavailable) | 3500/3501/3502 | 4077/4078/4079 | 7962/7963/7964 | 2674/2675/2676 |
| `M.ready_delay0` / `M.ready` | 3506/3521 | 4083/4098 | 7968/7983 | 2680/2695 |
| `copy` u/v (unavailable) | 3542/3543 | 4119/4120 | 8004/8005 | 2716/2717 |
| `M.comp.c3_0.L` u/v (W_M; unavailable) | 3495/3496 | 4072/4073 | 7955/7956 | 2667/2668 |
| `done.edge` / `done.edge_inh` | 3632/3633 | 4209/4210 | 8094/8095 | 2806/2807 |
| `Q.reset` / `reset_inh` (unavailable) | 3277/3278 | 3854/3855 | 7697/7698 | 2409/2410 |
| `Q.ready_delay0` / `Q.ready` | 3283/3298 | 3860/3875 | 7703/7718 | 2415/2430 |
| `Q.comp.c3_0.L.u` | 3272 | 3849 | 7690 | 2402 |
| `creq[0].u` / `creq[1].u` | 3634/3636 | 4211/4213 | 8096/8098 | 2808/2810 |
| `idle[0].u` / `idle[1].u` | 3301/3303 | 3878/3880 | 7721/7723 | 2433/2435 |

All edges described here have **18-step synaptic delay (1.8 ms)**; whole-neuron
response latency is additional. A latch loop is +3621 quanta per direction.
`commit_pulse -> commit_in -> COMMIT.u`, `M.ready -> COPY.u`, ignition relays
into latches, and **W_M into each DONE neuron** use **+4655**. Delay-chain edges
use +3621. `grant.L.u -> M.reset` is +4655; its edge detector gets +3621 and
inhibits reset by −7966. Reset/kill pulses deliver **−2716** to both latch
members (0.75 × loop). `done.edge -> Q.reset` is +4655, and
`done.edge -> idled.d0` is +3621.

The only inputs to each `done.edge` are **W_M (+4655)** and its own
`done.edge_inh` (**−7966**). That inhibitor's **only synaptic input is W_M
(+4655)**. There is no DONE hold inhibitor driven by COMMIT, stage validity,
IDLE, a request or a fault gate. Neither DONE neuron is a rate-conditioned gate.
Thus RR retains this same edge-relay topology, despite changing the upstream
validity/completion readers. The quanta above are the rebuilt nominal wiring;
per-copy noisy weights and membrane states are not in these artifacts.

## Per-copy event evidence

| Event | base 17: c4_sel | base 74: c5_sub | rr 11: c10_xor | rr 24: c2_sel |
|---|---:|---:|---:|---:|
| Failed START | 43,403,114 | 19,101,769 | 49,119,495 | 37,190,672 |
| Q completion first spike | 43,406,188 | 19,106,388 | 49,123,340 | 37,193,523 |
| Commit command | 43,408,414 | 19,107,156 | 49,124,135 | 37,195,011 |
| `commit_in` | 43,408,455 | 19,107,198 | 49,124,173 | 37,195,052 |
| COMMIT latch u first spike | 43,408,496 | 19,107,239 | 49,124,217 | 37,195,095 |
| `creq[0].u` renewed train | 43,408,455 | 19,107,207 | 49,124,199 | 37,195,052 |
| `creq[1]` last u / v spikes | 43,408,511 / 43,408,547 | 19,107,302 / 19,107,266 | 49,124,261 / 49,124,228 | 37,195,146 / 37,195,113 |
| `commit.idle_again.d15` | 43,409,242 | 19,108,030 | 49,124,960 | 37,195,834 |
| `M.ready_delay0` | 43,409,259 | 19,107,985 | 49,124,996 | 37,195,847 |
| Last `done.edge_inh` spike | 43,409,354 | 19,108,141 | 49,125,043 | 37,195,945 |
| `M.ready` | 43,410,011 | 19,108,764 | 49,125,774 | 37,196,625 |
| Failed transaction DONE / Q READY / idled | none | none | none | none |

All 16 master READY-chain neurons fire once for the failed transaction.
All eleven Q valid-u latches, every Q completion-tree latch, both COMMIT members
and the busy `idle[0]` latch continue in uninterrupted trains through capture end.
`creq[0]` holds and `creq[1]` remains dark after the listed tail spikes. Every
target-cell Q fault gate and its fault latch is silent throughout the capture.

The configured four-tap kill controller produces **five inhibitor spikes nominally**,
including a trailing spike from residual charge. `add_kill_train(pulses=4)` and
`add_reset(pulses=4)` both do this in noise-free RefSim; the fifth spike is not a
noise-induced extra. Perturbations can suppress it (as in the request copy-28
clear), so tap count does not guarantee output spike count. These copies' recorded
`commit.kill.inh` trains are:

| Copy / inhibitor ID | Spike steps |
|---|---|
| base 17 / 28162 | 43,408,507; 43,408,555; 43,408,598; 43,408,643; 43,408,744 |
| base 74 / 28360 | 19,107,245; 19,107,289; 19,107,335; 19,107,384; 19,107,457 |
| rr 11 / 30126 | 49,124,223; 49,124,266; 49,124,312; 49,124,356; 49,124,424 |
| rr 24 / 29194 | 37,195,103; 37,195,150; 37,195,190; 37,195,239; 37,195,315 |

These −2716 edges target
`creq[1]`'s two members. Their observed extinction and the sustained `creq[0]`
train rule out a surviving pending-commit request as the blockage in these four
transactions. They do not test the unavailable master's reset train.

### Base 17: c4_sel

Previous DONE **3632** is at **43,358,034**, followed by Q READY **3298** at
**43,358,907** and an IDLE-true train beginning **43,359,119**. Failed START
**3672** consumes that readiness; IDLE-true u/v last fire at **43,403,206/235**.
After the command and master READY above, DONE **3632**, Q READY-chain head
**3283** and `idled.d0..19` **3652–3671** never fire. Q completion **3272**
holds to **45,271,487** and COMMIT **3522** to **45,271,494**.

The old DONE inhibitor train **3633** ends at 43,409,354, **65.7 ms before
master READY**. Every c4 master fault gate is silent. The strongest supported
localization is the **COPY/master-completion/DONE-input path after master READY**.
Whether COPY **3542/3543** failed to ignite, a copy arm/rail failed, or a master
valid/tree latch failed to clear or re-ignite is unavailable.

### Base 74: c5_sub

Previous DONE **4209** is at **19,052,463**, Q READY **3875** at
**19,053,341**, and IDLE-true rises at **19,053,549**. Failed START **4249**
leaves IDLE-true u/v dark after **19,101,908/874**. After master READY,
DONE **4209**, Q READY-chain head **3860**, and IDLE delay chain **4229–4248**
never fire. Q completion **3849** holds to **20,956,574**, COMMIT **4099** to
**20,956,588**. DONE inhibitor **4210** ends **62.3 ms before master READY**.

There is one additional observation: **`c5_sub.M.fault3.and`, 3967**, fires
**5,414 times**, first **19,109,292** (52.8 ms after master READY), then
19,109,674; 19,110,057; 19,110,442; …; last **20,956,487**. It never fired
before this transaction in the window. The other ten master fault gates remain
silent. This is a master bit-3 anomaly, not a captured stage-fault discard.

Its nominal inputs are **3908, `M.b3r0.u`, and 3910, `M.b3r1.u`**, each
**+211 quanta / 18 steps**. Both are omitted. This legacy rate-mode AND can
fire from one fast correct rail, as the seed-109 recapture demonstrated. A true
double rail, a false fault from a fast rail, or a rate disturbance following
rewrite cannot be distinguished here. The master fault gate has no wired
control output in this build: its activity itself does not veto DONE or reset Q.

The omitted bit-3 copy path is precise:

```text
COPY.u 4119 -> cp3r0.driver.edge 4145 -> cp3r0.edge 4147 -> M.b3r0.u 3908
           -> cp3r1.driver.edge 4149 -> cp3r1.edge 4151 -> M.b3r1.u 3910
Q.b3r0.u 3685 requires the r0 arm; Q.b3r1.u 3687 requires the r1 arm
opposite Q rail / faultL.u 3894 -> arm veto 4148 / 4152 -> arm inhibition
M.reset_inh 4078 -> both members of both M bit-3 latches
```

Driver edges get +4655 from COPY and −7966 from their driver inhibitors
**4146/4150**. The biased copy qualifiers (**−2 mV**) get **+2379** from the
driver pulse, **+291** from the selected stage rail, and **−543** from the
opposite-rail/fault veto interneuron; the veto's inputs are +3621. Arm output
ignition is +4655; master reset inhibition is −2716. Master valid3 OR **3961**
gets +766 from each rail, ignites valid latch **3962/3963** through edge **3964**,
with inhibitors **3965** (−7966) and **3966** (−905). These missing probes are
needed to identify the affected bit and any failed clear/ignition. Fault3 alone
does not explain the absent whole-word completion.

### RR 11: c10_xor

Previous DONE **8094** is at **49,077,465**, Q READY **7718** at
**49,078,334**, and IDLE-true rises at **49,078,548**. Failed START **8134**
leaves IDLE-true u/v dark after **49,119,650/612**. Q completion **7690** holds
to **51,005,476**, COMMIT **7984** to **51,005,470**. DONE **8094**, Q READY-chain
head **7703** and IDLE delay chain **8114–8133** never fire for the transaction.
DONE inhibitor **8095** ends **73.1 ms before master READY**. Every master fault
gate is silent. The missing input/path is again a renewed master completion
into DONE; COPY **8004/8005**, master root **7955/7956**, and intermediate
master probes are omitted, so a particular failed latch cannot be named.

### RR 24: c2_sel

Previous DONE **2806** is at **37,137,624**, Q READY **2430** at
**37,138,514**, and IDLE-true rises at **37,138,716**. Failed START **2846**
leaves IDLE-true u/v dark after **37,190,770/799**. Q completion **2402** holds
to **39,039,287**, COMMIT **2696** to **39,039,299**. DONE **2806**, Q READY-chain
head **2415** and IDLE delay chain **2826–2845** never fire for the transaction.
DONE inhibitor **2807** ends **68.0 ms before master READY**. Every master fault
gate is silent. COPY **2716/2717**, master root **2667/2668**, and intermediate
master probes are omitted. No bit-specific failed clear or ignition is established.

## Shared signature versus established cause

For comparison, the immediately preceding healthy master READY→DONE intervals
are **250.1, 265.2, 234.6 and 229.6 ms**. In those transactions the old DONE
inhibitor train also stops before master READY, then resumes near the new DONE.
In the failed transactions neither DONE neuron produces new spikes after READY,
for approximately another 183–188 neural seconds. Residual inhibition could
affect an early pulse; the capture does not measure membrane voltage. But the
evidence does **not** show a continuously firing inhibitor holding down a live
new completion root. It instead strongly suggests that W_M never re-asserts.
Because W_M is omitted, this remains an inference: simultaneous downstream
relay failures or a brief lost W_M ignition are not formally excluded.

| Proposed waiting condition | Evidence in all four |
|---|---|
| Commit request must clear | `creq[1]` clears; `creq[0]` holds; neither is a DONE input. |
| Master must reset and become ready | Master READY chain fires completely; actual latch clear is unrecorded. |
| New master completion must arrive | This is DONE's excitatory input; the root, its tree and copy path are unrecorded. No renewed DONE-inhibitor train is observed. |
| Stage must clear first | Ordering is DONE→Q reset. Q valids/root remain live; no Q READY chain occurs. |
| IDLE/READY must re-arm first | Ordering is DONE→idled→IDLE. The old IDLE-true train dies at START, while busy remains lit. |
| `done.edge_inh` must stop | It stops near master reset in all four and never resumes. Persistent inhibitor firing is not the observed mechanism. |

Thus the shared class is **commit issued / master rewrite started / no observed
renewed completion / stage and busy state retained**. Copy 74's master-bit-3
fault signature makes assuming an identical initiating failure especially
unwarranted. A surviving master rail could leave a stale or double-railed master;
a missing COPY/rail ignition or stale valid/tree edge detector could lose
completion. Master rails and master fault gates are not COPY-arm veto inputs.
None of these initiating mechanisms is demonstrated by these filters.

## Exact recapture and expected size

Run on the same recorded netlists and CUDA backend, retaining **all 100 copies**.
Changing batch size changes the device-drawn stray stream. These commands retain
the original global diagnostic roles and add complete Q/M domains, ACT and its
repair-delay chain, the
commit guard delay, grant, COPY and every copy arm of the two relevant cells.
They preserve the original replay tick counts and ceilings. No recapture was
executed locally.

```sh
python -m drosophilos.bench.stage_d --ticks 740 --seed 108 \
  --backend torch-fast --device cuda --copies 100 --mix B \
  --max-ms 6500000 --dump-copies 17,74 \
  --dump-roles '(?:\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)|^(?:c4_sel|c5_sub)\.(?:Q\.|M\.|act(?:\.|d)|guardd|grant|copy|cp[0-9]))' \
  --dump-out data/stage_d/mixB_s108_c100_completion740 \
  --out data/stage_d/mixB_s108_c100_completion740.json

python -m drosophilos.bench.stage_d --ticks 870 --seed 108 \
  --backend torch-fast --device cuda --copies 100 --mix B --rate-robust \
  --max-ms 7500000 --dump-copies 11,24 \
  --dump-roles '(?:\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)|^(?:c10_xor|c2_sel)\.(?:Q\.|M\.|act(?:\.|d)|guardd|grant|copy|cp[0-9]))' \
  --dump-out data/stage_d/mixB_s108_c100_rr_completion870 \
  --out data/stage_d/mixB_s108_c100_rr_completion870.json
```

The default window policy is still 30 s before the last observed output commit
and 10 s after detected stall. The extra role domains add **692 captured IDs**
to the base filter (6,366 total) and **818** to RR (6,855 total), including the
RR `.rate` readouts and their reset consumers. The two original base payloads
are about **269/281 MiB**, RR **312/318 MiB**, at eight bytes per spike plus small
role tables. Expect roughly **350–450 MiB per base dump** and **400–550 MiB per
RR dump**, about **1.5–2 GiB for all four**; reserve 2.5 GiB plus JSON records.
These are estimates: omitted trains' rates/activity cannot be measured from the
old filter. Even at the refractory ceiling, the newly selected IDs alone add
less than approximately 0.55 GiB per base file / 0.64 GiB per RR file over
these 213–216 s windows. Numeric step/neuron arrays remain int32 here.

The decisive readout is the first break in this chain, compared with the prior
healthy transaction:

1. Did M reset actually stop **both members** of old rail, valid and tree latches?
   Did each ignition relay's source/inhibitor get a sufficiently long gap?
2. Did master READY ignite COPY's u and v, and did each selected-stage-rail
   qualifier emit an ignition? Record its driver edge, veto and RR rate tap.
3. Did every intended master rail ignite exclusively, then its valid OR,
   ignition edge and valid latch? Which completion-tree join first stayed dark?
4. Did W_M's u/v fire with a fresh rise? If they did, compare their exact
   arrivals into `done.edge` and `done.edge_inh` with the preceding reset gap.
5. For copy 74, did bit 3 actually double-rail, survive reset or run fast on
   one rail? Establish whether that event precedes the first missing completion
   node; the fault detector itself has no control edge to DONE.

## Fix and regression disposition

**No fix recommendation is justified yet**, and no netlist-building code was
changed. A COPY/qualifier failure belongs in `lib/staged.py` and
`protocol/celement.py`; a failed master validity/tree re-arm belongs in
`protocol/celement.py` / `protocol/latch.py`; a captured fresh W_M with a lost
DONE belongs in `protocol/latch.py:add_edge_relay`. Those are diagnostic
ownership boundaries, not proposed changes. Neuron/synapse/latency costs cannot
be assigned without selecting an evidenced mechanism.

The existing controls already distinguish the proposed depths: true guards
passed this transaction's commit command; `copy_requires_rail` still requires
the selected Q rail and must not be relaxed on the strength of a valid-OR
indicator. Four × 0.75 kernel kills successfully clear these `creq[1]` latches;
the master's four × 0.75 reset is a different, unobserved domain. The control
machine retains three × 0.75 kill trains. The earlier stronger/longer-train
experiments' reload failures in [tick_stalls.md](tick_stalls.md) prevent treating
a global strength increase as an evidence-based remedy. RR v2 conditions and
clears upstream rate readers/qualifiers; it does not condition DONE or guarantee
storage ignition. Stalls in both builds therefore do not by themselves implicate
or exonerate rate conditioning.

The commands above are the exact campaign reproductions. After the recapture
names the first failed primitive, a bounded RefSim regression should construct
that corner directly: complete one staged transaction, reproduce the observed
reset/remaining-train/next-copy timing on the identified neuron and incoming
edges, and attempt the next commit. It should assert the observed old-build
failure, then exclusive master rails, a fresh master completion, exactly one
DONE, Q clear/READY and IDLE re-arm with the candidate correction. Keep a third
transaction to test recovery and duplicate suppression. Include both base and
RR v2, retain true guards and selected-rail COPY, and check kernel versus machine
kill/reload timing if the shared primitive changes. Dropping a random DONE pulse
would reproduce the symptom without establishing this campaign's mechanism.

## Validation

The read-only analysis rebuilt both netlists, checked all four role maps and
capture membership, inspected raw fault spikes and grouped latch trains, and
traced the nominal incoming edges quoted above. The reports and original data
were not modified. `test -s docs/stage_d_completion_stall.md` is the required
document check.

`TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest` was attempted and blocked
by the sandbox at `/Users/jeremiahgassensmith/.cache/uv/sdists-v9/.git`.
With a writable workspace `UV_CACHE_DIR`, the existing read-only project venv
and `UV_NO_SYNC=1`, a full-suite attempt passed 36 tests before interruption
(exit 130, 495.67 s); this is not a full-suite pass. Focused diagnostic checks
used the same requested TMPDIR and a writable `--basetemp`:

```text
uv run pytest tests/test_stall_diag.py -k 'stage_d_record_rebuilds or compact_dump_roundtrip or obsolete' --basetemp=<worker-tmp>/pytest-completion
8 passed, 11 deselected in 5.59s
```

The required nonempty-file check passed. A separate read-only assertion script
also checks the sustained stage/COMMIT/busy trains, cleared pending requests,
silent post-START DONE/READY/IDLE delay chains, master fault gates' absent control
outputs, and both recapture regexes' counts and coverage of all missing domains.

## Isolated reconstruction from static draws (2026-10-04)

**Mechanism established for base 74, base 17 and RR 24; RR 11 is not reproduced.**
In each of the three, one old **master rail latch survives the master's own reset**
(`add_reset`, four taps × 0.75, inside `add_register`). It is the copy-28 failure
class, now in the master. Each surviving latch is fast at that copy's draws. Strays
throw it into a coincident 34–35-step mode, and that copy's reset controller drops
its residual fifth inhibitor spike. The four arrivals then land just after both
members fire, so the latch is **entrained, not killed**.

The survivor keeps its master valid OR running. The reset kills the valid latch,
and its one-shot ignition relay has not recovered by the time the OR restarts. That
valid latch, the completion tree above it and W_M stay dark, and DONE never fires.

Copy 74's survivor is `b3r0`, on a bit that flips 0→1. COPY then writes `b3r1`
beside it, which reproduces the captured fault3 train. Copies 17 and 24 survive on
unchanged bits, so their master fault gates stay silent, as captured. The device
stray stream is not replayed. This shows the copies' static draws make the
mechanism reproducible. It does not prove it was the event at the captured step.

### Method

The Stage D netlists were rebuilt from the repository with the records' build options:
base 29,375 / 52,808, and RR v2 30,643 / 55,936. `make_perturbed_sim`'s host draws
were regenerated with `default_rng(108)` and B = 100, in its order: weights `(100, nnz)`
in topology edge order, then V_th, then bias. As an independent check, copy 28's
request latch comes out at exactly the values pinned in
`tests/test_request_clear_entrainment.py`: V_th −44.89665616330281 mV and `u→v` 4,050 q.

Each failing cell was rebuilt from the real primitives:
- `add_register` (11-bit stage),
- `kernel._fault_latch`,
- `add_staged_commit` with `ordered_grant=True` and `copy_requires_rail=True`,
- for the state cell `c4_sel`, the power-up veto latch.

That is 522 / 1,082 neurons/edges for `c4_sel`, 520 / 1,077 for `c5_sub` and
604 / 1,267 for the RR cells. Draws are mapped by (source role, target role,
delay). Every primitive edge has its kernel edge with identical nominal quanta,
and the kernel has no other edge among these neurons. Outside inputs reach only
the stage rails (ALU/Z output relays) and `commit_in`. **M, grant, COPY, its arms
and DONE have no input from outside this path.**

The words are the reference values (`kernel_outputs`) at the failing tick, located
from the output cells' commit counts. Bits 8/9/10 are C/Z/V.

| Copy / cell | Tick | Old master | New stage word | Bits that change |
|---|---:|---|---|---|
| base 17 `c4_sel` | 728 | 0 (Z) | 48 | 4, 5, 9 |
| base 74 `c5_sub` | 323 | 197 (C/Z/V=0) | 187 (C/Z/V=0) | 1, 2, 3, 4, 5, 6 |
| RR 11 `c10_xor` | 864 | 6 | 59 | 0, 2, 3, 4, 5 |
| RR 24 `c2_sel` | 654 | 0 (Z) | 48 | 4, 5, 9 |

Stimulus, in steps:
- step 10: ignite the old master word (its power-up DONE clears the empty stage);
- step 6,000 + U[0, 100] per rail: ignite the new stage word;
- step 10,300 + U[0, 599]: one ignition event into `commit_in`;
- run to step 17,400.

Strays follow the mix-B law: Bernoulli 5 Hz × 150 q on every neuron. Noise-free,
the path keeps the captured schedule. Commit → master READY is 1,593 steps for
copy 17 (captured 1,597). READY → DONE is 2,538 steps (prior healthy capture 2,501),
and 2,690 for copy 74 (captured 2,652).

### Failure rates of the isolated path

Each node replays its own `default_rng(seed)`, seeds 0–1,599: commit phase, load
offsets and strays. A stall means no DONE in the ≥ 6,500 steps after commit;
a healthy commit → DONE takes about 4,100–4,300 steps.

| Copy | Own draws: stalls | Old-rail survivors (stalled / completed) | Nominal draws | Compact master reset, same seeds |
|---|---:|---|---:|---:|
| base 74 | **24 / 1,600** | 50 × `b3r0` (24 / 26) | 0 / 1,600 | **0 / 1,600** (no survivors) |
| base 17 | **4 / 1,600** | 4 × `b8r0` (4 / 0) | 0 / 1,600 | 0 / 400 * |
| RR 24 | **1 / 1,600** | 1 × `b6r0` (1 / 0) | 0 / 1,600 | 0 / 400 * |
| RR 11 | 0 / 1,600 | none | 0 / 800 | — |

\* A separate 400-realization set in which the shipped reset gave 6 (74), 1 (17)
and 1 (24) stalls.

Every stall shows the same pattern:
- exactly one old rail is still firing at master READY;
- that bit's valid latch is dark, along with the tree branch above it and W_M;
- every selected copy arm fires and no opposite arm fires;
- master fault activity appears only on copy 74's bit 3.

In none of the realizations, at any setting, did COPY fail, an arm fail, a valid or
tree latch survive, or master READY go missing.

The 26 non-stalling copy-74 survivals commit with bit 3 **double-railed**: DONE
fires while `b3r0` and `b3r1` both hold. The master fault gate has no control
output, and the downstream effect is not modelled here.

### The first break: reset survival

Noise-free, the master reset inhibitor arrives at these offsets (steps after the
first arrival):

| Draws | Arrivals |
|---|---|
| nominal | 0/45/89/136/215 |
| copy 17 | 0/49/96/140 |
| copy 74 | 0/48/94/141 |
| copy 24 | 0/49/98/147 |
| copy 11 | 0/47/90/138/241 |

As with copy 28's request clear, the tap count is not the arrival count.

The three surviving latches. Nominal loop weight is 3,621 q, V_th −45 mV.

| Latch | `u→v` | `v→u` | V_th u / v (mV) | Fast-mode share of strayed u intervals (≤ 38 steps) |
|---|---:|---:|---|---:|
| 74 `b3r0` | 3,833 (+5.9 %) | 4,083 (+12.8 %) | −45.46 / −44.78 | 16.8 % |
| 17 `b8r0` | 4,105 (+13.4 %) | 3,811 (+5.2 %) | −44.70 / −45.38 | 1.7 % |
| 24 `b6r0` | 3,644 (+0.6 %) | 4,153 (+14.7 %) | −45.35 / −44.77 | 42.9 % (coincident even noise-free) |
| nominal | 3,621 | 3,621 | −45 / −45 | 0 % |

In the survivor traces both members fire 2–6 steps apart at a 34–37-step period
before the reset. Each arrival comes within 0–14 steps of a member spike, mostly
just after both have fired, while they are refractory (22 steps). It therefore
delays the pair instead of stopping it: copy 28's entrainment.

**Reduced primitive.** The model is one `add_latch` plus the real `add_reset`, with
role-mapped draws on all seven neurons. Strays hit every neuron, the reset comes at a
random step in 3,000–4,999, and survival means the latch is still firing 150 ms after
the reset. Results:
- 74 `b3r0`: **221 / 40,000** (another 46 / 8,000).
- 17 `b8r0`: **9 / 40,000** (another 29 / 40,000).
- 24 `b6r0`: **42 / 40,000** (another 5 / 8,000).
- Every other old rail of these words: 0 / 4,000.

Swapping one parameter group to nominal (8,000 resets each):

| Variant | 74 `b3r0` | 24 `b6r0` |
|---|---:|---:|
| copy draws | 46 | 5 |
| nominal reset controller (five spikes) | 3 | 0 |
| nominal fan-out weights | 30 | 23 |
| nominal loop weights | 0 | 0 |
| nominal latch V_th / bias | 0 | 0 |
| all nominal | 0 | 0 |
| compact master reset, copy draws | **0** | **0** |

**The fragile parameters are the latch's loop weights together with its
thresholds, which create the coincident fast mode. The controller draws that
suppress the fifth inhibitor spike are also needed.** Neither alone suffices at
these copies. Fan-out draws matter little.

### Why a survivor blocks completion

The master valid latch is re-ignited through `celement._ignite_from`. That edge
relay fires once per OR activation, and each OR spike's edge inhibitor delivers
2.2 × loop to it. The reset inhibits the OR gate, but a surviving rail restarts it.

At copy 74's draws, with the old rail kept alive, the OR falls silent from +19 to
+574 steps after the trigger, a 555-step gap, and the relay never fires again. At
nominal draws (five-spike reset) the gap runs from −61 to +674, 735 steps. The relay
fires at +730, and the word completes.

To test survivors directly, the reset's fan-out onto a single old rail was removed.
The resulting forced survivor blocks completion on:
- 4 / 11 bits for copy 17,
- 6 / 11 for copy 74,
- 5 / 11 for copy 11,
- 10 / 11 for copy 24,
- 0 / 11 at nominal draws (base words, and RR copy 11's word).

The race depends on the draws. Base copies 17 and 74 block on no bit if their
reset controller alone is made nominal. They also block on none if everything except
their four-spike controller is made nominal. RR copy 11 blocks with five spikes. RR
nominal draws with copy 24's four-spike controller block all 11 bits.

Under strays the outcome varies: 26 of copy 74's 50 survivals re-lit valid3.
Forced `b3r0` survival at copy 74's draws fires fault3 first at **master READY + 526**,
then every **380–385** steps. The capture shows +528, then 382/383/385 steps.

In copy 11's forced screen, surviving valid latches and internal tree latches never
block. A surviving root would keep DONE's inhibitor firing, which the captures exclude.
No valid- or tree-latch survivor appeared in any strayed realization.

### RR copy 11: not established

The forced-survivor screen shows which bits of copy 11's word could block. Flipping-bit
survivors (`b2r1`, `b5r0`) would fire their conditioned RR fault gates, every 175–237
steps, but the capture's master fault gates are silent. That leaves the unchanged bits
`b7r0`, `b8r0` and `b9r0`. Each survives **0 / 40,000** reduced resets; the 95 % upper
bound is 7.5 × 10⁻⁵ per reset.

The whole path stalls in 0 / 1,600 realizations. Copy 11's controller keeps its
fifth spike, even though `b7r0` spends 10.7 % of its strayed intervals in the fast
mode. Either the same mechanism acts at a rate below these samples, or something
outside the isolated path is responsible. The Juno recapture command above is
still the way to settle copy 11.

### Fix recommendation

**Depth:** the staged master's reset train. Its kill margin is the first break.
Do not "repair" the blocking half: re-arming the valid relay under a surviving
old rail would convert these fail-stops into completions with a stale or
double-railed master, which 26 / 1,600 copy-74 realizations already show.

**Candidate:** an opt-in compact master reset that uses the
`control.add_request_clear` form inside `add_register`, for staged masters only:
- the trigger → relay1 → relay2 → relay3 links go from 18 to **0 steps** of delay;
- inhibition onto every master **latch member** rises from 0.75 to **1.1 × loop**;
- inhibition onto gates stays at 0.75.

Isolated results:
- full path: 0 / 1,600 stalls and 0 survivors for copy 74, on the seeds that gave 24 stalls;
- reduced resets: 0 / 40,000 for each fragile latch (copy draws, compact reset);
- inhibitor arrivals stay at four: nominal 0/43/81/138, copy 74 0/45/84/144, copy 24 0/48/89/149.

**Reload:** COPY's rail ignitions arrive **976–992 steps after the trigger spike** at
nominal and copy draws. All-phase reload bounds are on a 25-step grid with a nominal
controller. They are measured from the trigger's input event, which precedes its
spike, so the margins below are conservative:

| Latch corner | Shipped reset | Compact reset |
|---|---:|---:|
| nominal | 675 | 650 |
| loop −8 % / V_th +0.4 mV | 800 | 800 |
| loop −12 % / V_th +0.6 mV | 925 | 900 |

At the slow corner that leaves margins of **≥ +51** (shipped) and **≥ +76**
(compact) steps. Noise-free master READY → DONE moves by at most ±45 steps.

**Costs:**
- **+0 neurons and +0 synapses.**
- Per `tick2.c` build: 14 staged masters, 42 tap links set to 0 delay, and 1,214
  latch-member fan-out edges strengthened from −2,716 to about −3,983 q.
- The 288 gate edges are unchanged.
- RR has 576 mirrored readout edges, and `mirror_inhibition` would scale them with
  the fan-out (16 × gain). The evaluated candidate left them at shipped strength, so
  that part is untested.
- No added hops: master READY is the fixed 15-hop chain from the trigger.
- Pinned netlist hashes move, and campaigns must be compared by per-seed totals.

**Existing opt-ins:**
- `zero_once` changes only Z0 ignition.
- `robust_request_clear` changes only DONE-side request k1 trains; the master reset is untouched.
- `rate_robust` does not cover it: copies 11 and 24 are RR builds, and RR leaves latches and resets unchanged.

Only a compact or stronger master clear addresses the break. A plain ordinary
4 × 1.0 master train was not evaluated here; the request analysis found that form
loses slow-corner reload.

**Qualification still needed:**
- the noisy kernel campaign on Juno (seeds 108–110, both builds);
- stage and other register resets, which are untouched;
- `test_machine.py`, `test_mul_diag.py` and the build-option hash tests.

### Regression test

`tests/test_completion_stall.py` rebuilds both netlists in about a second; no external
data is needed. The fast tests take about 17 s:
- **Path match:** for all four copies, the role-mapped primitive path exactly matches
  the kernel's subgraph and its only outside inputs.
- **Inhibitor spike counts:** four for copies 17, 74 and 24; five for copy 11 and nominal.
- **Noise-free commit:** copy 74's path commits on the captured schedule.
- **Forced survivor:** at copy 74's draws, a forced `b3r0` survivor blocks DONE, leaves
  valid3 dark and fires fault3 with the captured timing. At nominal draws the same
  survivor completes.
- **Replay:** seed 23 stalls through `b3r0` at copy 74's draws; the identical stimulus
  and strays at nominal draws commit.
- **Reduced reset:** 2,000 strayed resets give 8 survivors at copy 74's draws, 0 at
  nominal. Compact candidate surveys are in `test_robust_reset.py`.

A slow test (`-m slow`) repeats 400 full-path realizations for copies 74 and 17.
At their own draws, every stall must have exactly one surviving old rail and that
bit's valid latch dark. Nominal draws must give no stalls.

### What the isolated model cannot establish

- **Not a replay of the stall itself.** The CUDA stray stream, the captured reset
  phase and the master trains at the stall are not replayed; the master rails are
  unrecorded. The only direct link to a capture is copy 74's fault3 onset and period.
- **Rates.** Rates are for the failing word pair only; other words light other rails.
  They are not calibrated against the campaign's per-transaction frequency. Suppose
  each copy's full-path rate applied to every earlier transaction. The chance of
  reaching the failing tick without a stall would then be about 0.8 % (copy 74,
  tick 323), 16 % (copy 17, tick 728) and 66 % (copy 24, tick 654). Copy 74's
  average rate over other words must be lower; that is not measured.
- **Stage word.** The stage word is ignited ideally. Arm qualification never failed
  here, but the ALU relays' real timing is not modelled.
- **Copy 11.** Its stall remains unexplained.
- **The compact master reset.** It is evaluated only on this path and a single-latch
  reload probe. Kernel-wide noise, the RR readout clears at scale, and its interaction
  with the stage reset and readers are unqualified.
- **Double-railed commits.** Their downstream consequence is not modelled.

Commands run:

```text
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_completion_stall.py
```

## Register reset qualification and opt-in (2026-10-04)

**Review correction: the opt-in is withdrawn.** `robust_register_reset=True`
now raises a descriptive `ValueError`, including through campaigns, diagnostic
rebuilds and Stage D recheck. **No tested setting satisfies the requested full
mix-B envelope.** A longer READY chain fixes the isolated rail recovery race,
but does not qualify the entire extended reset domain. At commit `b62de0e` the
experimental final pass was available only to tests. The closed-loop evaluation
below subsequently exposes it under an explicitly experimental campaign flag;
the robust option remains withdrawn.

The option off preserves the ordered topology, weights, delays, biases and
machine fingerprints. The preparatory whitelist is narrowed to exercised 2/8-bit
ADD/SUB/AND/XOR/SEL kernels with cell/constant operands. MOV, OR, ROM LOAD,
parameter operands, widths 1/4, RAM, multipliers, pacing and changed timing are
rejected; even a whitelisted configuration cannot enable the withdrawn option.

### Why the previous qualification was insufficient

The old policy used four taps, zero-delay inter-tap links, 1.75 × loop
inhibition and four 100-step READY links (+328 steps / 32.8 ms).
Its advertised minimum recovery margin of 20 steps was an isolated, noise-free
latch measurement with **zero tonic bias**, not an actual master COPY margin.
The independent review added −0.2 mV storage bias and measured recovery at 975
steps versus READY at 945: **−30 steps**, with **353/1,000** same-rail noisy
reload failures. The committed full staged-cell path also lost its second DONE
in base and RR. Those observations supersede the earlier recovery claims.

The revised tests include log-normal weight sigma 0.04, threshold sigma 0.2 mV,
tonic-bias sigma 0.2 mV and Bernoulli 5-Hz × 150-q strays. Explicit combined
three-sigma corners use `exp(±3 × .04)` weights and ±0.6 mV thresholds **and**
biases. These are finite conditional experiments, not a guarantee over the
unbounded tails of a Gaussian distribution.

### Kill evidence and actual reload frontier

The four captured primitives retain role-mapped seed-108, B=100 draws from the
original topology: master 74 `c5_sub.M.b3r0`, master 17 `c4_sel.M.b8r0`,
master 24 `c2_sel.M.b6r0`, and stage 18 `c3_xor.Q.b9r0`.
Four close initial ignitions, a later 300-q kick, random reset phases and strays
challenge coincident modes. A survivor fires 1,500 steps after reset. Each
configuration has 40,000 trials; the reset controller is simulated, not replaced
by a prescribed list of inhibitory arrivals.

The rejected 1.5 strength's **59/40,000** fast-storage/weak-clear/slow-controller
survivors now have an explicit slow regression,
`test_rejected_1_5_strength_has_59_survivors`. The 1.75 policy retains zero
survivors at the captured draws and tested corners, including +0.6 mV fast
storage against −0.6 mV slow controller bias. All **eighteen 40,000-trial
settings passed: 0/720,000 survivors**. Delaying READY does not alter that kill
experiment.

| Primitive/draw settings | Trials per setting | Survivors per setting |
|---|---:|---:|
| Master 74, master 17, master 24, stage Z0 18 | 40,000 | 0 |
| Nominal, slow8, slow12, fast20, fast30, asymmetric, mirror | 40,000 | 0 |
| Combined slow/fast three-sigma, biased fast30 | 40,000 | 0 |
| Fast30 with independent fast/slow controller | 40,000 | 0 |
| Biased fast three-sigma/fast30 with biased slow controller | 40,000 | 0 |

For recovery, eight final READY links at 100 steps add **656 steps / 65.6 ms**.
No neurons or logical edges are added. In the slow-storage/fast-READY corner,
the storage loop and ignition are `exp(-.12)`, reset edges `exp(.12)`,
storage threshold +0.6 mV, storage bias −0.6 mV; READY drive is
`exp(.12)`, threshold −0.6 mV, bias +0.6 mV.

| 1.75 strength, delayed READY links | Added READY delay | Isolated recovery margin |
|---|---:|---:|
| 4 (withdrawn policy) | 32.8 ms | −17.8 ms |
| 6 | 49.2 ms | −1.4 ms |
| 8 (experimental candidate) | 65.6 ms | +15.0 ms |

These margins are **asserted**, including the negative results. They use a
conservative 25-step recovery grid; they are not mislabeled COPY-arrival margins.

The actual-path test builds a real one-bit stage/master cell with FAULT,
COMMIT, grant, selected-rail COPY arms, validity and DONE. It commits twice,
loading the **same stage rail one step after the observed Q READY**. It observes
every simulator step, so host polling cannot add hidden recovery time. Both Q
READY and M READY are also advanced by 100 steps: successful reloads therefore
demonstrate at least **10 ms of margin on actual stage loading and the real
READY → COPY → master path**. No latch ignition is substituted for COPY.
Ten thousand trials per build cover independently sampled mix-B static draws;
another ten thousand per build hold the adverse three-sigma rail/READY corner
and sample strays on every cell neuron. The full eleven-bit staged-cell
regression separately repeats the −0.2/−0.6 mV bias failure and asserts two
correctly decoded words with a 100-step advance of the second Q load.

The eight-link trials finished with **0 failures, 0 faults, 0 survivors and no
missing READY in all four 10,000-trial settings** (40,000 trials total). These
one-bit trials isolate the reset contract. They **do not qualify the ALU,
completion trees or whole extended domain**.

| Compact strength / READY links | Hard kill corner survivors | Adverse actual-path failures, base | Adverse actual-path failures, RR |
|---|---:|---:|---:|
| 1.5 / any | 59/40,000 | not selected after kill failure | not selected after kill failure |
| 1.75 / 4 | 0/40,000 | 1,000/1,000 | 998/1,000 |
| 1.75 / 6 | 0/40,000 | 238/1,000 | 227/1,000 |
| 1.75 / 8, READY advanced by 100 steps | 0/40,000 | 0/10,000 | 0/10,000 |

The kill column is the same fast30/slow-controller experiment, independent of
READY delay. Four/six-link rows load at actual READY+1 with no advance; the
eight-link row imposes the additional 10-ms margin. The rejected path counts
are pinned in slow tests. The additional eight-link random-static mix-B rows
also gave 0/10,000 failures in each build.

### The full-domain limit: no qualified setting

`_extended_domain_trial` runs two actual 2-bit ADD transactions, using the same
word twice so previously active state must rearm. The fully wired cell Q/M reset
targets include rails, valid/completion latches, ALU state/gates, FAULT, ACT,
COMMIT, grant/COPY, zero_once and rate-robust mirrors. All positive inputs to
those targets are multiplied by `exp(-.12)`, negative inputs by `exp(.12)`,
thresholds increase 0.6 mV and tonic biases decrease 0.6 mV **on top of compiled
biases**. READY independently gets the opposite, fast corner.

| Reset policy | Base outputs / 2 | RR outputs / 2 |
|---|---:|---:|
| Option off, ordinary reset | 0 | 0 |
| 1.5, compact taps, four delayed READY links | 0 | 0 |
| 1.75, compact taps, four delayed links | 0 | 0 |
| 1.75, compact taps, eight delayed links | 0 | 0 |
| 2.0, compact taps, eight delayed links | 0 | 0 |
| 1.75, compact taps, sixteen delayed links | 0 | 0 |

**Base fails before Q completion and before either cell reset fires.** Changing
cell reset strength, tap timing or READY delay cannot repair an event that
precedes those signals. RR reaches Q completion and resets M, but never
completes M, even with all sixteen READY links delayed. Neither build clears Q
after a successful commit. The tests assert these facts, not merely the absence
of final outputs. This is an existing analogue/datapath limit, **not evidence
that the eight-link policy has the old after-hyperpolarisation race**.

Perturbing ALU state, ACT/COMMIT/grant/COPY, or RR mirrors as separate groups
does allow both transactions. Separate adverse-FAULT tests discard two successive
double-rail words, rearm FAULT, then accept a clean word, advancing each new
load by 100 steps before Q READY. Those useful
partial passes cannot replace the failing combined-domain experiment.
No noisy 10,000-trial full-domain success is claimed: the deterministic corner
already falsifies full qualification. Resolving that limit requires more than
the reset timing/strength choices under review.

Adding mix-B strays to this full-domain corner (seeds 0–3) also failed all four
two-transaction trials in each build. Base produced no outputs in any trial;
RR produced one output at seed 1 and none at seeds 0, 2 or 3. The slow regression
asserts failure to complete both transactions. This is eight observed failures,
not an estimated full-domain failure rate or a 10,000-trial success claim.

### Cost, compatibility and reproducibility

For the **previous four-link policy**, reviewers measured nominal tick2 steady
tick intervals increasing **149–174 ms (+2.7–3.1%)**, and the three-tick probe's
first completed tick increasing **311 ms**. A cell's Q READY is unread, so its
extra delay is inert in the kernel. M READY gates COPY, and input-stage READY
gates upstream loading; only consumed delays contribute to the critical path.
The old statement that every cell pays two READY delays was incorrect.

The experimental eight-link pass changes 84 tap delays and 224 READY delays in
the tick build, still with 29,375 neurons / 52,808 edges (RR: 30,643 / 55,936).
The same 6,322 base / 7,776 RR inhibitory edges are rescaled; absolute quanta rise
by 22,891,962 / 92,813,472. Its per-tick cost has **not** been measured and cannot
be inferred by doubling the four-link measurements. It is not a released policy.

The compiler pass is idempotent, including overlapping domains, and rescales
from retained original integer weights with one rounding. It validates the
tap/READY chain shape before mutation and raises `ValueError` for short chains.
The 100-step physical delay is a named timing choice, checked against the
simulator limit rather than derived from it. The dead `tap_delay_steps` argument
and duplicate 1.1 candidate implementation are removed.

Stage D recheck uses nested `build_options` as authoritative, falling back to
top-level fields only when absent. A missing top-level `rate_robust` retains a
nested True; a record missing both fields rebuilds False. Small synthetic legacy
records cover older guard/timing/drive policies and authoritative nested fields.
Records requesting the withdrawn reset are rejected explicitly, including
nested-only records; they cannot silently rebuild a different circuit.

Copy 74's committed words are **197 → 187, C/Z/V all zero**. The earlier “C”
annotation has been corrected. Historical isolated rates refer to these words,
not to a newly reconstructed carry-set word or a replay of the CUDA strays.

The requested uv command is blocked by the sandbox's default-cache denial:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_robust_reset.py tests/test_completion_stall.py tests/test_stage_d.py tests/test_build_options.py -m 'not slow'
```

Validation uses the existing repository virtualenv, this worktree on
`PYTHONPATH=.:tests`, `PYTHONDONTWRITEBYTECODE=1`, and a writable task-local
`TMPDIR`/pytest basetemp: **all 216 requested non-slow tests passed**, including
the thirteen pinned default/machine topology fingerprints. All **32 slow tests
in `test_robust_reset.py`** passed in separate selections: eighteen survivor
settings, the 1.5 rejection, four 10,000-trial reload settings, four rejected
READY settings, the candidate table, two full-domain frontiers and two noisy
full-domain failures. The older 1,600-seed completion surveys were not rerun.
`git diff --check` passes. Integration and committing belong to the orchestrator;
no commit is attempted. A long noisy Stage D campaign and RR11's original
completion stall remain outside these conditional experiments.

## Closed-loop verification and experimental campaign fallback (2026-10-04)

**Choice: fallback B.** The tested closed-loop prototype does not meet the
complete contract. `verified_register_reset=False` is recorded by every kernel
build; requesting True raises a descriptive qualification error, including
from either CLI, diagnostic rebuilds and Stage D `--recheck`. It must not
silently select a timed circuit. `robust_register_reset=True` still raises.
The 1.75-loop/eight-READY-link candidate can now be measured in campaigns with
`experimental_register_reset=True` / `--experimental-register-reset`. This is
an **unqualified experiment with known full-domain failures**, not a robustness
claim. The flags are mutually exclusive.

### Neural verification design tested

`latch.add_reset_verification` and `handshake.verify_register_resets` implement
the prototype for direct qualification experiments. No simulator, integration
rule, host-generated READY or prescribed reset-spike list is involved.

1. Finalize after all wiring. Monitor **every target of the register inhibitor's
   negative fan-out**, including both members of rails, valid/tree latches,
   ALU state and gates, FAULT, ACT, COMMIT, grant/COPY, power-up vetoes,
   `zero_once`, and compiled `rate_robust` reader mirrors. The original ordinary
   four-tap, 0.75-loop train and all its integer fan-out weights remain intact.
2. All monitored targets excite one refractory-limited BUSY neuron. BUSY
   inhibits each of four timer neurons at 0.75-loop strength. The existing
   fifteen-link recovery prefix feeds these four **100-step** timer links;
   only a propagated certificate can drive the existing READY output.
   The measured certificate interval is about 52–57 ms, comfortably longer
   than the measured storage periods (up to 6 ms in the deterministic screen).
   The busy aggregate limits veto firing rate as register width grows; it does
   not eliminate residual inhibitory charge or analogue timing assumptions.
3. A parallel seven-link deadline is vetoed by the certificate. If no
   certificate cancels it, it starts another ordinary four-tap train through
   the **same reset inhibitor**. This preserves mirrored gains exactly.
   Three separate retry stages are statically unrolled: there is no feedback
   edge that can start a fifth attempt.
4. The final uncancelled deadline ignites a sticky exhaustion latch. It holds
   all four attempt entries and READY down and drives the register's existing
   FAULT latch where present. Masters without such a latch expose the
   `<register>.verify.exhausted` group as explicit terminal stall evidence.
   Exhaustion is outside the reset domain and requires a new image/simulator
   to clear. A forced surviving rail gets no READY, exactly four attempts,
   and exhaustion; a later trigger pulse cannot restart it.

This is a **candidate silence certificate**, not a proof for arbitrary spike
schedules. Sustained survivors veto it in the tested draws. A finite
silence interval cannot prove membrane recovery, exclude all late external
ignitions, or fix a datapath that cannot complete before any reset. Neither
exhaustion nor an unobserved READY is counted as a successfully emptied domain.

### Primitive evidence, latency and cost

The four captured primitives retain the original role-mapped seed-108/B=100
weights, thresholds and biases. New detector/controller edges and neurons have
independent log-normal sigma .04 weights and sigma .2 mV thresholds/biases.
Every neuron receives Bernoulli 5-Hz, 150-q strays. Four initial ignition pulses,
the later 300-q kick and random reset phases challenge the entrained modes.

The exploratory noisy screen ran **256 trials per captured primitive**. All
1,024 reached READY with zero observed survivors, false READY, exhaustion or
undecided outcomes. Latency is measured from the original trigger's spike to
READY, in ms. “Retry” includes every trial needing at least one extra train;
two master-74 trials needed two retries.

| Draw | Trials needing retry | Common latency min / median / p95 / max | Retry latency min / median / p95 / max |
|---|---:|---|---|
| Master 74 | 41 / 256 | 140.2 / 143.0 / 145.3 / 146.7 | 311.2 / 315.0 / 321.5 / 492.7 |
| Master 17 | 195 / 256 | 138.8 / 141.5 / 143.3 / 144.4 | 306.5 / 311.3 / 314.83 / 317.5 |
| Master 24, RR capture | 133 / 256 | 140.6 / 143.6 / 145.29 / 146.9 | 309.8 / 314.2 / 317.8 / 320.0 |
| Stage Z0 18 | 43 / 256 | 140.8 / 143.6 / 146.4 / 147.6 | 311.4 / 314.6 / 319.23 / 321.2 |

Minimum certificate intervals in these four settings were 52.6, 52.3, 51.9,
and 52.4 ms. The nominal one-rail, non-entrained common case is 142.7 ms from
trigger to READY, versus the ordinary chain's 84.8 ms: **+57.9 ms**. No stronger
inhibition is applied to storage in that common case.

The deterministic screen covers 32 phases in each captured draw and all ten
previous corner settings. There are no false READY or undecided outcomes.
However, the additional fast20, fast30 and biased-fast30 stress settings each
leave **32/32 live terminal survivors after all four trains**, with READY
withheld and exhaustion at **716.1 ms**. These extra-fast settings extend beyond
the combined three-sigma storage corner; they are not relabelled as its failure
rate. A separate biased-fast30/biased-slow-controller screen also exhausts
**64/64 noisy trials**. Bounded retries provide a fail-stop, not stronger kill
margin. At the combined fast-three-sigma storage/fast-biased-detector setting,
64 noisy trials yielded 64 safe READYs, 53 needing a retry.

The prototype adds **104 neurons and D + 152 + F synapses per register**, where
D is the number of distinct final reset targets and F is 1 if an existing
FAULT latch is driven, otherwise 0. It reuses the original recovery chain and
inhibitor; it rewires the old final READY edge. In the measured two-bit
ADD/XOR kernel with `zero_once` and `robust_request_clear`:

| Register | Base D / added synapses | RR D / added synapses |
|---|---:|---:|
| Input Q | 27 / 180 | 37 / 190 |
| Input M | 17 / 169 | 23 / 175 |
| Each cell Q | 157 / 310 | 185 / 338 |
| Each cell M | 47 / 199 | 65 / 217 |

### Actual reload paths and the release blockers

The actual one-bit staged-cell harness loads the same Q rail at **observed
Q READY + 1 step** and copies through the real M READY → COPY latch → selected
arm → M rail → completion path. At the adverse slow-three-sigma rail/fast-READY
corner, ten noisy trials per base/RR build completed both transactions with
**0 failures, faults, survivors or missing READY**. COPY's minimum arrival
after M READY was 151 steps (base), 142 (RR). These are small screens, not a
10,000-trial qualification. The verified harness observes 32,000 steps to
include retries: the timed harness's 18,000-step limit initially mislabeled a
late second DONE as a failure; extending observation showed it completing.

Two stronger counterexamples prevent release:

* **The combined full-domain corner still fails.** The real two-bit ADD path
  applies adverse three-sigma weights, thresholds and biases to the entire
  extended domain, with READY independently fast. Base produces no Q completion
  and neither Q nor M reset fires; RR completes Q and resets M but produces no
  output. Both requested transactions fail to complete. The new detector has
  no causal input before a reset, so it cannot repair the base failure. The
  old timed candidate and the option-off baseline fail the same experiment.
* **A replacement READY is insufficient for kernel sequencing.** Cell Q READY
  is not consumed by the existing kernel. IDLE is re-lit at a fixed delay from
  DONE, and the next operation starts while verification is still running.
  In the nominal two-cell ADD/XOR run, a Q retry clears newly active state. Of
  the expected outputs `[0, 3, 1]`, only the first, correct `0` is produced,
  in base and RR, with both auxiliary opt-ins off and with `zero_once` plus
  `robust_request_clear` on. Tests assert that the second START precedes Q READY
  and a retry follows that START. An eventual verified policy needs a stage
  READY/IDLE interlock as well as a separately repaired full-domain datapath.

The new slow tests retain 40,000-trial primitive safety surveys (reporting
exhaustion and terminal survivors separately) and 10,000-trial actual noisy
reload qualification gates. **They were not run for the rejected prototype;
no 40,000-reset or 10,000-reload success is claimed for verified reset.** The
finite counterexamples already disqualify it. The earlier 0/720,000 kill and
four 0/10,000 reload results in this document belong exclusively to the timed
1.75/eight-link candidate and its stated, narrower experiments.

### Campaign exposure and compatibility

The experimental flag applies the existing idempotent final pass after all
ALU/control/mirror extensions, to input and cell Q/M registers only. It adds
**zero neurons and zero synapses per register**; three tap links become zero
delay and eight READY links become 100 steps. Ordinary READY's nominal
84.8 ms becomes 150.4 ms, a **65.6-ms** increase. It has no conditional retry
case. Whole-tick overhead and its real Stage D failure rate remain unmeasured.
The prior whitelist is retained: standard physics/drive/timing, true guards,
selected-rail COPY, 2/8-bit single-stream ADD/SUB/AND/XOR/SEL with cell/constant
operands, no memories/parameters/pacing or `retry_clear`. Existing `zero_once`,
`robust_request_clear`, and `rate_robust` options remain independent.

Both new booleans are recorded in `build_options`, Stage D's top-level record
and kernel_campaign records. Diagnostic rebuild and recheck preserve the
recorded option; authoritative nested Stage D options override top-level
fallbacks. Older records default False. Tests run the experimental multi-cell
kernel for every combination of the three existing opt-ins, with generic and
specialized datapaths, and require `[0, 3, 1]`. Ordered default netlist hashes
and machine hashes remain pinned to their pre-change values.

Example campaign (not run here; output is explicitly experimental):

```sh
uv run python -m drosophilos.bench.stage_d --ticks 1000 --copies 100 \
  --mix B --seed 108 --backend torch-fast --device cuda \
  --experimental-register-reset --out stage_d_experimental_reset_108.json
uv run python -m drosophilos.bench.stage_d \
  --recheck stage_d_experimental_reset_108.json
```

The exact requested command was attempted:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_verified_reset.py tests/test_robust_reset.py tests/test_build_options.py -m 'not slow'
```

`uv` is blocked by sandbox denial opening
`/Users/jeremiahgassensmith/.cache/uv/sdists-v9/.git`. Validation instead uses
the existing repository virtualenv, `PYTHONPATH=.:tests`,
`PYTHONDONTWRITEBYTECODE=1`, and task-local TMPDIR/pytest basetemp/cache.
All **209 requested non-slow tests pass** in separate file runs: 40 in
`test_verified_reset.py`, 86 in `test_robust_reset.py`, and 83 in
`test_build_options.py`, including the thirteen ordered default/machine hashes.
The four additional 256-trial noisy capture screens completed as reported above.
`git diff --check` passes.
No simulator or machine source was changed. Integration and committing remain
with the orchestrator; no commit is attempted.

## Capture verification (2026-10-05)

**The Juno 441837 recapture confirms the master-survivor mechanism for base
copies 17 and 74, including the predicted bits, failed valid re-ignition and
successful selected-rail COPY. It refines the reset phase account: the arrivals
do not all land just after both latch members fire.** Copy 74 is directly
double-railed on bit 3; its fault train is no longer merely indirect evidence.
This section supersedes the earlier evidence limits for these two base copies.
RR copies 11 and 24 were not included in this recapture.

### Capture and rebuild

Inputs are under
`/Users/jeremiahgassensmith/programming/drosophilos/data/stage_d/`:
`mixB_s108_c100_completion740.json`,
`mixB_s108_c100_completion740_copy17.npz` and
`mixB_s108_c100_completion740_copy74.npz`. The accompanying diagnostic reports
are `reports/base_copy17_completion.md` and `reports/base_copy74_completion.md`.
The failed commit commands are unchanged: **43,408,414**, neuron **28143**
(`c4_sel`), and **19,107,156**, neuron **28341** (`c5_sub`).

`stall_diag.build_tick_pipeline(record)` rebuilt the authoritative nested
`build_options`: generic datapath, true guards v2, selected-rail COPY,
rate-robust off and shipped reset strength 0.75. The rebuilt netlist has
**29,375 neurons / 52,808 synapses**. `load_dump` validated the recorded roles
with **zero ID drift** in both captures. Each dump explicitly captures **6,366
IDs**, including the target cell's complete Q/M domains, guard, grant, COPY and
all arms; silence of the probes below is therefore observable. Copy 17 contains
46,301,284 spikes in **43,114,738–45,271,503**; copy 74 contains 49,165,920 in
**18,799,802–20,956,617**, inclusive.

`datapath_view` from the failed STARTs (**43,403,114 / 19,101,769**) shows all
eleven Q valids rising and exclusive Q rails holding the new words **48 / 187**,
with C/Z/V all zero. Direct master-rail inspection before reset gives the old
words **0 with Z=1 / 197 with C/Z/V=0**, agreeing with the isolated model's
reference words. Further event extraction used the rebuilt cell's neuron IDs
and nominal synaptic delays, rather than replaying static draws or CUDA strays.
Steps remain 0.1 ms. The ≤141-step train criterion is used for latch members;
the slower OR and fault gates are inspected as individual spikes.

### Exactly one old master rail survives in each copy

| Copy | Master reset trigger | Survivor u/v | Spikes bracketing master READY | Last survivor u/v spikes |
|---|---|---|---|---|
| 17 | `M.reset` **3500**, **43,409,210** | `M.b8r0` **3351/3352** | u: 43,410,010 → 43,410,054; v: 43,409,985 → 43,410,029; READY **3521** at **43,410,011** | **45,271,480 / 45,271,499** |
| 74 | `M.reset` **4077**, **19,107,932** | `M.b3r0` **3908/3909** | u: 19,108,743 → 19,108,786; v: 19,108,724 → 19,108,767; READY **4098** at **19,108,764** | **20,956,595 / 20,956,576** |

Both members of each survivor fire continuously across reset and through capture
end. Their largest post-trigger gaps are **80/83 steps** (17 u/v) and **85/83**
(74 u/v), including the reset disturbance. Every other old master rail stops
before READY. Unchanged rails then restart only after their selected COPY arm,
with **857–965-step** gaps in copy 17 and **885–998-step** gaps in copy 74;
flipped old rails remain silent. No master valid latch or completion-tree latch
survives the reset. Thus this is old data storage surviving, not a retained
completion root or a COPY failure.

### Actual inhibitor arrivals and entrainment

The captured reset inhibitors emit **four spikes, with no fifth**:

| Copy / inhibitor | Emission steps | Arrival steps at both survivor members (+18) | Arrival offsets from first |
|---|---|---|---|
| 17 / **3501** | 43,409,267; 43,409,317; 43,409,366; 43,409,413 | **43,409,285; 43,409,335; 43,409,384; 43,409,431** | **0/50/99/146** |
| 74 / **4078** | 19,107,985; 19,108,033; 19,108,079; 19,108,126 | **19,108,003; 19,108,051; 19,108,097; 19,108,144** | **0/48/94/141** |

The rebuilt inhibitor fan-out has −2,716-q edges to both members, all with
18-step delays. These are nominal quanta; the capture does not record each
copy's perturbed weights. The following are the last member spikes at or before
each arrival; parenthesized numbers are their ages in steps. An age of zero
means the same recorded step, without a claim about within-step event order.

| Copy | Inhibitor arrival | Survivor u spike (age) | Survivor v spike (age) |
|---|---:|---|---|
| 17 | 43,409,285 | 43,409,255 (30) | 43,409,258 (27) |
| 17 | 43,409,335 | 43,409,333 (2) | 43,409,334 (1) |
| 17 | 43,409,384 | 43,409,381 (3) | 43,409,376 (8) |
| 17 | 43,409,431 | 43,409,381 (50) | 43,409,425 (6) |
| 74 | 19,108,003 | 19,107,972 (31) | 19,108,003 (0) |
| 74 | 19,108,051 | 19,108,045 (6) | 19,108,043 (8) |
| 74 | 19,108,097 | 19,108,090 (7) | 19,108,091 (6) |
| 74 | 19,108,144 | 19,108,141 (3) | 19,108,091 (53) |

Before inhibition, both captured survivors have **34-step member periods**, with
their members three steps apart: copy 17 u/v at 43,409,187/43,409,190,
43,409,221/43,409,224, 43,409,255/43,409,258; copy 74 u/v at
19,107,904/19,107,901, 19,107,938/19,107,935, 19,107,972/19,107,969.
The middle two arrivals in each copy find both members within
their **22-step refractory interval**. The first copy-17 arrival finds neither
member in that interval; the first copy-74 arrival coincides with v. The final
arrivals find only v (17) or u (74) recently fired, after the pair has separated.
After the last arrival, copy 17 next fires u/v at **43,409,460/43,409,508**;
copy 74 at **19,108,226/19,108,174**. Inhibition stretches the train and changes
its phase but does not extinguish either loop.

This directly confirms survival in the predicted fast-mode/reset interaction,
while refining the isolated reconstruction's simplified “each arrival within
0–14 steps / just after both” account. Copy 74's **0/48/94/141** arrival offsets
match its isolated noise-free prediction exactly. Copy 17's captured
**0/50/99/146** differs slightly from the predicted **0/49/96/140**. Spike timing
and the wired delays are captured evidence; refractory/membrane effects are
interpreted using the recorded circuit physics, not measured voltage traces.

### The survivor's OR resumes; valid ignition and completion do not

| Probe | Copy 17, bit 8 | Copy 74, bit 3 |
|---|---|---|
| Valid OR | **3419**: last pre-gap **43,409,258**, resumes **43,409,870**, last **45,271,484** | **3961**: last pre-gap **19,107,969**, resumes **19,108,553**, last **20,956,544** |
| OR interspike gap | **612 steps**; last/resumed at reset trigger +48/+660 | **584 steps**; last/resumed at trigger +37/+621 |
| Valid latch u/v last spikes | **3420/3421**: **43,409,307 / 43,409,273** | **3962/3963**: **19,107,986 / 19,108,024** |
| Ignition relay | **3422**: last **43,355,991**, no spike for this reset/rewrite | **3964**: last **19,050,256**, no spike for this reset/rewrite |
| OR-driven `ign.edge_inh` | **3423**: **43,409,279 → 43,409,916**, then fires through **45,271,355** | **3965**: **19,107,988 → 19,108,599**, then fires through **20,956,562** |
| Valid-driven `ign.hold_inh` tail | **3424**, last **43,409,381** | **3966**, last **19,108,057** |

The OR resumes **before READY and COPY** in each copy, so its initial renewed
drive comes from the old survivor. After the final reset arrival, the OR fires
**12,501 / 25,231** times and its edge inhibitor **12,685 / 27,240** times.
The valid latch remains dark throughout. The edge inhibitor's nominal
**−7,966-q / 18-step** connection continues delivering inhibition to the silent
ignition relay; the lighter valid-driven hold inhibitor dies. This is the
predicted failure to re-arm, with renewed OR excitation and sustained inhibitory
input observed directly. The gap is insufficient at these draws, consistent
with the isolated experiments. Copy 74's captured OR gap **584** is slightly
longer than the isolated survivor's **555** (+19 → +574), with the same outcome.

All ten other master valids re-ignite. Only the survivor-dependent tree path
fails to re-ignite: copy 17 `c0_4.L` **3465/3466**, `c1_2.L` **3483/3484**,
root `c3_0.L` **3495/3496**; copy 74 `c0_1.L` **4024/4025**, `c1_0.L`
**4048/4049**, `c2_0.L` **4066/4067**, root **4072/4073**. The roots' last u/v
spikes are **43,409,258/43,409,283** and **19,107,987/19,107,966**; neither root fires after
READY. DONE **3632/4209** never fires for these transactions. DONE inhibitors
**3633/4210** stop at **43,409,354 / 19,108,141**, confirming that the missing
input is renewed master completion, rather than a sustained DONE inhibitor.

### COPY writes the selected word; copy 74 retains an extra bit-3 rail

COPY u/v **3542/3543** starts at **43,410,054/43,410,112** in copy 17, and
**4119/4120** at **19,108,808/19,108,867** in copy 74. Every selected `cp*.edge` fires
exactly once after reset; every opposite arm is silent. The complete arm evidence
is below (`r` is the selected rail; each entry is neuron ID @ spike step).

| Bit | Copy 17 r / arm ID @ step | Copy 74 r / arm ID @ step |
|---:|---|---|
| 0 | r0 / **3546 @ 43,410,135** | r1 / **4127 @ 19,108,892** |
| 1 | r0 / **3554 @ 43,410,135** | r1 / **4135 @ 19,108,886** |
| 2 | r0 / **3562 @ 43,410,132** | r0 / **4139 @ 19,108,894** |
| 3 | r0 / **3570 @ 43,410,136** | r1 / **4151 @ 19,108,888** |
| 4 | r1 / **3582 @ 43,410,143** | r1 / **4159 @ 19,108,904** |
| 5 | r1 / **3590 @ 43,410,135** | r1 / **4167 @ 19,108,893** |
| 6 | r0 / **3594 @ 43,410,138** | r0 / **4171 @ 19,108,889** |
| 7 | r0 / **3602 @ 43,410,136** | r1 / **4183 @ 19,108,889** |
| 8 | r0 / **3610 @ 43,410,134** | r0 / **4187 @ 19,108,881** |
| 9 | r0 / **3618 @ 43,410,136** | r0 / **4195 @ 19,108,899** |
| 10 | r0 / **3626 @ 43,410,149** | r0 / **4203 @ 19,108,895** |

In copy 17, the ten cleared selected master u neurons start between
**43,410,178–43,410,190**; bit 8 r0 was already live. In particular new bit 4 r1
**3337** starts **43,410,189**, bit 5 r1 **3341** at **43,410,179**, and
Z=0 rail **3355** at **43,410,183**. The final master is the exclusive word
**48, C/Z/V=0**, despite its missing valid8/completion. COPY therefore delivers
the desired data, but does not complete the transaction.

In copy 74 all eleven selected master u neurons start between
**19,108,921–19,108,953** and hold to capture end. For bit 3, Q's selected r1 **3687**
drives arm **4151** at **19,108,888**; the new `M.b3r1` u/v **3910/3911**
starts at **19,108,936/19,108,994**. The opposite arm **4147** never fires;
the old `M.b3r0` **3908/3909** is the reset survivor, not a second COPY write.
Both rails then persist beside each other. The other ten bits match **187,
C/Z/V=0** exclusively, so COPY writes the requested rails but does not leave
an exclusive architectural word.

`M.fault3.and` **3967** fires **5,414** times, starting at **19,109,292**,
exactly **READY +528**, then **19,109,674 / 19,110,057 / 19,110,442**
(gaps **382/383/385**), through **20,956,487**. This agrees with the isolated
forced-survivor prediction (**READY +526**, initial periods **380–385**).
The entire captured fault train has variable intervals (**217–394**, median
**359** steps), so the initial near-periodic match is not a claim of a fixed
period over the whole strayed capture. All other master fault gates in copy 74,
and all master fault gates in copy 17, remain silent.

The capture thus confirms the predicted unchanged-bit survivor **17 b8r0** and
flipping-bit survivor **74 b3r0**, their blocked valid one-shots, and successful
COPY with the latter's persistent double rail. The refinements concern the
actual reset phase and OR recovery timing, not the initiating latch or the
completion blockage. Analysis and assertion scripts were kept in the task's
scratch directory outside the repository and run with the existing repository
virtualenv; only this document is changed.
