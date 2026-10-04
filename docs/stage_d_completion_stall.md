# Stage D completion stalls: seed 108, base 17/74 and rate-robust 11/24

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
| base 74 `c5_sub` | 323 | 197, C | 187, C | 1, 2, 3, 4, 5, 6 |
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
  nominal, and 0 with the compact candidate.

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

The committed reconstruction was rerun in float64 **RefSim**. Recording now keeps
only the observed neurons, coalesces same-step integer external events, and slices
each node once before classifying it. Integration, delivery, static draws and RNG
streams are unchanged; a regression compares the optimized observation with the
ordinary RefSim trace. The thirteen non-slow completion tests finish in about
12 seconds on this worker, including the original 2,000-reset result **8 / 0 / 0**.
The full **1,600 seeds per copy and setting** are now explicitly slow-marked.

| Repeated full isolated path | Stalls / 1,600 | Old-rail survivors / 1,600 |
|---|---:|---:|
| base 74, original reset | 24 | 50 |
| base 17, original reset | 4 | 4 |
| RR 24, original reset | 1 | 1 |
| RR 11, original reset | 0 | 0 |
| base 74, selected robust stage/master resets, identical seeds | 0 | 0 |

These reruns use the committed numeric words. In particular, copy 74's literals
are **197 → 187**, with C/Z/V all zero; the earlier table's “C” annotation does
not match those literals. The reproduced rates refer to the code's words, not a
new reconstruction with C set. They remain conditional isolated experiments,
not a replay of the missing CUDA stream or a full Stage D campaign. The nominal
1,600-seed settings and compact settings for the other copies remain available in
the slow suite; the table above lists the full-path settings rerun for this change.

### Choice and rejected candidates

`robust_register_reset=False` remains the default. Enabling it selects **four
taps, zero-delay links between taps, and 1.75 × loop inhibition** for the input
stage/master and each cell's Q/master domains. The final **four** links of their
existing fifteen-hop READY chains use **100-step delays instead of 18**. Thus each
READY is delayed exactly **328 steps / 32.8 ms** from the same reset trigger.
There is no change to the control machine, producer resets or kernel/request kills.

The request clear's 1.1 strength was insufficient for the wider independent
register corners. These are 256 reset phases per setting, with four closely
spaced initial ignitions and a later 300-q kick. Storage is +30% loop, −1.2 mV
threshold, with independently **12% weak clear edges**. The slow controller has
12% weak positive edges and +0.6 mV thresholds, and emits only **three** spikes.

| Compact strength | Survivors, nominal controller / 256 | Survivors, slow controller / 256 | Slow-storage recovery from first arrival, fast READY | Margin with selected READY delay |
|---:|---:|---:|---:|---:|
| 1.1 | 76 | 256 | 825 steps | 120 steps |
| 1.25 | 0 | 256 | 850 | 95 |
| 1.35 | 0 | 208 | 875 | 70 |
| 1.5 | 0 | 0 | 900 | 45 |
| **1.75 (selected)** | **0** | **0** | **925** | **20** |

The noisy extension rejected 1.5 despite its clean phase sweep: **59 / 40,000**
survivors at fast storage / weak clear / slow controller. Strength 1.75 gives
**0 / 40,000** on the identical stream. This is why the selected strength is higher
than either the request clear or the deterministic minimum.

The unextended nominal READY budget is only 777 steps from the first inhibition,
and a fast READY chain reduces it to 617. Even 1.1 misses the slow-storage margin
without extending READY. Merely adding taps or counting configured taps as arrivals
would not qualify either the kill or recovery. The original reset also leaves
52/256 copy-74 and 2/256 copy-24 survivors in the deterministic phase regression.

### Survivor and recovery evaluation

The reduced primitive is the real `add_latch` plus `add_reset`, optionally with
the real fifteen-hop READY chain. For copies 74/17/24 and stage copy 18, every
included edge, threshold and bias is role-mapped from seed 108, B=100, at the
original topology. Copy 18 is **c3_xor.Q.b9r0**. Its four initial ignition offsets
match the captured burst (0/1/4/13 steps, nominal ignition quanta); the subsequent
strays are newly sampled, not captured. The known master draws retain their own
controllers and independently drawn fan-outs, rescaled to the candidate strength.

Each slow survivor setting applies **40,000 resets**, batched at 1,000, with reset
times uniform in [3,000, 4,500), four initial ignitions, a later kick alternating
members, and **5-Hz × 150-q strays on every primitive neuron** through step 6,200.
Survival means either member still fires 1,500 steps after reset. This also
challenges late re-ignition. The deterministic 256-phase cases are fast regressions.
The selected strength produced **0 survivors in all thirteen 40,000-reset
settings: 520,000 resets total**, including all four reconstructed draw sets.

Reload uses eleven reset phases and a conservative 25-step grid, tests sustained
firing 1,500 steps after the ignition, and additionally tests **exactly at READY**.
Slow corners independently weaken the loop and ignition by 8/12%, raise latch
thresholds by 0.4/0.6 mV, and strengthen clear edges by 8/12%. Fast corners use
+20/+30% loops, −0.8/−1.2 mV thresholds and 12% weak clears. Asymmetric and mirrored
corners mix +30/−12% loop directions, −1.2/+0.6 mV thresholds and −12/+12% clears.
The first and last arrivals are measured from **all actual inhibitor spikes**.

| Storage / controller | Survivors / 40,000 | Actual arrivals from first (steps) | Recovery from first | READY from first | Margin |
|---|---:|---|---:|---:|---:|
| master 74, its draws | 0 | 0/45/84/144 | 650 | 1,098 | 448 |
| master 17, its draws | 0 | 0/42/82/148 | 625 | 1,069 | 444 |
| master 24, its draws | 0 | 0/48/89/149 | 700 | 1,101 | 401 |
| stage Z0 18, its draws | 0 | 0/41/78/141 | 700 | 1,103 | 403 |
| nominal | 0 | 0/43/81/138 | 675 | 1,105 | 430 |
| slow 8% | 0 | 0/43/81/138 | 825 | 1,105 | 280 |
| slow 12% | 0 | 0/43/81/138 | 925 | 1,105 | 180 |
| fast 20%, weak clear | 0 | 0/43/81/138 | 525 | 1,105 | 580 |
| fast 30%, weak clear | 0 | 0/43/81/138 | 500 | 1,105 | 605 |
| asymmetric | 0 | 0/43/81/138 | 675 | 1,105 | 430 |
| mirrored asymmetric | 0 | 0/43/81/138 | 875 | 1,105 | 230 |
| fast 30% / fast controller (kill); slow 12% / fast controller (reload) | 0 | 0/34/66/111 | 900 | 955 | 55 |
| fast 30% / slow controller (kill); slow 12% / slow controller (reload) | 0 | 0/60/115 | 850 | 1,360 | 510 |
| slow 12% / independently fast READY | phase regression | 0/43/81/138 | 925 | 945 | **20** |

Fast/slow controllers perturb all their positive input edges by ±12% and their
thresholds by ∓0.6 mV; independently fast READY perturbs only READY-chain neurons.
The separate kill/reload storage settings in the last rows are intentional: fast
storage with weak clears stresses killing; slow storage with strong clears stresses
recovery. These are bounded conditional corners, not a universal noise guarantee.

### Complete reset domains, legitimate reloads and cost

`compact_register_resets` is a **final build pass**, after staged-commit wiring,
`extend_reset`, Z generation and rate-reader wiring. It rescales every inhibitory
edge from the selected controllers, including scaled/mirrored edges. Applying it
only inside the initial `add_register` construction would miss later targets.

| Reached state | Qualification |
|---|---|
| Q/M rails, validity and completion | Phase/noisy survivor sweeps; actual weak COPY ignition into slow master rails; a second stage word at the first exact Q READY; both resulting words decode correctly, no faults, base and RR |
| ALU internal latches/gates, ACT, COMMIT, grant and COPY | Two real ADD transactions; every target of the extended Q reset is silent from READY−100 onward after the final word; multi-cell ADD→XOR wraps and reloads correctly |
| FAULT | An injected double rail raises FAULT and discards Q; FAULT is dark before READY; a clean word loaded exactly at READY commits once, base and RR |
| `zero_once` OR/relay/inhibitors | Included in the same fan-out and quiet-at-READY assertion; multi-cell runs with the option on and off |
| RR v2 readers/qualifiers | All mirrored edges included; exact 16× reader-reset gain checked for every source; RR next-COPY, stage reload, fault discard and two-transaction domain tests |
| Master power-up veto; input TIMEOUT | Their reset edges are included by source, not by target-name filtering; state-cell veto coverage is asserted; default watchdog paths remain present |

The multi-cell regression covers all four combinations of `zero_once` and
`robust_request_clear`, each with base and rate-robust readers, using generic and
specialized datapaths. A further two-cell RefSim run uses seed-108 static mix-B
noise and geometric inter-arrival sampling of 5-Hz × 150-q strays; outputs are
correct with no faults, timeouts, refusals or bad outputs. This does not replace
a noisy, long-running whole Stage D campaign.

**Costs:** zero added neurons and zero added logical synapse edges. A tick build
still has **29,375 / 52,808**, or **30,643 / 55,936** with RR. Its 28 selected
controllers change 84 tap delays and 112 READY delays; 6,322 inhibitory edge
weights change in base mode, 7,776 with RR. Their quanta scale by 6,337/2,716
(total absolute quanta increases by 22,891,962 / 92,813,472 respectively, including mirrors).
READY gains **32.8 ms per reset**; a stage/master sequence contains two such
delays, so both must be budgeted when comparing transaction latency. This is not
a claim of zero placement-weight cost. Ordered default fingerprints, including
both control-machine hashes, remain unchanged with the option off.

The flag is recorded in `Pipeline.build_options`, exposed as
`--robust-register-reset` by `stage_d` and `kernel_campaign`, and honored by
`stall_diag` and Stage D `--recheck` (including nested build records). Old records
default false. Recheck now forwards the recorded timing/drive policy as well,
so an unqualified combination cannot silently rebuild using default timings.
Qualified builds use default physics/drive, standard control/timing, four ordinary
kernel/request taps, no retry-clear, selected-rail COPY, one unpaced input stream,
and 1/2/4/8-bit MOV/ADD/SUB/AND/OR/XOR/SEL/ROM LOAD cells. RAM, multipliers, pacing,
other widths/streams and changed timing/control policies are explicitly rejected.

The requested command is blocked before collection by sandbox denial of uv's
default cache, `~/.cache/uv/sdists-v9/.git`:

```sh
TMPDIR=/private/tmp/claude-501/tmpdir uv run pytest -q tests/test_robust_reset.py tests/test_completion_stall.py tests/test_build_options.py -m 'not slow'
```

Validation uses the same files and selection through `uv run --no-sync`, the
existing repository virtualenv, a writable task-local `UV_CACHE_DIR`/pytest
basetemp, and this worktree on `PYTHONPATH`: **131 non-slow tests passed**.
The final noisy-kernel event-batching optimization also passes its focused
regression (3.45 s). The 520,000-reset surveys and the five full-path rows above
were run separately; `git diff --check` passes.
No commit is attempted; integration and committing belong to the orchestrator.
