# Stage D completion stalls: seed 108, base 17/74 and rate-robust 11/24

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
