# Tick seed 107, copy 18: a live REQ-false repair defeats the next clear

The first blocked cell is **`c2_sel`, on its second START**. Neuron **11828,
`c2_sel.req.c0_addr0.u`**, stays lit when the second `c0_add` DONE should clear it;
its partner is 11829. The cause precedes the second output: the repair relay
**11970, `c2_sel.relight.c0_add.edge`, fires at 37,674 into an already-live latch**.
The rail accelerates from a 43-step period to a persistent 34-step period. At the
next DONE, the three clear pulses at 79,310 / 79,354 / 79,398 slow that train but
do not kill it. REQ true also stays lit, so false permanently vetoes the join.
This is in the opt-in repair path, not a missing input or an IDLE/commit failure.

## Artifact and method

Read-only sources, under `/Users/jeremiahgassensmith/programming/drosophilos/data/a2/`:
`tick_s107_node18_compact.npz` and `kc_tick_B90s107b.json`. The NPZ contains
23,030,349 spikes, 17,872 distinct neuron IDs, steps 22 through 299,999;
dt = 0.1 ms. This artifact is a **30 s observation**, not the whole 90 s campaign.
No new long simulation was run. Rebuilding the campaign's `tick` compiler spec
with `relight_requests=True` gives the original 28,439 neurons / 51,044 synapses;
every `(uniq, role_of)` entry matches the rebuilt `net.roles` exactly.

The record has only `c4_sel = [(66863, 25)]`, `c9_sel = [(102409, 88)]` for copy 18.
Tokens are `[5, 5, 250, 3, 0, 40, 40, 40]`; reference output pairs are
`[25,88], [30,86], [24,84], [27,82], [27,80], [67,78], [107,80], [0,78]`.
The record reports no faults, timeouts, or invalid output words.

All tables use integer steps. A rail interval lists its first and last observed
spikes, splitting at gaps >150 steps; a last spike bounds cessation, rather than
measuring membrane voltage. `end` means activity still present in the final
100 steps. `—` means no spike in the captured interval. Cell roles and `IN.*`
are fully in the capture filter; lowercase `input.commit_pulse` and `input.cg.*`
are outside it, so their absence is not evidence of silence.

## Healthy and failing request, compared in the same dump

| Event / role (prefix `c2_sel` unless stated) | Healthy first cycle | Failed second cycle |
|---|---|---|
| `c0_add.M` completion; `c0_add.done.edge` | 23842; 23881 | 79095; 79134 |
| `.trigger.c0_add` | 23922 | 79175 |
| `.req.c0_add.received` (11833) | 23963 | 79216 |
| `.req.c0_addr1.u` rises | 23964 | 79218; remains live through 299980 |
| `.req.c0_add.k1.start.edge` | 24006 | 79259 |
| `.req.c0_add.k1.h1`, `.h2` | 24059, 24112 | 79310, 79363 |
| `.req.c0_add.k1.inh` (11836), every pulse | 24057, 24100, 24145, 24248 | 79310, 79354, 79398 |
| False rail before clear | 43-step period | 34-step period |
| False u / v last spikes or recovery | Last 24101 / 24163 | u: 79313, 79355, 79416, 79512, 79579; v: 79310, 79353, 79449, 79544; both survive |
| `c1_and.done.edge`; its REQ true | 34924; 35005 | 90332; 90414 |
| `.go.g0.pb.edge`; `.go.p0.pulse` | 36106; 36150 | —; — |
| `.go.p0r1.u`; `.go.g1.pa.edge` | 36190; 36853 | —; — |
| START; ACT^d | 36896; 37467 | —; — |
| Stage completion; commit; master completion; DONE | 39805; 41329; 45391; 45432 | —; —; —; — |

The first clear has a fourth inhibitory spike even though the circuit has three
driving taps; the second emits three. Receipt and its edge detector **do recover**:
there are 55,253 steps between receipts. The failure is at the recipient latch,
not a lost receipt, a missing kill train, or a fresh request inside the repair window.

| The event that changes the false latch | Observed steps |
|---|---|
| START re-ignites false successfully | START 36896; false u 36936, 37015, 37074, then settles toward 44-step periods |
| Old true dies; its repair veto drains | True last 36998; veto last 37077 |
| ACT^d + 3 tap; repair edge | 37612; **37674** |
| False u around the unnecessary repair | 37577, 37621, 37665, **37699, 37732, 37765, 37799**, 37833, 37867 |
| False v around repair | 37601, 37645, 37689, 37727, 37762, 37796, 37830 |
| Persistent faster orbit | Every u and v interval in steps 38000–78000 is **34** (1176 intervals each) |
| Next clear fails; ordinary circulation resumes | After 79512, u gaps 67, 55, 50, 47, 46, 45; eventually **43**, still live at 299993 |

There is no other designed excitation of false: its inputs are its partner,
START, the repair, and the inhibitory clear. The repair inserts additional
circulating activity into a working latch, contradicting the earlier assumption
that re-ignition of a live rail is harmless. Spikes establish the transition;
the artifact does not contain membrane states or individual stray events.

For the second request, `.go.g0.ad.d11` begins at 79,840 while the other source's
false rail is still live; `pa` is correctly vetoed and its driver then holds its
edge inhibitor. The other source's false rail finally dies at 90,562.
`.go.g0.bd.d19` starts at 91,469, but `.go.g0.pb.veto` (11902) is continuously
driven by the **wrong live c0 false rail** from 36,989 through 299,975. Thus `pb`
never fires, the passed-true latch never rises again, and the final source
rechecks also retain this veto. IDLE has been true since 46,523. No second START
occurs by 299,999, though both source requests and both source masters are valid.

## Every cell's execution events

Each row is one transaction; `+` denotes the same guard doublet, not another
transaction. ACT^d is `.actd.d10`; stage/master entries are completion-u rises.
All start, ACT^d, commit and DONE pulses in the 30 s dump appear here.

| Cell / transaction | START | ACT^d | Stage | Commit pulse | Master | DONE |
|---|---:|---:|---:|---:|---:|---:|
| c0_add / 1 | 13731 | 14292 | 18131 | 19691 | 23842 | 23881 |
| c0_add / 2 | 68450 | 69014 | 73420 | 74981 | 79095 | 79134 |
| c1_and / 1 | 24668 | 25221 | 30026 | 30820 | 34882 | 34924 |
| c1_and / 2 | 79926 | 80482 | 85425 | 86219 | 90291 | 90332 |
| c2_sel / 1 | 36896 | 37467 | 39805 | 41329 | 45391 | 45432 |
| **c2_sel / 2** | **—** | — | — | — | — | — |
| c3_xor / 1 | 46232 | 46803 | 50362 | 51146 | 55368 | 55408 |
| c4_sel / 1 | 57374 | 57961 | 60404 | 62672 | **66863** | 66905 |
| c5_sub / 1 | 68858 | 69424 | 74339 | 75127 | 79193 | 79233 |
| c6_and / 1 | 80021 | 80580 | 85344 | 86146 | 90195 | 90235 |
| c7_add / 1 | 13711 | 14297 | 18438 | 19234 | 23308 | 23349 |
| c7_add / 2 | 103991 | 104573 | 108738 | 109532 | 113623 | 113664 |
| c8_sub / 1 | 13741 | 14301 | 18479 | 19280 | 23223 | 23265 |
| c8_sub / 2 | 104035 | 104594 | 108568 | 109372 | 113274 | 113316 |
| c9_sel / 1 | 92227 | 92801 | 95288 | 98343 | **102409** | 102450 |
| c10_xor / 1 | 104001+104042 | 104578+104626 | 108036 | 108799 | 112917 | 112959 |
| c11_sub / 1 | 13710 | 14272 | 18214 | 19015 | 23062 | 23108 |
| c11_sub / 2 | 125036 | 125605 | 129536 | 130340 | 134369 | 134415 |
| c12_sel / 1 | 114904 | 115480 | 117874 | 119412 | 123445 | 123485 |

After the second output (102,409), c7/c8 consume the newly published monster
state, c10 computes contact, c12 publishes health, and c11 finishes its second
decrement. Those are the tail of work already enabled; none can unblock c2.
Every started transaction completes and commits. No completed cell stage remains
pending at the end; only the input stage is waiting to commit.

## All request rails: rises and deaths

`F` / `T` are `{cell}.req.{source}r0.u` / `r1.u`. Sources listed here are
exactly the cell's request sources (constants have no requests). Image-time T
intervals identify feedback: c0←c4, c5/c7/c8←c9, c11/c12←c12. State inits are
c4=20, c9=90, c12=100. Input is an extra trigger for c7/c8/c11.

| Cell ← source | F intervals | T intervals |
|---|---|---|
| c0_add ← c4_sel | 13771–67125; 68491–end | 25–13869; 66991–68556 |
| c0_add ← input | 25–11926; 13774–21659; 68493–130063 | 11834–13849; 21511–68595; 129945–end |
| c1_and ← c0_add | 24–24064; 24716–79306; 79975–end | 23962–24777; 79215–80025 |
| **c2_sel ← c0_add** | 23–24101; **36936–299993** | 23964–36998; **79218–299980** |
| c2_sel ← c1_and | 25–35214; 36936–90562 | 35005–37048; 90414–end |
| c3_xor ← c2_sel | 24–45659; 46285–end | 45512–46365 |
| c4_sel ← c2_sel | 24–45628; 57417–end | 45518–57467 |
| c4_sel ← c3_xor | 24–55611; 57415–end | 55491–57521 |
| c5_sub ← c9_sel | 68899–102715 | 24–68979; 102529–end |
| c5_sub ← c4_sel | 27–67086; 68903–end | 66988–69014 |
| c6_and ← c5_sub | 24–79515; 80063–end | 79316–80124 |
| c7_add ← c9_sel | 13753–102640; 104033–end | 24–13898; 102530–104120 |
| c7_add ← input | 25–11957; 13750–21657; 104030–130064 | 11840–13831; 21518–104202; 129954–end |
| c8_sub ← c9_sel | 13782–102634; 104078–end | 26–13861; 102536–104135 |
| c8_sub ← input | 24–12015; 13783–21672; 104077–130089 | 11836–13884; 21514–104150; 129949–end |
| c9_sel ← c7_add | 24–23555; 92267–113946 | 23431–92327; 113746–end |
| c9_sel ← c8_sub | 24–23440; 92266–113509 | 23348–92346; 113399–end |
| c9_sel ← c6_and | 27–90460; 92272–end | 90320–92327 |
| c10_xor ← c9_sel | 24–102644; 104043–end | 102533–104111 |
| c10_xor ← c4_sel | 24–67097; 104040–end | 66985–104155 |
| c11_sub ← c12_sel | 13753–123716; 125078–end | 25–13823; 123567–125178 |
| c11_sub ← input | 25–11970; 13750–21665; 125079–130077 | 11834–13799; 21515–125187; 129950–end |
| c12_sel ← c12_sel | 114945–123675 | 24–115014; 123568–end |
| c12_sel ← c11_sub | 25–23310; 114947–134619 | 23189–115095; 134496–end |
| c12_sel ← c10_xor | 25–113171; 114945–end | 113040–115021 |

## All received one-shots and repair relays

Columns are `.req.{source}.received` and `.relight.{source}.edge`, respectively.
Feedback image requests intentionally have no received pulse. All source DONEs
have their corresponding received pulse; c2's c1 repair is vetoed on its only run.

| Cell ← source | Received pulses | Relight pulses |
|---|---|---|
| c0_add ← c4_sel | 66989 | 14522; 69245 |
| c0_add ← input | 11835; 21513; 129949 | 14516; 69240 |
| c1_and ← c0_add | 23959; 79212 | 25442; 80707 |
| c2_sel ← c0_add | 23963; 79216 | **37674** |
| c2_sel ← c1_and | 35007; 90416 | — |
| c3_xor ← c2_sel | 45513 | 47014 |
| c4_sel ← c2_sel | 45523 | 58165 |
| c4_sel ← c3_xor | 55491 | 58179 |
| c5_sub ← c9_sel | 102534 | 69642 |
| c5_sub ← c4_sel | 66992 | 69641 |
| c6_and ← c5_sub | 79315 | 80795 |
| c7_add ← c9_sel | 102531 | 14505; 104786 |
| c7_add ← input | 11836; 21514; 129949 | 14506; 104792 |
| c8_sub ← c9_sel | 102535 | 14520; 104810 |
| c8_sub ← input | 11838; 21516; 129951 | 14524; 104813 |
| c9_sel ← c7_add | 23433; 113748 | 93009 |
| c9_sel ← c8_sub | 23346; 113399 | 93011 |
| c9_sel ← c6_and | 90317 | 93007 |
| c10_xor ← c9_sel | 102532 | 104781 |
| c10_xor ← c4_sel | 66991 | 104789 |
| c11_sub ← c12_sel | 123562 | 14486; 125816 |
| c11_sub ← input | 11833; 21513; 129950 | 14487; 125820 |
| c12_sel ← c12_sel | 123565 | 115685 |
| c12_sel ← c11_sub | 23191; 134498 | 115692 |
| c12_sel ← c10_xor | 113041 | 115678 |

## IDLE and the input register

All IDLE pairs finish correctly true. There is no stuck-busy cell.

| Cell | IDLE false intervals | IDLE true intervals |
|---|---|---|
| c0_add | 13773–25079; 68492–80342 | 25–13834; 24952–68544; 80214–end |
| c1_and | 24711–36111; 79969–91509 | 26–24758; 36008–80016; 91415–end |
| c2_sel | 36936–46617 | 25–37022; 46523–end |
| c3_xor | 46272–56641 | 23–46341; 56511–end |
| c4_sel | 57415–68113 | 26–57507; 67999–end |
| c5_sub | 68898–80444 | 23–68997; 80334–end |
| c6_and | 80065–91434 | 24–80161; 91326–end |
| c7_add | 13753–24555; 104032–114896 | 25–13809; 24453–104103; 114755–end |
| c8_sub | 13783–24500; 104076–114514 | 23–13860; 24343–104136; 114395–end |
| c9_sel | 92267–103643 | 25–92327; 103547–end |
| c10_xor | 104043–114177 | 25–104123; 114031–end |
| c11_sub | 13751–24300; 125077–135619 | 24–13833; 24199–125133; 135515–end |
| c12_sel | 114947–124682 | 25–115054; 124569–end |

| Input event | Token 1 (5) | Token 2 (5) | Token 3 (250) | Token 4 (3) |
|---|---:|---:|---:|---:|
| `IN.Q.comp.c2_0.L.u` rise | 5023 | 14656 | 24383 | 132808 |
| `IN.M.comp.c2_0.L.u` rise | 11714 | 21392 | 129827 | — |
| `IN.done.edge` | 11754 | 21432 | 129868 | — |
| `IN.Q.ready` (host's next-load permission) | 12611 | 22292 | 130721 | — |

The host initially loads token 1, then loads behind those three READY pulses.
Token 3 waits in Q from 24,383 until the last input reader, c11, starts on token 2
at 125,036. Token 4 completes Q and stays there (last spike 299,981);
`IN.creqr1.u` is live from 132,894 through 299,998. All four input readers retain
token 3 requests, awaiting feedback that cannot propagate past c2. The lack of
another READY is backpressure, not the initiating fault or a host pacing error.

## Repair and validation

In `relight_requests=True`, add **false as well as true** to the existing repair
relay's veto inputs. A working false rail is established well before the tap and
blocks the unnecessary ignition. A failed START ignition's isolated early spike
has over 55 ms to drain before the tap; the existing dark-rail regression checks
that recovery. Request ownership remains unchanged: only START clears true,
and DONE clears false even when a late request misses the repair veto window.
No delay or shared handshake is changed; the default flag remains false.

`test_live_request_false_repair_cannot_accelerate_the_latch_and_defeat_done_clear`
uses two RefSim copies of a one-bit constant-zero MOV with two input triggers:
1,295 neurons per copy (2,590 total), **7.9 s** neural time. Both receive the
observed source-DONE spacings and a START+716-step repair tap; a timed extra
input supplies the healthy first clear's fourth inhibitory spike. Copy 1 zeros
only the new false-to-veto edge. Both share a selected bounded corner on false:
loops +12%, kill magnitudes −3.1%/−4.0%, START/repair ignitions −3.2%/−5.3%,
threshold −0.4 mV and bias +0.4 mV. These are **synthetic parameters**, not a
reconstruction of the campaign's unrecorded weights or stray stream. The period
change is 41→40 rather than 43→34; the reproduced causal sequence is a live
repair, persistent acceleration, failed clear, both-live request, and lost START.

| Regression event | Fixed copy 0 | Old repair on copy 1 |
|---|---|---|
| First clear, all inhibitory spikes | 3203, 3248, 3292, 3374 | Same; false clears |
| First START; repair tap | 16057; 16773 | Same |
| Repair relay | — | 16835 |
| False period, steps 20000–57000 | 41 | 40 |
| Second clear, all inhibitory spikes | 58456, 58501, 58545 | Same |
| False after second clear | Last spike 58510 | Survives; returns to period 41 |
| Second request pending before other source | True live, false dark | **Both live** |
| Second START; second DONE | 71465; 78459 | —; — |

Assertions cover successful first clearing, received pulses, the tap timing,
the two rail rates, pending requests, failed clearing only on copy 1, and exact
START/DONE counts. Default four-bit MULP counts remain **7,108 / 12,400** (asserted);
its roles, edges, weights, delays, groups and biases also match HEAD, including
byte-for-byte equality of the topology arrays. The opt-in tick adds 25 synapses
and no neurons. Validation passed: `/Users/jeremiahgassensmith/programming/drosophilos/.venv/bin/python -m pytest tests/test_kernel.py -q -m 'not slow'`
(18 passed, the two expected §10.4 xfails); no slow tests ran. The macOS run used
worktree `TMPDIR`, Command Line Tools first on `PATH`, and its macOS `SDKROOT`.
The 100-copy mix-B rerun and the separate fan-out wrong-value
mechanism are not established by this one-copy localization.
