# Comparing the current neuron model with the released Shiu model

External event rate is not neuronal firing rate. At the project's 100,000-quanta
stimulus, 2 Hz events produce **13.67 Hz** during the 300-ms drive on an isolated
neuron under M0, versus **2 Hz** under M5. One event produces seven M0 spikes but
one M5 spike within 100 ms. These are different input-to-output conversions.

**The existing anatomical H0 circuit does not retain its AND behavior under M5.**
The real MCNS 15-neuron embedding passes all four transactions under M0, including
reset and the existing <60-ms completion check. M5 passes only (0,0); (0,1), (1,0)
and (1,1) never complete in 150 ms. The same outcome appears on the planted synthetic
fixture. No circuit weight or transaction stimulus was retuned.

The first table is the requested event-rate conversion. All columns use the **same
100,000 quanta = 1718.75 mV per event** to isolate the switches. Firing is spikes
during [0,300 ms) divided by 0.3 s and averaged across 50 periodic stimulus phases;
spikes after drive stops are excluded. Boundary latency therefore makes M0's 2-Hz
entry slightly less than 2 × 7. These periodic averages are not Poisson estimates.

| configuration | 0 Hz events → firing Hz | 2 Hz → firing Hz | 10 Hz → firing Hz | 50 Hz → firing Hz |
| --- | --- | --- | --- | --- |
| M0_current | 0 | 13.67 | 68.13 | 340.8 |
| M1_clear_g | 0 | 2 | 10 | 50 |
| M2_freeze_g | 0 | 78.53 | 234.33 | 336.67 |
| M3_stim_v | 0 | 2 | 10 | 50 |
| M4_stim_norefrac | 0 | 94.53 | 471.73 | 2357.53 |
| M5_shiu | 0 | 2 | 10 | 50 |

Clearing g or moving stimulation to V removes the isolated-neuron burst. Freezing g
alone prolongs it, and removing refractory alone greatly increases its spike count.
These effects interact: with g cleared, freezing g and removing refractory no longer
change the single-event count; with V stimulation, none of the other switches changes
that count on F1. The factorial appendix measures these interactions directly.

I recommend **M5 as the primary model for new biological/connectome experiments**,
because it reproduces the released Shiu rules, while preserving M0 as a labelled
engineering and historical comparison. This is a recommendation for the project owner,
not a default change. Adopting M5 has a concrete cost: H0 and every dependent circuit
would need recertification or redesign. Matching biological constants alone does not
make the two implementations interchangeable.

M5 currently runs only in ref64; the fast backends deliberately refuse it. A future
whole-brain replication would first need backend support certified against this mode.

This bounded experiment does **not** show that the ~30,000-neuron surround runaway
or the 0/50 live placed-adder results were stimulation artefacts. Those were whole-brain
runs and are not repeated here. In particular, the silent-surround adder failures cannot
be attributed to sensory event amplitude by this experiment.

## Configurations and fixtures

All constants are the unchanged `Params` defaults: dt 0.1 ms, E_L/reset −52 mV,
threshold −45 mV, tau_m 20 ms, tau_s 5 ms, refractory 2.2 ms, w_syn 0.275 mV,
delay 1.8 ms. `ModelRules` keeps integration choices separate from those constants.

| configuration | clear g | freeze g in refractory | external target | disable target refractory |
| --- | --- | --- | --- | --- |
| M0_current | no | no | g | no |
| M1_clear_g | yes | no | g | no |
| M2_freeze_g | no | yes | g | no |
| M3_stim_v | no | no | v | no |
| M4_stim_norefrac | no | no | g | yes |
| M5_shiu | yes | yes | v | yes |

F1 is one neuron with no connections. F2 has reciprocal excitatory edges of **2300
quanta = 39.53125 mV**, with 1.8-ms delays. With an M0 force event at 5 ms it is still
spiking in the final 50 ms of a 1-s run; reducing each edge to 2250 quanta stops the
activity before that window. This is an empirical near-boundary choice, not a proof
of an exact critical weight or infinite persistence. Both are below the isolated
single-pulse threshold of about 2586 quanta; accumulated drive supports the loop.

F3 was built with `embed_h0.find_embeddings(max_solutions=50)`, `best_embedding` and
`h0_run.isolated_topology`, with the existing Profile 2 policy and parasitic edges
zeroed. Loading the real MCNS cache succeeded through `--mcns-dir`. The selected
15 bodyIds exactly match [the published H0 circuit](../h0_report.md):
10643, 10864, 10881, 11127, 11429, 14160, 14210, 26764, 32461, 37111, 512079,
520816, 523040, 523983, 555296. Topology hash, edge counts/quanta, roles and search
statistics are in the JSON. F3's protocol target is a1's driver, bodyId 32461.
No full-graph topology or simulator was constructed. Persistent F3 activity can be
the intended latch memory; it is not itself evidence of runaway.

## Protocols and metric definitions

There are exactly 18 primary conditions (6 configurations × 3 fixtures), each starting
fresh at rest for every protocol. Replicas share topology but have independent state;
each replica has at most 15 neurons. All main-protocol events use 100,000 quanta.

- **P_A:** one event at 5 ms; observe [0,100 ms). Drive stops at 5.1 ms, immediately
  after delivery, so residual g is measured before the event's first integration step.
- **P_B:** 2-Hz periodic drive for 300 ms, then 300 ms off. Across 50 evenly spaced
  phases `(b+0.5)/50`, 30 events occur: the ensemble event rate is exactly 2 Hz, even
  though individual replicas receive zero or one event. This avoids treating one
  forced event in 300 ms as an observed 2-Hz trial.
- **P_C:** the same P_B streams, plus a probe to the same target at 350 ms, 50 ms
  after drive stops. Compare each replica with its paired no-probe P_B control.
- **P_D:** independently calibrate integer event amplitude for each configuration and
  fixture at 2, 10 and 50 Hz event rates over 1 s, then measure 300 ms off. Ten phases
  provide mean firing-rate resolution 0.1 Hz. Bisection over 0..100,000 quanta retains
  the closest sampled rate, preferring lower amplitude on ties; matching means within
  5% of the target. Recurrence need not be monotonic, so a failed search does not prove
  that no matching amplitude exists. Also replay F1's matched amplitude into F2/F3:
  this matches isolated-neuron input conversion, while allowing recurrent firing to differ.

Spike counts are mean total circuit spikes per replica; mean rates divide by neuron
count and the entire observation duration (100, 600 or 1300 ms). Target drive rates
use only the driven neuron and the drive interval. Residual g is sampled exactly at
the end of the drive, before any off steps. Tables show its maximum absolute value
over neurons/replicas; the JSON also contains signed per-neuron means. Persistence
is the time of the last observed circuit spike after drive stops; `—` means none.
An asterisk marks activity in the last 50 ms: persistence is observation-limited.

The probe response is an extra target spike within [350,380 ms) absent at that step
in the no-probe trace. This detects an effect of the probe, but in an already active
loop it may be a phase shift rather than an additional spike count. The JSON retains
each replica's latency and the raw first-spike latency as well as the response fraction.

## Results

The result inventory is [`model_comparison.json`](model_comparison.json). All tables
below were derived from it with Python's `json` module; values are rounded only for
display. The JSON contains unrounded per-condition/protocol measurements and all
calibration trials.

### P_A: single event, 100 ms

| fixture | configuration | spikes | mean Hz/neuron | max abs(g) at stop mV | last off spike ms |
| --- | --- | --- | --- | --- | --- |
| F1_single | M0_current | 7 | 70 | 1718.75 | 17.6 |
| F1_single | M1_clear_g | 1 | 10 | 1718.75 | 0 |
| F1_single | M2_freeze_g | 41 | 410 | 1718.75 | 93.6* |
| F1_single | M3_stim_v | 1 | 10 | 0 | 0 |
| F1_single | M4_stim_norefrac | 48 | 480 | 1718.75 | 20 |
| F1_single | M5_shiu | 1 | 10 | 0 | 0 |
| F2_pair | M0_current | 29 | 145 | 1718.75 | 91.5* |
| F2_pair | M1_clear_g | 1 | 5 | 1718.75 | 0 |
| F2_pair | M2_freeze_g | 63 | 315 | 1718.75 | 93* |
| F2_pair | M3_stim_v | 1 | 5 | 0 | 0 |
| F2_pair | M4_stim_norefrac | 105 | 525 | 1718.75 | 94.2* |
| F2_pair | M5_shiu | 1 | 5 | 0 | 0 |
| F3_h0 | M0_current | 49 | 32.67 | 1718.75 | 94.8* |
| F3_h0 | M1_clear_g | 18 | 12 | 1718.75 | 90.1* |
| F3_h0 | M2_freeze_g | 96 | 64 | 1718.75 | 93* |
| F3_h0 | M3_stim_v | 36 | 24 | 0 | 93.1* |
| F3_h0 | M4_stim_norefrac | 103 | 68.67 | 1718.75 | 38.5 |
| F3_h0 | M5_shiu | 18 | 12 | 0 | 90.1* |

Freezing g alone retains enough drive that the F1 count is still observation-limited
at 100 ms (41 spikes). Clearing g leaves one spike; eliminating refractory without
clearing g produces 48. V stimulation leaves no synaptic reservoir on F1.

### P_B: 2-Hz drive, 300 ms on + 300 ms off

| fixture | configuration | spikes | mean Hz/neuron | target drive Hz | max abs(g) at stop mV | last off spike ms |
| --- | --- | --- | --- | --- | --- | --- |
| F1_single | M0_current | 4.2 | 7 | 13.67 | 645.07 | 12.7 |
| F1_single | M1_clear_g | 0.6 | 1 | 2 | 0 | — |
| F1_single | M2_freeze_g | 28.8 | 48 | 78.53 | 1618.66 | 113.8 |
| F1_single | M3_stim_v | 0.6 | 1 | 2 | 0 | — |
| F1_single | M4_stim_norefrac | 28.8 | 48 | 94.53 | 645.07 | 15.1 |
| F1_single | M5_shiu | 0.6 | 1 | 2 | 0 | — |
| F2_pair | M0_current | 48.86 | 40.72 | 40 | 645.07 | 299.7* |
| F2_pair | M1_clear_g | 0.6 | 0.5 | 2 | 21.7 | — |
| F2_pair | M2_freeze_g | 145.94 | 121.62 | 107.13 | 1618.66 | 299.9* |
| F2_pair | M3_stim_v | 0.6 | 0.5 | 2 | 21.7 | — |
| F2_pair | M4_stim_norefrac | 149 | 124.17 | 179.67 | 682.3 | 299.9* |
| F2_pair | M5_shiu | 0.6 | 0.5 | 2 | 21.7 | — |
| F3_h0 | M0_current | 121.74 | 13.53 | 74.53 | 645.07 | 299.9* |
| F3_h0 | M1_clear_g | 51.24 | 5.69 | 29.33 | 62.24 | 299.8* |
| F3_h0 | M2_freeze_g | 192.6 | 21.4 | 108.07 | 1618.66 | 299.9* |
| F3_h0 | M3_stim_v | 112.36 | 12.48 | 60.4 | 102.13 | 299.7* |
| F3_h0 | M4_stim_norefrac | 61.8 | 6.87 | 111.8 | 887.26 | 33.6 |
| F3_h0 | M5_shiu | 51.24 | 5.69 | 29.33 | 62.24 | 299.8* |

### Released stimulus amplitude control

Shiu's external stimulus is **68.75 mV = 4000 quanta**, 25 times smaller than the
project's force event, and enters V. The main comparison holds amplitude fixed;
this additional P_B measurement applies 68.75 mV to every configuration. On F1,
all six configurations then produce 2-Hz drive firing: the large force amplitude
is essential to the isolated M0 bursting observed above. M5's 68.75-mV V-input row
is the released-stimulus comparison; M5 at 1718.75 mV isolates rules at project amplitude.

| fixture | configuration | 68.75-mV target drive Hz | spikes in 600 ms | last off spike ms |
| --- | --- | --- | --- | --- |
| F1_single | M0_current | 2 | 0.6 | — |
| F1_single | M1_clear_g | 2 | 0.6 | — |
| F1_single | M2_freeze_g | 2 | 0.6 | — |
| F1_single | M3_stim_v | 2 | 0.6 | — |
| F1_single | M4_stim_norefrac | 2 | 0.6 | — |
| F1_single | M5_shiu | 2 | 0.6 | — |
| F2_pair | M0_current | 2 | 0.6 | — |
| F2_pair | M1_clear_g | 2 | 0.6 | — |
| F2_pair | M2_freeze_g | 2 | 0.6 | — |
| F2_pair | M3_stim_v | 2 | 0.6 | — |
| F2_pair | M4_stim_norefrac | 2 | 0.6 | — |
| F2_pair | M5_shiu | 2 | 0.6 | — |
| F3_h0 | M0_current | 60.53 | 112.5 | 299.8* |
| F3_h0 | M1_clear_g | 28.67 | 50.88 | 299.8* |
| F3_h0 | M2_freeze_g | 83.67 | 123 | 299.9* |
| F3_h0 | M3_stim_v | 60.4 | 112.36 | 299.7* |
| F3_h0 | M4_stim_norefrac | 44.47 | 36 | 62.1 |
| F3_h0 | M5_shiu | 29.33 | 51.24 | 299.8* |

### P_C: probe responsiveness

Residual g at drive stop equals P_B's value, since the probe is later. Counts/rates
include the probe and all 600 ms; the latency averages only responding replicas.

| fixture | configuration | spikes | mean Hz/neuron | probe response fraction | latency ms | last off spike ms |
| --- | --- | --- | --- | --- | --- | --- |
| F1_single | M0_current | 11.2 | 18.67 | 1 | 0.1 | 67.7 |
| F1_single | M1_clear_g | 1.6 | 2.67 | 1 | 0.1 | 50.1 |
| F1_single | M2_freeze_g | 71.06 | 118.43 | 0.88 | 0.1 | 177.8 |
| F1_single | M3_stim_v | 1.6 | 2.67 | 1 | 0.1 | 50.1 |
| F1_single | M4_stim_norefrac | 76.8 | 128 | 1 | 0.1 | 70.1 |
| F1_single | M5_shiu | 1.6 | 2.67 | 1 | 0.1 | 50.1 |
| F2_pair | M0_current | 80.94 | 67.45 | 1 | 0.16 | 299.9* |
| F2_pair | M1_clear_g | 1.6 | 1.33 | 1 | 0.1 | 50.1 |
| F2_pair | M2_freeze_g | 208.88 | 174.07 | 0.68 | 0.29 | 299.9* |
| F2_pair | M3_stim_v | 1.6 | 1.33 | 1 | 0.1 | 50.1 |
| F2_pair | M4_stim_norefrac | 257.7 | 214.75 | 1 | 0.1 | 299.6* |
| F2_pair | M5_shiu | 1.6 | 1.33 | 1 | 0.1 | 50.1 |
| F3_h0 | M0_current | 171.46 | 19.05 | 1 | 0.47 | 299.8* |
| F3_h0 | M1_clear_g | 71.1 | 7.9 | 0.98 | 0.26 | 299.2* |
| F3_h0 | M2_freeze_g | 283.18 | 31.46 | 0.52 | 0.1 | 299.9* |
| F3_h0 | M3_stim_v | 154.12 | 17.12 | 0.72 | 0.1 | 299.9* |
| F3_h0 | M4_stim_norefrac | 164.46 | 18.27 | 1 | 0.13 | 90.8 |
| F3_h0 | M5_shiu | 71.08 | 7.9 | 0.98 | 0.1 | 299.7* |

### P_D: firing-matched amplitude calibration

All 18 F1 calibrations match exactly, including both g-input and V-input modes.
Across all fixtures **33/54** searches match; none of F3's local firing rates match.
The remaining rows are explicitly unsuccessful matches, not fair comparisons at
equal local firing. For a recurrent latch, crossing the ignition threshold can produce
a rate much higher than the intended 2 or 10 Hz. Nonmonotonic responses also limit
bisection; the JSON records every sampled amplitude and residual error.

Each cell below is `quanta / achieved Hz`; a dagger marks error greater than 5%.
The event rate equals the requested firing-rate column, so differences within a column
are due to calibrated amplitude, rules and recurrence rather than event frequency.

| fixture | configuration | aim 2 Hz: q / Hz | aim 10 Hz: q / Hz | aim 50 Hz: q / Hz |
| --- | --- | --- | --- | --- |
| F1_single | M0_current | 2587 / 2 | 2986 / 10 | 3701 / 50 |
| F1_single | M1_clear_g | 2587 / 2 | 3000 / 10 | 10121 / 50 |
| F1_single | M2_freeze_g | 2587 / 2 | 2979 / 10 | 2761 / 50 |
| F1_single | M3_stim_v | 410 / 2 | 410 / 10 | 410 / 50 |
| F1_single | M4_stim_norefrac | 2587 / 2 | 2981 / 10 | 2955 / 50 |
| F1_single | M5_shiu | 410 / 2 | 410 / 10 | 410 / 50 |
| F2_pair | M0_current | 2587 / 2 | 2986 / 10 | 1561 / 16† |
| F2_pair | M1_clear_g | 2587 / 2 | 3000 / 10 | 10121 / 50 |
| F2_pair | M2_freeze_g | 2587 / 2 | 2979 / 10 | 1561 / 16† |
| F2_pair | M3_stim_v | 410 / 2 | 410 / 10 | 410 / 50 |
| F2_pair | M4_stim_norefrac | 2587 / 2 | 2981 / 10 | 1561 / 16† |
| F2_pair | M5_shiu | 410 / 2 | 410 / 10 | 410 / 50 |
| F3_h0 | M0_current | 1561 / 0† | 1561 / 0† | 780 / 0† |
| F3_h0 | M1_clear_g | 1561 / 0† | 1561 / 0† | 1412 / 83.3† |
| F3_h0 | M2_freeze_g | 1561 / 0† | 1561 / 0† | 780 / 0† |
| F3_h0 | M3_stim_v | 389 / 0† | 389 / 0† | 194 / 0† |
| F3_h0 | M4_stim_norefrac | 1561 / 0† | 1561 / 0† | 1412 / 96.6† |
| F3_h0 | M5_shiu | 389 / 0† | 389 / 0† | 259 / 87.5† |

For F1, the 10-Hz match requires 2986 quanta (51.32 mV) into g under M0 and
410 quanta (7.05 mV) into V under M5. At 50 Hz, M1 uses 10,121 quanta,
M0 3701 quanta and M5 410. Clearing g prevents inter-event accumulation, and
the exact drive-window count also requires late events to fire before the window
ends. These are protocol-specific calibrations, not single-pulse thresholds; finite
window latency contributes to the larger amplitudes, especially for M1 at 50 Hz.
Equal firing does not imply equal residual g or persistence; the following table
shows the target-10-Hz measurements, including unsuccessful recurrent matches.
All three target rates' full metrics are in the JSON under `P_D[].measurement`.

| fixture | configuration | achieved target Hz | spikes / 1.3 s | mean Hz/neuron | max abs(g) at stop mV | last off spike ms |
| --- | --- | --- | --- | --- | --- | --- |
| F1_single | M0_current | 10 | 10 | 7.69 | 19.26 | — |
| F1_single | M1_clear_g | 10 | 10 | 7.69 | 0 | — |
| F1_single | M2_freeze_g | 10 | 10 | 7.69 | 19.22 | — |
| F1_single | M3_stim_v | 10 | 10 | 7.69 | 0 | — |
| F1_single | M4_stim_norefrac | 10 | 10 | 7.69 | 19.23 | — |
| F1_single | M5_shiu | 10 | 10 | 7.69 | 0 | — |
| F2_pair | M0_current | 10 | 10 | 3.85 | 19.26 | — |
| F2_pair | M1_clear_g | 10 | 10 | 3.85 | 7.67 | — |
| F2_pair | M2_freeze_g | 10 | 10 | 3.85 | 19.22 | — |
| F2_pair | M3_stim_v | 10 | 10 | 3.85 | 21.7 | — |
| F2_pair | M4_stim_norefrac | 10 | 10 | 3.85 | 19.23 | — |
| F2_pair | M5_shiu | 10 | 10 | 3.85 | 21.7 | — |
| F3_h0 | M0_current | 0 | 0 | 0 | 10.07 | — |
| F3_h0 | M1_clear_g | 0 | 0 | 0 | 10.07 | — |
| F3_h0 | M2_freeze_g | 0 | 0 | 0 | 10.07 | — |
| F3_h0 | M3_stim_v | 0 | 0 | 0 | 0 | — |
| F3_h0 | M4_stim_norefrac | 0 | 0 | 0 | 10.07 | — |
| F3_h0 | M5_shiu | 0 | 0 | 0 | 0 | — |

The alternative matched-input control below uses each configuration's **exact F1
10-Hz calibration**, replayed into each fixture without retuning. Thus isolated
input firing is matched even when recurrent local firing cannot be. This separates
the input conversion from the effect of recurrence. It does not claim to match the
actual recurrent target rate. The 2- and 50-Hz controls are also retained in the JSON.

| fixture | configuration | F1-matched q | actual target Hz | spikes / 1.3 s | max abs(g) at stop mV | last off spike ms |
| --- | --- | --- | --- | --- | --- | --- |
| F1_single | M0_current | 2986 | 10 | 10 | 19.26 | — |
| F1_single | M1_clear_g | 3000 | 10 | 10 | 0 | — |
| F1_single | M2_freeze_g | 2979 | 10 | 10 | 19.22 | — |
| F1_single | M3_stim_v | 410 | 10 | 10 | 0 | — |
| F1_single | M4_stim_norefrac | 2981 | 10 | 10 | 19.23 | — |
| F1_single | M5_shiu | 410 | 10 | 10 | 0 | — |
| F2_pair | M0_current | 2986 | 10 | 10 | 19.26 | — |
| F2_pair | M1_clear_g | 3000 | 10 | 10 | 7.67 | — |
| F2_pair | M2_freeze_g | 2979 | 10 | 10 | 19.22 | — |
| F2_pair | M3_stim_v | 410 | 10 | 10 | 21.7 | — |
| F2_pair | M4_stim_norefrac | 2981 | 10 | 10 | 19.23 | — |
| F2_pair | M5_shiu | 410 | 10 | 10 | 21.7 | — |
| F3_h0 | M0_current | 2986 | 206.5 | 542.9 | 117.49 | 299.9* |
| F3_h0 | M1_clear_g | 3000 | 91.5 | 239.3 | 61 | 299.5* |
| F3_h0 | M2_freeze_g | 2979 | 283 | 773 | 254.7 | 299* |
| F3_h0 | M3_stim_v | 410 | 206.4 | 542.5 | 104.47 | 299.5* |
| F3_h0 | M4_stim_norefrac | 2981 | 130 | 315 | 19.03 | — |
| F3_h0 | M5_shiu | 410 | 95.5 | 242.4 | 56.31 | 299.7* |

### P_D: matched-amplitude probe responsiveness

Each P_D measurement also has a paired trial with a probe at **1050 ms**, 50 ms
after its 1-s drive stops, using that trial's calibrated amplitude. The same 30-ms
extra-spike criterion applies. The target-10-Hz results are below; all three target
rates, complete probe-trial metrics and per-replica latencies are in the JSON under
`measurement.probe_trial` and `F1_matched_input_measurement.probe_trial`.
Local calibrations that failed to match remain failed here; a zero response at their
subthreshold amplitude does not demonstrate loss of responsiveness at matched firing.

| fixture | configuration | local-calibration probe fraction | latency ms | F1-matched probe fraction | latency ms |
| --- | --- | --- | --- | --- | --- |
| F1_single | M0_current | 1 | 4.88 | 1 | 4.88 |
| F1_single | M1_clear_g | 1 | 4.9 | 1 | 4.9 |
| F1_single | M2_freeze_g | 1 | 4.85 | 1 | 4.85 |
| F1_single | M3_stim_v | 1 | 0.1 | 1 | 0.1 |
| F1_single | M4_stim_norefrac | 1 | 4.86 | 1 | 4.86 |
| F1_single | M5_shiu | 1 | 0.1 | 1 | 0.1 |
| F2_pair | M0_current | 1 | 4.88 | 1 | 4.88 |
| F2_pair | M1_clear_g | 1 | 4.9 | 1 | 4.9 |
| F2_pair | M2_freeze_g | 1 | 4.85 | 1 | 4.85 |
| F2_pair | M3_stim_v | 1 | 0.1 | 1 | 0.1 |
| F2_pair | M4_stim_norefrac | 1 | 4.86 | 1 | 4.86 |
| F2_pair | M5_shiu | 1 | 0.1 | 1 | 0.1 |
| F3_h0 | M0_current | 0 | — | 1 | 2.1 |
| F3_h0 | M1_clear_g | 0 | — | 1 | 4.18 |
| F3_h0 | M2_freeze_g | 0 | — | 0.3 | 1.53 |
| F3_h0 | M3_stim_v | 0.1 | 0.1 | 0.5 | 0.1 |
| F3_h0 | M4_stim_norefrac | 0 | — | 1 | 4.8 |
| F3_h0 | M5_shiu | 0.1 | 0.1 | 1 | 0.1 |

### H0: all four AND transactions

These are separate functional checks with the existing transaction stimulus:
3621 quanta delivered to each selected rail driver at 5 ms, 150 ms observation.
They are not force-event tests and are not recalibrated P_D transactions. The local
runner uses the same decoder as `h0_run.run_transaction`; the synthetic test checks
its default results against that unchanged function. Passing requires correct dual-rail
AND decode, completion, `reset_ok`, and completion latency <60 ms exactly as the
existing H0 test does. JSON also retains y1/y0/completion flags and spikes per role.

| configuration | a,b | correct decode + completion | reset_ok | completion ms | passes existing checks |
| --- | --- | --- | --- | --- | --- |
| M0_current | 0,0 | True | True | 22.8 | True |
| M0_current | 0,1 | True | True | 36.9 | True |
| M0_current | 1,0 | True | True | 36.9 | True |
| M0_current | 1,1 | True | True | 56.3 | True |
| M1_clear_g | 0,0 | True | True | 25.3 | True |
| M1_clear_g | 0,1 | False | False | — | False |
| M1_clear_g | 1,0 | False | False | — | False |
| M1_clear_g | 1,1 | False | False | — | False |
| M2_freeze_g | 0,0 | True | True | 22 | True |
| M2_freeze_g | 0,1 | True | True | 31.4 | True |
| M2_freeze_g | 1,0 | True | True | 31.4 | True |
| M2_freeze_g | 1,1 | True | True | 41.1 | True |
| M3_stim_v | 0,0 | True | True | 21.9 | True |
| M3_stim_v | 0,1 | True | True | 36.7 | True |
| M3_stim_v | 1,0 | True | True | 36.7 | True |
| M3_stim_v | 1,1 | True | True | 56.1 | True |
| M4_stim_norefrac | 0,0 | True | True | 21.9 | True |
| M4_stim_norefrac | 0,1 | True | True | 28.3 | True |
| M4_stim_norefrac | 1,0 | True | True | 28.3 | True |
| M4_stim_norefrac | 1,1 | True | True | 33.2 | True |
| M5_shiu | 0,0 | True | True | 21.9 | True |
| M5_shiu | 0,1 | False | False | — | False |
| M5_shiu | 1,0 | False | False | — | False |
| M5_shiu | 1,1 | False | False | — | False |

Clearing g alone breaks the three transactions that M5 loses. In these fixtures,
that identifies a sufficient cause of the H0 compatibility loss; it does not establish
all interactions in a full H0 factorial. Freezing g alone, V stimulation alone and
removing target refractory alone preserve all four existing checks, with changed
latencies. M0's 22.8/36.9/36.9/56.3-ms latencies reproduce the published baseline.

With the F1-matched 10-Hz input replayed into F3, its driver fires at 206.5 Hz under
M0 and 95.5 Hz under M5. The existing gate weights were designed for M0's roughly
213-Hz latch train. This is consistent with reduced temporal summation at the gates;
these measurements do not test a circuit redesigned for the slower train.

## Independent equation parity and regression checks

`tests/test_shiu_parity.py` uses the quoted released equation text, strict `v > v_th`,
reset `v = v_rst; w = 0; g = 0 * mV`, and per-neuron `rfc`. Both derivatives have
`(unless refractory)`. A fixed replay of external V arrivals runs in the synapses
slot, matching `PoissonInput` timing without comparing unrelated RNG streams.
Its weight is the released 68.75 mV; only input targets have rfc=0.

The comparison source is Shiu et al. (2024)'s
[released model.py](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py).

Over **250 ms**, the single neuron produces 24 spikes in both implementations;
the recurrent pair produces 25 source and six partner spikes. **Maximum spike-time
difference is zero dt** in all three traces, tighter than the required one-dt limit.
The original project-rule Brian2 test is unchanged. NumPy Brian2 code generation
was used; this is deterministic equation/schedule parity, not a Poisson RNG parity claim.

A SHA-256 captured by executing ref64 before editing covers final V/g/r, the integer
delay ring, spike records and recorded V/g trajectories on a 15-neuron, two-replica
fixture. The new default reproduces the digest exactly:
`a526bf81bba989b095073c8d5a9307c6ca21e25e02eac9c2f37c88aee5f806b6` (921 spikes).
Each individual non-default rule is tested for explicit refusal by TorchSim and
FastSim. Snapshot testing covers registered refractory exemptions and rule mismatch.

## F1 factorial appendix: single force event, 100 ms

This is the optional 16-way appendix on F1 only, not additional primary fixtures.
All events are 100,000 quanta. Counts are within 100 ms; the freeze-only 41-spike
response is still active near the observation end. When refractory is disabled on
the only neuron, freezing g has no effect because that neuron is never held.

| clear g | freeze g | target | disable refractory | spikes |
| --- | --- | --- | --- | --- |
| False | False | g | False | 7 |
| False | False | g | True | 48 |
| False | False | v | False | 1 |
| False | False | v | True | 1 |
| False | True | g | False | 41 |
| False | True | g | True | 48 |
| False | True | v | False | 1 |
| False | True | v | True | 1 |
| True | False | g | False | 1 |
| True | False | g | True | 1 |
| True | False | v | False | 1 |
| True | False | v | True | 1 |
| True | True | g | False | 1 |
| True | True | g | True | 1 |
| True | True | v | False | 1 |
| True | True | v | True | 1 |

## Reproduction

From the worktree root, using the project's Python environment:

```bash
PYTHONPATH="$PWD" python -m drosophilos.bench.model_compare \
  --mcns-dir /Users/jeremiahgassensmith/programming/drosophilos/data/mcns \
  --out docs/stage_h/model_comparison.json

# Development-only alternative; reuses synthetic_mcns() from the existing test.
PYTHONPATH="$PWD" python -m drosophilos.bench.model_compare --synthetic \
  --out /path/to/writable/synthetic_model_comparison.json

PYTHONPATH="$PWD" python -m pytest tests/test_shiu_parity.py tests/test_model_compare.py -q
PYTHONPATH="$PWD" python -m pytest tests/test_embed_h0.py tests/test_brian2_parity.py tests/test_dale_refractory.py -q
PYTHONPATH="$PWD" python -m pytest -q -x -m 'not slow'
```

If the data path cannot be read, the command records all six real F3 conditions as
`not_run` with the exception and completes F1/F2; it does not download, copy or
symlink MCNS. This run did access the existing real cache successfully. The bounded
campaign uses no background surround, fitted biological responses or whole-brain
stability criterion. Any primary-model decision should preserve these limits and
separately repeat the relevant whole-brain experiments under the chosen protocol.
