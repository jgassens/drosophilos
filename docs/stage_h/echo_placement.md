# Placement scored by output exposure: LOCAL ECHO

## Question

Can selecting hosts with fewer anatomical outputs or weaker reciprocal connections
make the configuration-F flip-flop adder compute inside the simulated whole brain?
A host is a real MCNS neuron assigned a designed circuit neuron. A **local echo**
is activity leaving hosts through real output edges, recruiting surrounding neurons,
and returning through their real inputs to hosts.

## Method

The adder has 740 designed neurons and 1,516 designed edge pairs, with autapse
(self-connection) latches. Every live run uses condition A: every missing designed
edge is restored as a Profile 3 artificial edge. Thus **carried** counts designed
edges supported by real wiring, not the number of designed edges available to
compute. Parasitic host-to-host edges are zeroed. Unplaced circuit neurons are
synthetic. All placement settings pass their isolated condition-A checks, 50/50.

The placement feature landed at code commit `0b7b166`. Its per-candidate penalty is

```text
isolation_in * log1p(in_syn)
+ isolation_out * log1p(out_syn)
+ echo_weight * log1p(self_echo)
self_echo(h) = sum over X != h of syn(h -> X) * syn(X -> h)
```

Here `syn` is the anatomical synapse count, and `in_syn` and `out_syn` are totals
over all anatomical partners while ranking candidates. The final objective uses
outside-host input/output exposure; self-echo still includes other hosts.
The search trades carried edges against the summed penalty.
The baseline `w0` sets every penalty weight to zero; `out1/out2/out4/out8` set only
`isolation_out` to the named value; `echo1/echo2/echo4` set only `echo_weight`.
All use region `all`. **Host echo paths** (`host_echo_paths`) sum synapse-count
products over two-step paths from any host through a non-host neuron to any host,
including a different host. **Echo neurons** (`host_echo_neurons`) count the
non-host intermediates on those paths. These are unsigned structural measures.
Self-echo scores returns to the same candidate neuron, not the full host set.

The **published** surround (`pub`) uses Shiu excitatory and inhibitory gains of
1 and 1. The **calibrated** surround (`cal`) uses excitatory gain 0.25 and
inhibitory gain 2, uncapped. These gains apply to anatomical edges with at least
one non-host endpoint; host-to-host and designed/artificial edges are excluded.
Both surrounds are silent: there is no external sensory drive. Live trials use
50 fresh additions, seed 0, and a 1,200 ms maximum per addition; failures may end
earlier. The selected calibration and its responsiveness limits are described
in [surround calibration](surround_calibration.md).

**OUT** measures anatomical deliveries from hosts to real non-host neurons;
**IN** measures deliveries from real non-host neurons into hosts, split into
excitatory and inhibitory components. A delivery count counts edge deliveries,
not distinct spikes or anatomical synapses. **Absolute quanta** sum delivered
absolute embedded weights (16 quanta per unit synapse weight), so inhibitory
weights contribute positively. IN/OUT is a traffic ratio, not a signed net current
or a causal estimate of which outputs produced each return.

The machine-readable inventory is [echo_placement.json](echo_placement.json).
It contains source filenames, jobs, outcomes, exact traffic totals, stored metrics,
and the derived analysis. Tables below are calculated from that JSON; percentages
and correlations are rounded for display. Source data remain outside the repository.
The baseline uses `all_w0_rescore.json` (G2 4048888), whose host assignments were
verified identical to the original `all_w0`; its recorded source commit is retained.
Only the supplied placement, live-trace and control files are read.

Placement commands (defaults select configuration F and autapse latches):

```sh
for w in 1 2 4 8; do
  python -m drosophilos.bench.h_stage_place --region all \
    --isolation-in 0 --isolation-out "$w" --echo-weight 0 \
    --cases 50 --conditions AB --restarts 4 --time-limit 600 --seed 0 \
    --out "data/stage_h/place/all_out${w}.json" \
    --mapping-out "data/stage_h/place/all_out${w}_mapping.json"
done
for w in 1 2 4; do
  python -m drosophilos.bench.h_stage_place --region all \
    --isolation-in 0 --isolation-out 0 --echo-weight "$w" \
    --cases 50 --conditions AB --restarts 4 --time-limit 600 --seed 0 \
    --out "data/stage_h/place/all_echo${w}.json" \
    --mapping-out "data/stage_h/place/all_echo${w}_mapping.json"
done
```

Live commands, run on G2; the following reproduces both surrounds:

```sh
for tag in w0 out1 out8 echo1; do
  python -m drosophilos.bench.h1_fullgraph \
    --mapping "data/stage_h/place/all_${tag}_mapping.json" \
    --n-cases 50 --seed 0 --surround silent --max-ms 1200 --echo-trace \
    --out "data/stage_h/fullgraph/all_${tag}_echotrace_pub_silent.json"
  python -m drosophilos.bench.h1_fullgraph \
    --mapping "data/stage_h/place/all_${tag}_mapping.json" \
    --n-cases 50 --seed 0 --surround silent --max-ms 1200 --echo-trace \
    --surround-scale-exc 0.25 --surround-scale-inh 2 \
    --out "data/stage_h/fullgraph/all_${tag}_echotrace_cal_silent.json"
done
```

| Placement | Published G2 job | Calibrated G2 job |
| --- | --- | --- |
| w0 | 4048871 | 4048872 |
| out1 | 4048877 | 4048878 |
| out8 | 4048873 | 4048874 |
| echo1 | 4048875 | 4048876 |

Controls use the calibrated command below. For the sealed condition add
`--inputs-zeroed`; both conditions include `--outputs-zeroed`. Earlier w0 controls
use their recorded filenames and case counts in the inventory.

```sh
for tag in echo1 out8; do
  python -m drosophilos.bench.h1_fullgraph \
    --mapping "data/stage_h/place/all_${tag}_mapping.json" \
    --n-cases 10 --seed 0 --surround silent --max-ms 1200 \
    --surround-scale-exc 0.25 --surround-scale-inh 2 --outputs-zeroed \
    --out "data/stage_h/fullgraph/all_${tag}_cal_silent_outzero.json"
  python -m drosophilos.bench.h1_fullgraph \
    --mapping "data/stage_h/place/all_${tag}_mapping.json" \
    --n-cases 10 --seed 0 --surround silent --max-ms 1200 \
    --surround-scale-exc 0.25 --surround-scale-inh 2 \
    --outputs-zeroed --inputs-zeroed \
    --out "data/stage_h/fullgraph/all_${tag}_cal_silent_sealed.json"
done
```

Rebuild the inventory locally from the repository root without running simulations:

```sh
python -m drosophilos.bench.h_echo_report \
  --place-dir data/stage_h/place \
  --fullgraph-dir data/stage_h/fullgraph \
  --out docs/stage_h/echo_placement.json
```

## Trade-off table

Each live cell gives correct/additions, then whole-window IN/OUT by absolute
quanta / count. “Not measured” means no supplied live trial was scheduled for
that setting.

| Setting | Carried / designed | Host echo paths | Echo neurons | Published: correct/n; weight / count return | Calibrated: correct/n; weight / count return |
| --- | --- | --- | --- | --- | --- |
| w0 | 809/1516 | 1,099,380,131 | 56,543 | 0/50; 70.38% / 49.00% | 0/50; 12.45% / 4.91% |
| out1 | 646/1516 | 406,703,946 | 37,424 | 0/50; 82.52% / 57.13% | 0/50; 5.44% / 3.66% |
| out2 | 583/1516 | 376,731,713 | 35,609 | not measured | not measured |
| out4 | 563/1516 | 304,813,484 | 37,345 | not measured | not measured |
| out8 | 542/1516 | 285,033,288 | 36,076 | 0/50; 83.41% / 63.42% | 0/50; 3.37% / 1.87% |
| echo1 | 554/1516 | 290,789,986 | 36,644 | 0/50; 62.38% / 52.72% | 0/50; 1.49% / 0.72% |
| echo2 | 549/1516 | 372,123,843 | 41,113 | not measured | not measured |
| echo4 | 531/1516 | 307,883,824 | 42,204 | not measured | not measured |

The output penalty reduces structural echo by giving up real wiring: from w0 to
out8, carried edges fall from 809 to 542 while host echo paths fall from
1,099,380,131 to 285,033,288 (about 1,099 million to 285 million). This search did
not find echo-poor wiring that preserves baseline carry and restores computation.
Every finished live trial, including published w0 (G2 4048871), is 0/50.
Published echo1 (G2 4048875) is also 0/50, with all 50 additions incomplete.

The strongest output penalty gives fewer host echo paths than any tested
self-echo penalty: echo1 has 290,789,986, echo2 has 372,123,843, and echo4 has
307,883,824. Increasing the self-echo penalty does not monotonically reduce
host-set echo. This is not a general superiority claim at equal penalty weights:
echo1 has fewer paths than out1's 406,703,946.

Outcome counts below are mutually exclusive. Every finished run has zero correct
and zero wrong accepted answers; failure often occurs before an answer is accepted.

| Placement | Surround | Fault | No accept | Timeout | Incomplete |
| --- | --- | --- | --- | --- | --- |
| w0 | pub | 22 | 2 | 26 | 0 |
| w0 | cal | 1 | 49 | 0 | 0 |
| out1 | pub | 0 | 3 | 37 | 10 |
| out1 | cal | 2 | 48 | 0 | 0 |
| out8 | pub | 0 | 0 | 49 | 1 |
| out8 | cal | 0 | 30 | 20 | 0 |
| echo1 | pub | 0 | 0 | 0 | 50 |
| echo1 | cal | 0 | 50 | 0 | 0 |

## Controls

A **sealed** boundary zeros both incoming and outgoing anatomical host/surround
weights. **Outputs-zeroed** removes only outgoing weights, leaving incoming
connections available. In the silent surround this tests whether hosts can compute
when prevented from exciting their surroundings.

| Sealed | Outputs-zeroed | Reading when live fails |
| --- | --- | --- |
| Pass | Pass | Echo is the cause. |
| Pass | Fail | Incoming activity, not echo, explains failure. |
| Fail | Either | The placement itself fails the sealed check. |

| Placement | Condition | Correct/n | Outcomes | G2 job |
| --- | --- | --- | --- | --- |
| w0 | sealed | 5/5 | 5 correct | earlier control; not supplied |
| w0 | outputs-zeroed | 10/10 | 10 correct | earlier control; not supplied |
| echo1 | sealed | 10/10 | 10 correct | 4049043 |
| echo1 | outputs-zeroed | 10/10 | 10 correct | 4049037 |
| out8 | sealed | 10/10 | 10 correct | 4049044 |
| out8 | outputs-zeroed | 10/10 | 10 correct | 4049038 |

Echo1 and out8 pass both controls, 10/10 in each condition; earlier w0 controls
also pass. Together with live failures, this identifies circuit-driven local echo,
rather than intrinsically unusable hosts, as the cause under these silent calibrated
conditions. Echo1 fails even with only 1.49% absolute-weight return, of which
94.57% is inhibitory. Out8's corresponding return is 3.37%, 77.35% inhibitory.
Absolute weight does not measure the signed balance or the effect on each host.
The out1 hosts have no supplied boundary controls.

## Lag analysis

For every addition and each unit separately, the script computes Pearson
correlation of OUT[t] and IN[t+lag] for lags from 0 through 50 ms. Each overlapping
slice has its own mean subtracted. Additions with zero variance in either slice,
or fewer than 3 overlapping bins, are skipped at that lag. Correlations are averaged
equally across eligible additions; bins are never concatenated across resets.
The peak is selected from the whole mean curve, with the earliest lag winning
a tie. The JSON records the entire curve and the number used at every lag.

### Mean-subtracted correlation: absolute quanta

| Placement | Surround | r(0 ms) | r(1 ms) | r(2 ms) | r(3 ms) | r(5 ms) | r(10 ms) | r(20 ms) | r(30 ms) | r(50 ms) | Peak ms / r | Additions used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w0 | pub | 0.6254 | 0.3332 | 0.5262 | 0.4461 | 0.5615 | 0.1702 | 0.6287 | -0.2239 | -0.2017 | 9 / 0.6814 | 50 |
| w0 | cal | 0.0210 | 0.0180 | 0.0132 | 0.0074 | 0.0199 | 0.0202 | 0.0153 | 0.0121 | 0.0009 | 15 / 0.0251 | 50 |
| out1 | pub | 0.5113 | 0.5687 | 0.4935 | 0.5798 | 0.5682 | 0.5363 | 0.4094 | 0.2827 | 0.0583 | 3 / 0.5798 | 50 |
| out1 | cal | 0.2005 | 0.2004 | 0.1639 | 0.1692 | 0.1716 | 0.2036 | 0.1543 | 0.1628 | 0.1656 | 10 / 0.2036 | 50 |
| out8 | pub | 0.8413 | 0.8247 | 0.8285 | 0.8262 | 0.8307 | 0.8103 | 0.7716 | 0.6986 | 0.5748 | 0 / 0.8413 | 50 |
| out8 | cal | 0.0407 | 0.0430 | 0.0380 | 0.0303 | 0.0327 | 0.0303 | 0.0317 | 0.0215 | 0.0369 | 1 / 0.0430 | 50 |
| echo1 | pub | 0.6040 | 0.6139 | 0.6062 | 0.6093 | 0.6088 | 0.6129 | 0.5663 | 0.5659 | 0.4926 | 8 / 0.6143 | 50 |
| echo1 | cal | 0.0374 | 0.0271 | 0.0333 | 0.0289 | 0.0262 | 0.0230 | -0.0113 | -0.0125 | -0.0073 | 0 / 0.0374 | 50 |

### Mean-subtracted correlation: counts

| Placement | Surround | r(0 ms) | r(1 ms) | r(2 ms) | r(3 ms) | r(5 ms) | r(10 ms) | r(20 ms) | r(30 ms) | r(50 ms) | Peak ms / r | Additions used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w0 | pub | 0.5263 | 0.4868 | 0.5055 | 0.5237 | 0.5509 | 0.4663 | 0.5612 | 0.1920 | 0.0088 | 9 / 0.5728 | 50 |
| w0 | cal | 0.1197 | 0.0649 | 0.0882 | 0.0988 | 0.0925 | 0.0924 | 0.0553 | 0.0483 | 0.0228 | 0 / 0.1197 | 50 |
| out1 | pub | 0.5985 | 0.6182 | 0.5966 | 0.6228 | 0.6194 | 0.6091 | 0.5255 | 0.4392 | 0.1725 | 3 / 0.6228 | 50 |
| out1 | cal | 0.4735 | 0.4582 | 0.4263 | 0.4410 | 0.4497 | 0.4827 | 0.4157 | 0.4022 | 0.4123 | 10 / 0.4827 | 50 |
| out8 | pub | 0.8566 | 0.8467 | 0.8471 | 0.8501 | 0.8511 | 0.8426 | 0.8039 | 0.7393 | 0.5985 | 0 / 0.8566 | 50 |
| out8 | cal | 0.1430 | 0.1240 | 0.1270 | 0.1215 | 0.1222 | 0.1071 | 0.0913 | 0.0862 | 0.0860 | 0 / 0.1430 | 50 |
| echo1 | pub | 0.6006 | 0.5963 | 0.5980 | 0.5976 | 0.6016 | 0.5987 | 0.5675 | 0.5553 | 0.4826 | 5 / 0.6016 | 50 |
| echo1 | cal | 0.1400 | 0.1200 | 0.1497 | 0.1598 | 0.1213 | 0.1141 | 0.0748 | 0.0449 | 0.0298 | 3 / 0.1598 | 50 |

All displayed lags use 50 additions in each finished run, in both units.
These are descriptive correlations of recurrent activity, not impulse-response
estimates. The calibrated weight correlations are weak; a curve maximum does
not establish a propagation delay.

### Onset and early return

**Onset latency** is the first IN-positive bin minus the first OUT-positive bin
within each addition; additions missing either onset are excluded. Both units give
the same onset summaries here. **Early return** is IN summed over the onset bin
and the following bins through onset+19, divided by OUT summed over that same
interval, then summarized by the median across additions. An interval is truncated
at the addition end if necessary. The adjacent whole-window ratios pool totals
over all observed addition windows; they are not medians. Per-addition whole-window
medians and valid sample counts are also retained in the JSON.

| Placement | Surround | Onset median (min–max), ms | Onsets | Early weight | Whole weight | Early count | Whole count | IN inhibitory weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w0 | pub | 2 (2–3) | 50 | 31.89% | 70.38% | 14.40% | 49.00% | 34.62% |
| w0 | cal | 7 (7–9) | 50 | 2.19% | 12.45% | 1.36% | 4.91% | 55.89% |
| out1 | pub | 2 (2–2) | 50 | 27.45% | 82.52% | 17.54% | 57.13% | 36.36% |
| out1 | cal | 5 (5–7) | 50 | 2.38% | 5.44% | 1.08% | 3.66% | 49.99% |
| out8 | pub | 3 (2–7) | 50 | 23.25% | 83.41% | 12.28% | 63.42% | 50.22% |
| out8 | cal | 11 (5–14) | 50 | 0.93% | 3.37% | 0.60% | 1.87% | 77.35% |
| echo1 | pub | 2 (2–3) | 50 | 18.72% | 62.38% | 13.64% | 52.72% | 45.35% |
| echo1 | cal | 10 (8–11) | 50 | 0.59% | 1.49% | 0.32% | 0.72% | 94.57% |

Echo1 calibrated has a median IN onset 10 ms after OUT and a median early
absolute-weight return of 0.59%, versus 1.49% over the full observed windows.
Out8 calibrated has an 11 ms median onset and 0.93% median early weight return.
Positive onset latencies coexist with raw or centered correlation peaks at short
lags because ongoing OUT and IN overlap for much of the run.

### Stored metrics: record only

The simulator's stored `within_20_ms_fraction` is 1.000 for every finished run
and both units. It counts IN in any bin with **any** OUT in bins [t−19, t],
including the current bin. It does not pair a return with its originating output.
Recomputing that rule from the arrays also gives 1.000. The criterion saturates
because OUT is nearly continuous over the active periods. OUT is not literally
nonzero in every millisecond, as the measured zero-bin counts show:

| Placement | Surround | OUT zero bins / all bins | OUT-positive bins | Stored raw peak: weight / count, ms |
| --- | --- | --- | --- | --- |
| w0 | pub | 300/60000 | 99.50% | 0 / 0 |
| w0 | cal | 345/60000 | 99.42% | 11 / 0 |
| out1 | pub | 301/60000 | 99.50% | 1 / 1 |
| out1 | cal | 320/58470 | 99.45% | 1 / 0 |
| out8 | pub | 300/6026 | 95.02% | 0 / 0 |
| out8 | cal | 568/60000 | 99.05% | 6 / 0 |
| echo1 | pub | 301/60000 | 99.50% | 1 / 0 |
| echo1 | cal | 350/60000 | 99.42% | 0 / 3 |

The stored window is resolved on 1 ms bins. With the simulation's 0.1 ms steps,
an OUT-to-IN separation of 19.0 ms or less is always counted, 20.0 ms or more
is never counted, and only 19.1-19.9 ms depends on where the events fall within
their bins; it is not an exact event-level 20 ms threshold. This
saturated fraction is retained for the record and is not used to infer timing.

The stored `peak_lag_ms` uses an uncentred dot product summed across additions.
Its short peaks reflect sustained co-activity and amplitude, not an isolated
round-trip latency. They are not universally 0–3 ms: calibrated w0's weight peak
is 11 ms and calibrated out8's is 6 ms. The centered curves and onset measurements
above replace that metric for this report's timing description, while neither
alone establishes a causal delay.

## Conclusion

Penalizing output exposure buys lower two-step echo at the cost of carried real
wiring, and it does not restore live computation in any finished trial.
Penalizing same-neuron self-echo also fails to rescue the tested live placement.
Passing sealed and outputs-zeroed controls on echo1 and out8 identifies the
circuit's own interaction with the silent surround as the failure mechanism.
A small return ratio, even one dominated by inhibition, is insufficient protection.

## Limits

These results cover region `all`, the listed penalty settings, the two surround
weight settings, and seed 0. Only w0, out1, out8 and echo1 have scheduled live traces;
all eight trials are finished in this snapshot. The controls
test 10 additions per condition, except the earlier w0 sealed control with 5;
they do not establish full live-suite performance. Controls were calibrated and
silent, so their causal reading is not extended to externally driven surrounds
or the published-weight state.

MCNS is a male central nervous system; Shiu et al. used the female FlyWire
connectome. The chosen gains are an engineering candidate, with the responsiveness
limits documented in the calibration report. Host echo paths count two-step paths
only, ignore transmitter signs, and do not model longer recurrent routes. Measured
IN/OUT covers anatomical boundary deliveries during the observed windows, including
feedback after circuit failure; it is not a measure of net membrane influence,
and correlation or onset alone cannot assign a particular return to a particular
outgoing spike.
