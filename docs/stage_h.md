# Stage H: the flip-flop adder in the whole MCNS (2026-10-14)

## Answer

No: under Profile 2 (only real anatomical wiring and documented parameter edits), no tested region hosted configuration F's 740-neuron ordered four-bit adder reliably in the live simulated brain.  All six live runs were 0/50.  The complete condition-A image computed only after a large, measured boundary cut: in a silent surround, cut 50% of all_w0's host outputs (2,941,830 synapses) or 40% of vnc_w2's (1,334,045); under 2 Hz sensory input, cut 95% of whole-CNS w0 (all_w0)'s host inputs (4,111,318 synapses) or 90% of vnc_w2's (1,291,471).  Whole-CNS w0 under 2 Hz drive succeeds at the smallest tested rung of 95% (90% failed; fractions between 90% and 95% were not tested); central_brain w0 needs more than 95%, while VNC w2 succeeds at 90%.  Here, a *host* is a real MCNS neuron assigned to one designed adder neuron; a *Profile 3 edge* is a designed edge added where anatomy does not supply it.

An earlier whole-brain batch was discarded because `h1_fullgraph` did not inject the flip-flop register's power-on pulse (every addition faulted even with the brain fully cut off); this was fixed in `e8f5f3b`, after which the sealed control computed 3/3.

## Placement by region

The grid used configuration F, `k_max=4`, four restarts, seed 0, and 50 fresh plus 50 chained isolated additions per condition.  “Exposure” is mean input/output synapses per host followed by total input/output edge pairs to non-hosts.  Profile-3 columns split missing designed pairs into autapses and all other pairs.

| region | isolation weight | carried / 1,516 | P3 autapse / other | host exposure: synapses mean; edges total | unplaced | A; B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| all | 0 | 809 (53.4%) | 148 / 559 | 5,591.3 / 4,292.3; 412,685 / 357,393 | 1 | 50/50; 0/50 |
| central brain | 0 | 783 (51.6%) | 148 / 585 | 5,100.0 / 3,653.4; 369,849 / 317,183 | 6 | 50/50; 0/50 |
| VNC | 0 | 690 (45.5%) | 148 / 678 | 3,106.8 / 2,478.7; 250,541 / 204,596 | 24 | 50/50; 0/50 |
| all | 1 | 634 (41.8%) | 148 / 734 | 2,548.3 / 2,649.4; 197,733 / 199,283 | 7 | 50/50; 0/50 |
| central brain | 1 | 617 (40.7%) | 148 / 751 | 2,840.8 / 2,665.8; 223,601 / 214,718 | 14 | 50/50; 0/50 |
| VNC | 1 | 590 (38.9%) | 148 / 778 | 2,071.7 / 2,267.2; 185,019 / 173,752 | 30 | 50/50; 0/50 |
| central brain | 2 | 565 (37.3%) | 148 / 803 | 2,484.9 / 2,501.4; 203,979 / 198,195 | 15 | 50/50; 0/50 |
| VNC | 2 | 557 (36.7%) | 148 / 811 | 1,840.3 / 2,086.8; 170,084 / 163,098 | 29 | 50/50; 0/50 |
| all | 2 | 551 (36.3%) | 148 / 817 | 2,292.4 / 2,439.6; 180,995 / 182,801 | 9 | 50/50; 0/50 |
| optic lobe | 0 | 359 (23.7%) | 148 / 1,009 | 4,253.0 / 3,687.1; 381,607 / 371,741 | 219 | 50/50; 0/50 |
| optic lobe | 2 | 279 (18.4%) | 148 / 1,089 | 3,758.8 / 3,642.8; 313,032 / 300,606 | 283 | 50/50; 0/50 |
| optic lobe | 1 | 276 (18.2%) | 148 / 1,092 | 3,708.5 / 3,655.3; 309,123 / 299,244 | 285 | 50/50; 0/50 |

Condition A restores every missing designed edge and therefore cannot rank placements by real wiring; condition B uses carried edges only and was 0/50 everywhere.  The optic lobe is the poor fit: 219–285 neurons unplaced and just 18–24% carried.  Quiet hosts cost coverage: VNC w2 has 1,840.3 mean outside input synapses, about one third of all_w0's 5,591.3, but carries 37% rather than 53%.

## Inside the whole brain

All runs below used the condition-A image, RefSim float64, and 50 additions.  The final row is the sealed control, not a live run.

| placement | surround | correct | wrong / fault / no ACCEPT / timeout / incomplete | maximum entering spikes per ms, exc / inh |
| --- | --- | ---: | ---: | ---: |
| all_w0 | silent | 0/50 | 0 / 22 / 2 / 26 / 0 | 27,005 / 14,714 |
| all_w0 | 2 Hz Poisson | 0/50 | 0 / 49 / 0 / 1 / 0 | 30,646 / 16,325 |
| vnc_w2 | silent | 0/50 | 1 / 0 / 0 / 7 / 42 | 4,726 / 4,895 |
| vnc_w2 | 2 Hz Poisson | 0/50 | 0 / 1 / 0 / 0 / 49 | 5,350 / 5,448 |
| central_brain_w0 | silent | 0/50 | 0 / 1 / 15 / 34 / 0 | 27,874 / 14,611 |
| central_brain_w0 | 2 Hz Poisson | 0/50 | 0 / 1 / 35 / 14 / 0 | 30,431 / 15,359 |
| all_w0, sealed | silent | 3/3 | 0 / 0 / 0 / 0 / 0 | 0 / 0 |

## Minimal boundary cut

The ladder tables retain every measured rung.  “Cut” is edge pairs / anatomical synapses at that rung; `correct/n` is the full-graph addition tally.  The 50-addition confirmation is shown as a second row at the same rung.

### all_w0

| boundary and surround | fraction | cut: edges / synapses | correct/n |
| --- | ---: | ---: | ---: |
| outputs, silent | 0.01 | 3,574 / 677,216 | 0/10 |
| outputs, silent | 0.03 | 10,722 / 1,185,881 | 0/10 |
| outputs, silent | 0.10 | 35,740 / 1,960,782 | 0/10 |
| outputs, silent | 0.30 | 107,218 / 2,685,116 | 0/10 |
| outputs, silent | 0.40 | 142,958 / 2,839,400 | 1/10 |
| outputs, silent | 0.50 | 178,697 / 2,941,830 | 10/10 |
| outputs, silent (confirmation) | 0.50 | 178,697 / 2,941,830 | **50/50** |
| outputs, silent | 0.70 | 250,176 / 3,064,829 | 10/10 |
| outputs, silent | 1.00 | 357,393 / 3,172,046 | 10/10 |
| inputs, 2 Hz Poisson | 0.10 | 41,269 / 2,515,729 | 0/10 |
| inputs, 2 Hz Poisson | 0.30 | 123,806 / 3,503,051 | 0/10 |
| inputs, 2 Hz Poisson | 0.60 | 247,611 / 3,936,025 | 0/10 |
| inputs, 2 Hz Poisson | 0.80 | 330,148 / 4,049,415 | 0/10 |
| inputs, 2 Hz Poisson | 0.90 | 371,417 / 4,090,684 | 0/10 |
| inputs, 2 Hz Poisson | 0.95 | 392,051 / 4,111,318 | 10/10 |
| inputs, 2 Hz Poisson | 1.00 | 412,685 / 4,131,952 | 10/10 |
| inputs, 2 Hz Poisson (confirmation) | 1.00 | 412,685 / 4,131,952 | **50/50** |

Minimal confirmed rungs in the ladder: outputs/silent **0.50, 2,941,830 synapses, 50/50**; inputs/2 Hz **0.95, 4,111,318 synapses, 50/50** (replication).

For all_w0, the 0.95 rung reached 10/10 in the ladder and was confirmed at 50/50 on seeds 0 and 1 in the replication below.  The smallest confirmed rung is 0.95 (90% failed; fractions between 90% and 95% were not tested).  Here all_w0 is whole-CNS w0, not a separate placement.

### vnc_w2

| boundary and surround | fraction | cut: edges / synapses | correct/n |
| --- | ---: | ---: | ---: |
| outputs, silent | 0.01 | 1,631 / 287,947 | 0/10 |
| outputs, silent | 0.03 | 4,893 / 541,201 | 0/10 |
| outputs, silent | 0.10 | 16,310 / 924,393 | 0/10 |
| outputs, silent | 0.30 | 48,930 / 1,264,269 | 7/10 |
| outputs, silent | 0.40 | 65,240 / 1,334,045 | 10/10 |
| outputs, silent (confirmation) | 0.40 | 65,240 / 1,334,045 | **50/50** |
| outputs, silent | 0.50 | 81,549 / 1,379,734 | 10/10 |
| outputs, silent (confirmation) | 0.50 | 81,549 / 1,379,734 | 50/50 |
| outputs, silent | 0.70 | 114,169 / 1,434,777 | 10/10 |
| outputs, silent | 1.00 | 163,098 / 1,483,706 | 10/10 |
| inputs, 2 Hz Poisson | 0.10 | 17,009 / 795,764 | 0/10 |
| inputs, 2 Hz Poisson | 0.30 | 51,026 / 1,095,744 | 0/10 |
| inputs, 2 Hz Poisson | 0.60 | 102,051 / 1,236,955 | 4/10 |
| inputs, 2 Hz Poisson | 0.70 | 119,059 / 1,257,454 | 8/10 |
| inputs, 2 Hz Poisson | 0.80 | 136,068 / 1,274,463 | 10/10 |
| inputs, 2 Hz Poisson (confirmation) | 0.80 | 136,068 / 1,274,463 | 48/50 |
| inputs, 2 Hz Poisson | 0.90 | 153,076 / 1,291,471 | 10/10 |
| inputs, 2 Hz Poisson (confirmation) | 0.90 | 153,076 / 1,291,471 | **50/50** |
| inputs, 2 Hz Poisson | 1.00 | 170,084 / 1,308,479 | 10/10 |
| inputs, 2 Hz Poisson (confirmation) | 1.00 | 170,084 / 1,308,479 | 50/50 |

Minimal confirmed rungs: outputs/silent **0.40, 1,334,045 synapses, 50/50**; inputs/2 Hz **0.90, 1,291,471 synapses, 50/50**.  VNC's quiet-host placement reduces the needed cut relative to all_w0, but does not eliminate it.  The earlier H1 latch adder was also 0/30 live and 10/10 only after cutting all about 327 k host input edges or, in a silent surround, all about 302 k host output edges (docs/h1_placement.md).

## What blocks it

1. **Wiring fit.**  The best F coverage is 809/1,516 pairs (53.4%) and condition B is 0/50 for every grid point.  So anatomy alone has not supplied a functioning adder image.
2. **Parts supply.**  F demands 140 inhibitory latch autapses at -724 quanta and eight one-shot self-edges at -15,932.  The autapse census finds 101 anatomical self-edges, and zero full-rule-carriable latch or one-shot demands.  Under Dale's law, all relevant designed sources vote excitatory while their self-edge is inhibitory, so no real neuron can carry either demanded self-edge under the full rule.
3. **Live surround.**  With condition A complete, all live placements still score 0/50; suppressing their boundary is what restores 50/50.  Output cuts were measured only with a silent surround and input cuts only under 2 Hz Poisson drive, so attributing the failure to the circuit's own output waking the brain (silent) versus sensory drive flooding the inputs (2 Hz) also relies on H1's cross-condition control, where outputs zeroed under 2 Hz yielded 0/10 (docs/h1_placement.md).

On the present evidence, the live surround decides the whole-brain outcome: the same complete image computes in the sealed 3/3 control and only becomes reliable at the measured boundary thresholds.  Wiring fit and the unavailable F autapses independently rule out an unmodified Profile-2 realization, but they do not explain the transition within the already-complete image.

## Round 2 (2026-10-15/16): Dale-compatible latches, fault attribution and the shielding threshold

This round tests the two Dale-compatible latch designs documented in `docs/stage_h/dale_feedback.md`: `dale_private` uses a dedicated inhibitory feedback neuron for each latch member, while `dale_biased` is its sparse, asynchronous score-winning variant.  Here *carried* means a designed edge pair supplied by real anatomical wiring; *designed* is every pair required by that image.  The compact, strict source inventory is `docs/stage_h/round2.json`.

### Placement

The Dale images have 1,656 designed pairs rather than the autapse image's 1,516.  They remove nearly all of the original latch-autapse shortfall, but this does not establish a live computation.

| placement | feedback | carried / designed | placed / unplaced |
| --- | --- | ---: | ---: |
| all_w0 | autapse | 809 / 1,516 | 739 / 1 |
| all_w0 | dale_private | 1,049 / 1,656 | 879 / 1 |
| all_w0 | dale_biased | 1,063 / 1,656 | 875 / 5 |
| vnc_w2 | autapse | 557 / 1,516 | 711 / 29 |
| vnc_w2 | dale_private | 573 / 1,656 | 853 / 27 |
| vnc_w2 | dale_biased | 576 / 1,656 | 851 / 29 |

### Live brain, no cut

The no-cut `dale_biased` live runs are 0/50 in both surrounds and placements, just as the autapse runs are.  The equal 0/50 results show that replacing the latch did not rescue the computation.  Maximum entering excitation is reported per millisecond.

| placement | surround | autapse correct/n; max exc | dale_biased correct/n; max exc |
| --- | --- | ---: | ---: |
| all_w0 | silent | 0/50; 27,005 | 0/50; 40,126 |
| all_w0 | 2 Hz Poisson | 0/50; 30,646 | 0/50; 46,137 |
| vnc_w2 | silent | 0/50; 4,726 | 0/50; 4,877 |
| vnc_w2 | 2 Hz Poisson | 0/50; 5,350 | 0/50; 6,252 |

### Fault attribution

An *incoming boundary edge* is an anatomical edge from outside the host image into it.  A *scored edge* is an incoming edge with a spike arrival in the 20 ms window before a failure, and was therefore eligible for attribution.  Only 3,268 of 412,685 incoming edges in the silent run, and 38,555 in the Poisson run, carried such spikes in those windows; other incoming edges may have carried spikes outside the window.  Scores below are the top attribution scores, not a claim that cutting a named source type is sufficient.

| surround | incoming / scored edges | top source types (score) | top source superclasses (score) |
| --- | ---: | --- | --- |
| silent | 412,685 / 3,268 | lLN1_bc (18,627,200); lLN2P_a (2,144,448); GNG019 (1,060,128) | cb_intrinsic (34,081,360); descending_neuron (2,542,688); cb_sensory (73,920) |
| 2 Hz Poisson | 412,685 / 38,555 | lLN1_bc (11,554,064); GNG019 (981,600); AVLP080 (811,712) | cb_intrinsic (20,166,576); cb_sensory (8,919,248); vnc_sensory (2,549,616) |

### Shielding

Each row cuts all_w0 host inputs.  “Ranking” chooses boundary edges either by attribution score, by synapse count, or by the named source type; the fraction applies to the ranked edge list.  `correct/n` is additions correct out of additions attempted.  Every measured input-cut rung follows.

| ranking | surround | fraction | edges / synapses cut | correct/n | max excitatory spikes entering per ms |
| --- | --- | ---: | ---: | ---: | ---: |
| attribution | silent | 0.05 | 20,635 / 1,783,993 | 0/10 | 37,730 |
| attribution | 2 Hz Poisson | 0.05 | 20,635 / 357,669 | 0/10 | 26,221 |
| attribution | silent | 0.10 | 41,269 / 2,471,943 | 0/10 | 22,416 |
| attribution | 2 Hz Poisson | 0.10 | 41,269 / 989,672 | 0/10 | 37,420 |
| attribution | silent | 0.20 | 82,537 / 3,147,488 | 0/10 | 29,890 |
| attribution | 2 Hz Poisson | 0.20 | 82,537 / 2,830,363 | 0/10 | 29,179 |
| attribution | silent | 0.30 | 123,806 / 3,496,674 | 0/10 | 24,836 |
| attribution | 2 Hz Poisson | 0.30 | 123,806 / 3,376,405 | 0/10 | 27,921 |
| attribution | silent | 0.50 | 206,343 / 3,842,974 | 0/10 | 17,057 |
| attribution | 2 Hz Poisson | 0.50 | 206,343 / 3,821,051 | 0/10 | 17,984 |
| attribution | silent | 0.70 | 288,880 / 4,008,147 | 0/10 | 8,508 |
| attribution | 2 Hz Poisson | 0.70 | 288,880 / 4,008,147 | 0/10 | 8,377 |
| attribution | silent | 0.90 | 371,417 / 4,090,684 | 0/10 | 2,082 |
| attribution | 2 Hz Poisson | 0.90 | 371,417 / 4,090,684 | 0/10 | 2,285 |
| synapse | silent | 0.05 | 20,635 / 1,880,500 | 0/10 | 37,262 |
| synapse | silent | 0.10 | 41,269 / 2,515,729 | 0/10 | 23,295 |
| synapse | silent | 0.20 | 82,537 / 3,163,127 | 0/10 | 28,906 |
| synapse | silent | 0.30 | 123,806 / 3,503,051 | 0/10 | 27,514 |
| type: lLN1_bc | silent | — | 723 / 35,144 | 0/10 | 26,177 |
| type: lLN1_bc | 2 Hz Poisson | — | 723 / 35,144 | 0/10 | 28,267 |

There is no small, targeted shield in these data.  Attribution-ranked cuts did no better than the synapse-ranked cuts.  In the earlier synapse-ranked input ladder above, the all_w0 image's smallest tested successful rung is 95% of incoming boundary edges under 2 Hz drive (90% failed; fractions between 90% and 95% were not tested).  The attribution ranking remained 0/10 even at 90% of incoming edges in either surround.

### Replication (2026-10-16), 2 Hz Poisson surround, synapse-ranked input cuts, 50 additions each

| Placement | Cut | Seed | Correct | Fault | No accept | Timeout | Max exc spikes in per ms | Job |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all_w0 | 90% | 0 | 0/50 | 3 | 37 | 10 | 2,629 | Juno 459507 |
| all_w0 | 95% | 0 | 50/50 | 0 | 0 | 0 | 740 | Juno 459508 |
| all_w0 | 95% | 1 | 50/50 | 0 | 0 | 0 | 746 | Juno 459509 |
| all_w0 | 100% | 1 | 50/50 | 0 | 0 | 0 | 0 | G2 4023560 |
| central_brain_w0 | 95% | 0 | 5/50 | 0 | 0 | 45 | 1,555 | G2 4023561 |
| central_brain_w0 | 100% | 0 | 50/50 | 0 | 0 | 0 | 0 | G2 4023564 |

The whole-CNS w0 threshold replicates across seeds 0 and 1: 90% fails 0/50, while 95% passes 50/50 on both seeds.  The second placement, central_brain w0, needs more than 95% (5/50 at 95%, with all 45 failures timing out; 50/50 at 100%), so the shielding requirement is placement dependent, from about 80–90% for the VNC w2 hosts to more than 95% for central_brain w0.

Result files under `data/stage_h/fullgraph/`: `all_w0_cutin0.90_poisson_n50_s0.json`, `all_w0_cutin0.95_poisson_n50_s0.json`, `all_w0_cutin0.95_poisson_n50_s1.json`, `all_w0_cutin1.0_poisson_n50_s1.json`, `central_brain_w0_cutin0.95_poisson_n50_s0.json`, and `central_brain_w0_cutin1.0_poisson_n50_s0.json`; built at `ba2e7d5`.

### Round-2 limits

The Dale, attribution, and type-cut measurements use one seed and one placement.  The shielding threshold replication used two seeds (all_w0) and two placements (all_w0, central_brain_w0).  The shielding ladders use 10 additions at every rung below the minima; they are not 50-addition confirmations.

The [whole-brain surround calibration](stage_h/surround_calibration.md) separately finds that the selected stable surround does not restore the placed adder: all four live placement/surround runs were 0/50, while sealed and outputs-zeroed controls pass.  It also documents the stability-versus-responsiveness tradeoff and its proxy limits.

## Limits

The Dale, attribution, and type-cut measurements use one seed and one placement.  Ladder rungs use 10 additions, with minimal confirmed rungs checked at 50 (and extra non-minimal VNC confirmations reported above); the 0.95 all_w0 input rung was confirmed 50/50 on seeds 0 and 1 in the replication.  Unconfirmed rungs should not be read as failures.  Configuration F itself was qualified only on 8+8 isolated additions plus one noisy probe, not the separate 50+50 isolated suite.  The LIF surround's excitability is a model property, as `docs/h1_placement.md` already notes.  In several cut runs the spike-identical-to-isolated count is below the correct count: spikes still enter but are harmless (for example, all_w0 output-cut 50% is 25/50 identical while 50/50 correct; VNC output-cut 40% is 3/50 identical while 50/50 correct).

## Reproduce

The Slurm logs record commits `e8f5f3b` and `6f9a702`; they differ only in `stall_diag.py`, `fault_diag.py`, and `test_stall_diag.py`, which whole-brain runs do not import, so the simulated code is identical.  The placement command (substitute the region, weight, and output names for each grid point) was:

```sh
python -m drosophilos.bench.h_stage_place --region vnc --isolation-weight 2 \
  --cases 50 --restarts 4 --time-limit 600 --seed 0 \
  --out data/stage_h/place/vnc_w2.json \
  --mapping-out data/stage_h/place/vnc_w2_mapping.json
```

The whole-brain commands used the corresponding mapping and either boundary ladder flag:

```sh
python -m drosophilos.bench.h1_fullgraph --mapping data/stage_h/place/vnc_w2_mapping.json \
  --n-cases 10 --seed 0 --surround silent --rate 2 --max-ms 1200 \
  --cut-outputs-frac 0.4 --out data/stage_h/fullgraph/vnc_w2_cutout0.4_silent.json
python -m drosophilos.bench.h1_fullgraph --mapping data/stage_h/place/vnc_w2_mapping.json \
  --n-cases 50 --seed 0 --surround poisson --rate 2 --max-ms 1200 \
  --cut-inputs-frac 0.9 --out data/stage_h/fullgraph/vnc_w2_cutin0.9_poisson_n50.json
```

Submit the same module and flags through either cluster with `slurm/submit.sh`; Juno used the CPU script and G2 the default cluster script:

```sh
slurm/submit.sh --cluster juno --script juno-cpu --time=2-00:00:00 -- drosophilos.bench.h1_fullgraph [flags above]
slurm/submit.sh --cluster g2 --time=2-00:00:00 -- drosophilos.bench.h1_fullgraph [flags above]
```

`docs/stage_h/fullgraph.json` inventories the 44 ladder runs, including source counts, cluster, and job id.  The six replication runs (Juno 459507–459509; G2 4023560, 4023561, 4023564) are listed in the replication table above.
