# Stage H: the flip-flop adder in the whole MCNS (2026-10-14)

## Answer

No: under Profile 2 (only real anatomical wiring and documented parameter edits), no tested region hosted configuration F's 740-neuron ordered four-bit adder reliably in the live simulated brain.  All six live runs were 0/50.  The complete condition-A image computed only after a large, measured boundary cut: in a silent surround, cut 50% of all_w0's host outputs (2,941,830 synapses) or 40% of vnc_w2's (1,334,045); under 2 Hz sensory input, cut 100% of all_w0's host inputs (4,131,952) or 90% of vnc_w2's (1,291,471).  The all_w0 input minimum is confirmed at 1.00; its 0.95 rung reached 10/10 but was not run at 50, placing the threshold above 0.90 and possibly at or below 0.95.  Here, a *host* is a real MCNS neuron assigned to one designed adder neuron; a *Profile 3 edge* is a designed edge added where anatomy does not supply it.

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

Minimal confirmed rungs: outputs/silent **0.50, 2,941,830 synapses, 50/50**; inputs/2 Hz **1.00, 4,131,952 synapses, 50/50**.

For all_w0 inputs, the 50/50 minimum is 1.00; the 0.95 rung reached 10/10 but was not confirmed at 50, so the threshold lies above 0.90 and possibly at or below 0.95.

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

## Limits

This is one placement seed.  Ladder rungs use 10 additions, with only the minimal confirmed rungs checked at 50 (and extra non-minimal VNC confirmations reported above); all_w0 input 0.95 reached 10/10 but was not confirmed at 50, so unconfirmed rungs should not be read as failures.  Configuration F itself was qualified only on 8+8 isolated additions plus one noisy probe, not the separate 50+50 isolated suite.  The LIF surround's excitability is a model property, as `docs/h1_placement.md` already notes.  In several cut runs the spike-identical-to-isolated count is below the correct count: spikes still enter but are harmless (for example, all_w0 output-cut 50% is 25/50 identical while 50/50 correct; VNC output-cut 40% is 3/50 identical while 50/50 correct).

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

`docs/stage_h/fullgraph.json` is the strict compact inventory of all 44 runs, including the source counts, cluster, and job id.
