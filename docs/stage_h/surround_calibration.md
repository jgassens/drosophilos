# Calibrating the whole-brain surround

Can the MCNS LIF surround support a sparse, bounded evoked response that decays when
drive stops, and can the placed adder compute inside that state with its boundary live?
[H0 findings, section 5](../h0_findings.md) reports that 2 Hz sensory drive alone recruits
roughly 30,000 neurons into sustained runaway. [Stage H](../stage_h.md) reports 0/50 live
placed-adder runs and recovery only after cutting 90–95% of incoming edges. First test
the surround **without a circuit**, then repeat the live adder experiment using settings
that pass the surround checks. No whole-brain simulations should run on the laptop.

The baseline remains the Shiu et al. (2024) whole-brain LIF parameter set in `Params`:
E_L = −52 mV, V_th = −45 mV, tau_m = 20 ms, tau_s = 5 ms, refractory period = 2.2 ms,
w_syn = 0.275 mV per anatomical synapse, delay = 1.8 ms, dt = 0.1 ms. These experiments
change anatomical weights, leaving those neuronal parameters fixed. The inhibitory
transmitters are GABA, glutamate, and histamine under the existing MCNS convention.

## Weight knobs and edge scope

`full_graph_topology` and `h1_fullgraph` accept these opt-in settings:

| Python setting / CLI flag | Default | Effect |
| --- | --- | --- |
| `surround_scale_exc` / `--surround-scale-exc` | 1 | Multiply each eligible positive anatomical weight. |
| `surround_scale_inh` / `--surround-scale-inh` | 1 | Multiply each eligible negative anatomical weight, preserving its sign. |
| `surround_input_cap` / `--surround-input-cap` | None (uncapped) | After the gains, proportionally reduce eligible excitatory inputs to each postsynaptic neuron whose summed input exceeds this cap. |

All values must be finite and nonnegative; zero disables the corresponding weights.
The cap's unit is a **single anatomical synapse's weight**, or 16 integer quanta,
equivalent to synapse counts at unit gain. For eligible positive weights q_ij after
the excitatory gain, let S_j = sum_i(q_ij / 16). Multiply those inputs by
min(1, cap / S_j), with zero-input destinations unchanged. Thus the cap acts on the
gain-scaled input, rather than the original anatomical synapse count.

Eligible edges are anatomical edges with **at least one endpoint outside the image**:
surround → surround, host → surround, and surround → host. Every host → host edge is
excluded, including kept parasitic edges and host autapses. Designed/carried edges,
Profile 3 edges (including synthetic neurons), their delays, and image biases are
untouched. Designed excitatory input does not consume the cap. In a circuit-free trial,
the image is empty and every anatomical edge is eligible.

Apply gains, then cap, then round **once** to the nearest integer quantum with ties to
even (`numpy.rint`, symmetric for both signs). Rounding can move the sum slightly above
the continuous cap, by at most half a quantum per rounded incoming edge. Very small
weights can become zero; zero-weight edges remain in the topology. Overflow is rejected.
Boundary cuts, if requested for an adder control, follow scaling and rank the original
anatomical counts. A cut does not cause the surviving edges to be scaled back up.

Non-default runs record all three settings in `full_graph_counts` (`fg.counts`), plus
eligible/changed edge counts, capped neuron count, total newly zeroed edges, and
`surround_zeroed_by_rounding`. The latter counts nonzero continuous weights rounded to
zero, excluding exact zeros produced by a zero gain/cap. Counts describe the scaling
operation before boundary cuts. The adder JSON also records explicitly supplied flags
in `args`. With no knobs, defaults are implicit and the legacy counts/args layout is
preserved. A SHA-256 regression frozen from the pre-change code covers topology arrays,
bias, index map, and counts; explicit (1, 1, None) gives the same digest.

## Circuit-free calibration job

`python -m drosophilos.bench.h_surround_calib` loads MCNS and its anatomical topology
once, then loops over a Cartesian `--grid` inside one job. Axes are `exc`, `inh`, and
`cap`; omitted axes use their defaults, and `none` disables the cap. For example,
`'exc=1,0.75,0.5,0.35,0.25;inh=1,2,4;cap=none,2000,500'` gives 45 configurations.
Each protocol starts a fresh simulator at rest, with no image, circuit bias, or other
drive. The same seeds and source choices are reused across settings for paired comparisons.

| Protocol | Drive and observation |
| --- | --- |
| P1 sensory | All neurons selected by `surround_populations(m)['sensory']`, independently driven with H0/H1's Bernoulli-per-step approximation to Poisson events at `--rate` Hz (default 2), for 300 ms; then 300 ms off. `--sensory-seed` defaults to 0. |
| P2 point source | `--n-sources` (default 1) non-sensory neurons selected uniformly without replacement, forced at 213 Hz for 300 ms; then 300 ms off. Repeat for `--seeds` (default `0,1,2`). No sensory drive. |
| P3 rest | No drive for 200 ms. Report any activity. |

P1 deliberately uses the same `surround_events` mechanism as the adder bench: external
events deliver 100,000 quanta. This is a forcing-current pulse, so the requested event
rate is not a guarantee of exactly that somatic spike rate; residual current, refractory
collisions, and recurrent input can change the observed rate. Output includes the event
count as well as observed rates. P2 instead clamps source voltage/current between its
scheduled spikes during drive, preventing forcing-current tails from inflating 213 Hz.
Ticks are rounded to the nearest dt (64 source spikes over 300 ms). Sources evolve
freely after drive off. Output records source body IDs, observed source drive counts,
and recruitment excluding sources, making this distinction explicit.

`--out` is overwritten with compact JSONL: one self-contained record per configuration,
flushed immediately. Each includes settings, model parameters, connectome version,
topology edit counts, P1, each P2 replicate, and P3. There are **no spike dumps** or
sidecar traces. Spike batches are accumulated and discarded every simulator step, so a
runaway trial does not retain its event history in memory. Do not append a rerun to the
same output when comparing different job commits; use distinct filenames.

## Metrics and screening definitions

Every interval is left-inclusive and right-exclusive. Counts include forced source
spikes; P2 additionally gives `recruited_non_sources`. Each trial reports:

- Neurons spiking at least once over the complete trial, their fraction of **166,700**
  (`active_fraction_mcns`), and fraction of the simulated population. P1 also breaks
  activity down by superclass, with each class's population denominator.
- `spikes_per_bin`: total population spikes in consecutive 10 ms bins (60 driven-trial
  bins or 20 rest bins). Multiply by 100 for total population spikes/s; divide that by
  population size for the population-average per-neuron rate.
- Mean and median rates during the 300 ms drive among neurons with at least one
  **drive-period** spike. Off-only cells are excluded. Empty/no-drive rates are null.
- `last_100ms`: spike count, active cell count, population spikes/s, and mean per-neuron
  rate during the last 100 ms (500–600 ms in P1/P2, 100–200 ms at rest).
- `decay_ms`: first off-period bin start from which all remaining observed bins are
  strictly below 1% of the **drive-period** peak bin count. This sustained criterion
  rejects a transient dip followed by a rebound. Resolution is 10 ms; a null value
  means no sustained crossing was observed. Completely silent driven trials give zero;
  rest has no drive reference and gives null.
- `runaway`: true if the least-squares slope of the final five drive bins (250–300 ms)
  is positive, **or** if the 400–410 ms bin (100 ms after drive off) exceeds 10% of the
  drive peak. Both component flags are recorded. This is a screening flag: short noisy
  trials can have a positive slope by chance, so inspect paired seeds/time courses.
  Rest has no drive-related runaway flag; `any_activity` and total spikes are its check.

## Proposed acceptance targets — to check against literature

These are provisional engineering targets, **not established biological bounds**. Check
them against fly recordings, sensory stimulation protocols, and the published model's
activity statistics before calling any setting biologically calibrated. Compare a
candidate across all P2 seeds and additional P1 seeds before selecting it.

| Check | Proposed target |
| --- | --- |
| Bounded P1 response | No sustained rise or runaway flag across repeated seeds; at most 10% of all neurons active across the trial, with non-sensory recruitment at most 1% of the brain. |
| Plausible sensory sparsity | Separate directly driven sensory cells from recruited central/output cells. At 2 Hz over 300 ms, about 45% of sensory cells receive an event: roughly 8,100 of 17,937, already about 4.9% of the brain. A total-activity target below that would reject the drive itself. |
| Evoked firing rates | Initially screen for mean active drive rate below 20 Hz and median below 10 Hz; validate per superclass against recordings and assess the H0 forcing-current contribution. |
| Decay and persistence | Sustained decay below 1% of drive peak within 100 ms; last 100 ms off-period mean bin count below 1% of peak, ideally no spikes. No above-10%-at-100-ms flag. |
| Single 213 Hz source | At most 100 non-source neurons recruited per trial, no runaway flag, and zero last-100-ms activity after release, across all source-selection seeds. |
| Rest | Exactly zero spikes in 200 ms. This check alone cannot distinguish a quiescent useful surround from a silenced one. |

Avoid selecting a trivially disconnected graph solely because it passes decay and rest:
confirm a nonzero evoked response and quantify zeroed edges and recruited superclasses.
Once these checks pass, test whether they improve the adder with the complete live
boundary. Adder accuracy is a separate result from biological plausibility.

## Exact cluster commands

Run from the repository root **after the orchestrator has reviewed and committed the
changes**: `submit.sh` requires a clean committed checkout and pushes/submits that commit.
Its options (`--cluster`, `--script`, `--ref`, `--dry-run`) must come **before** sbatch
options. These commands select G2 CPU preempt nodes and disable GPU resources. Outputs
live under `data/` so they survive the job worktree cleanup.

```bash
slurm/submit.sh --cluster g2 --partition=cpu-preempt --gres=none \
  --cpus-per-task=4 --mem=64G --time=24:00:00 \
  -- drosophilos.bench.h_surround_calib \
  --grid 'exc=1,0.75,0.5,0.35,0.25;inh=1,2,4;cap=none,2000,500' \
  --rate 2 --n-sources 1 --sensory-seed 0 --seeds 0,1,2 \
  --out data/stage_h/surround_calibration/grid_seed0.jsonl
```

For candidates, repeat P1 with additional seeds, using the same P2 source set:

```bash
# EXAMPLE candidate values only; replace after reviewing the sweep and literature.
for seed in 1 2; do
  slurm/submit.sh --cluster g2 --partition=cpu-preempt --gres=none \
    --cpus-per-task=4 --mem=64G --time=12:00:00 \
    -- drosophilos.bench.h_surround_calib \
    --grid 'exc=0.5;inh=2;cap=500' \
    --rate 2 --n-sources 1 --sensory-seed "$seed" --seeds 0,1,2 \
    --out "data/stage_h/surround_calibration/candidate_seed${seed}.jsonl"
done
```

Then re-run the placed adder, 50 additions per surround, with **no boundary cuts**:

```bash
# EXAMPLE candidate values only; replace with the chosen calibration settings.
for surround in silent poisson; do
  slurm/submit.sh --cluster g2 --partition=cpu-preempt --gres=none \
    --cpus-per-task=4 --mem=64G --time=12:00:00 \
    -- drosophilos.bench.h1_fullgraph \
    --mapping data/stage_h/place/all_w0_mapping.json \
    --n-cases 50 --seed 0 --surround "$surround" --rate 2 --max-ms 1200 \
    --surround-scale-exc 0.5 --surround-scale-inh 2 --surround-input-cap 500 \
    --out "data/stage_h/fullgraph/all_w0_calibrated_${surround}_n50.json"
done
```

For an uncapped candidate, omit `--surround-input-cap` in the adder command and use
`cap=none` in calibration. Use the mapping's recorded builder options; do not override
storage/width to manufacture a different circuit. Match legacy fresh-per-addition
results first; a separate `--chained` run can test accumulated surround activity.

## Results (pending cluster runs)

- Commit / cluster job IDs / output paths: **TBD**
- Literature comparisons and revised acceptance bounds: **TBD**
- Baseline P1 / P2 / P3, decay and runaway: **TBD**
- Sweep candidates and zeroed-edge counts: **TBD**
- Selected settings and repeat-seed validation: **TBD**
- Live adder silent / Poisson accuracy, boundary envelope, and isolated comparison: **TBD**

Only synthetic tests were run locally; no whole-brain calibration or adder results are
claimed here.
