# DrosophilOS performance comparison

Status: complete.

Headline medians/min/max come only from unprofiled repeats. Primitive spikes are captured for all neurons on node 0 only, adding trace-copy cost; renders do not capture spikes.

## small

### Unchanged-circuit simulator-speed comparisons

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doom2-small / baseline | 469.7 | 969.4 | 480.6, 477.6, 990.3, 984.1 | 1443 | 2997 | 2.077 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom2-small / copies-8 | 75.75 | 898.4 | 89.64, 89.51, 1053, 1052 | 269.9 | 3196 | 11.84 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / baseline | 1352 | 4899 | 1402, —, 5079, — | 3600 | 1.31e+04 | 3.64 | 1439123 | 2865594 | — | 0 | 22 | 0 | 0 | 0 | 0 | no |
| doom4-small / copies-8 | 178.4 | 4309 | 229.9, 232.3, 5510, 5569 | 695.7 | 1.676e+04 | 24.1 | 1439123 | 2865594 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |

### Changed-circuit comparisons

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no selected configurations |—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|

### Combined

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doom2-small / baseline | 469.7 | 969.4 | 480.6, 477.6, 990.3, 984.1 | 1443 | 2997 | 2.077 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom2-small / copies-8 | 75.75 | 898.4 | 89.64, 89.51, 1053, 1052 | 269.9 | 3196 | 11.84 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / baseline | 1352 | 4899 | 1402, —, 5079, — | 3600 | 1.31e+04 | 3.64 | 1439123 | 2865594 | — | 0 | 22 | 0 | 0 | 0 | 0 | no |
| doom4-small / copies-8 | 178.4 | 4309 | 229.9, 232.3, 5510, 5569 | 695.7 | 1.676e+04 | 24.1 | 1439123 | 2865594 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
