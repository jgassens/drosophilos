# DrosophilOS performance comparison

Status: complete.

Headline medians/min/max come only from unprofiled repeats. Primitive spikes are captured for all neurons on node 0 only, adding trace-copy cost; renders do not capture spikes.

## small

### Unchanged-circuit simulator-speed comparisons

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doom2-small / baseline | 469.7 | 2934 | 480.6, 477.6, 3054, 3022 | 1443 | 9126 | 6.325 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom2-small / copies-8 | 75.74 | 1053 | 89.62, 89.5, 1249, 1248 | 269.8 | 3775 | 13.99 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / baseline | 1352 | 1.019e+04 | 1402, —, 1.077e+04, — | 3600 | 2.752e+04 | 7.645 | 1439123 | 2865594 | — | 0 | 22 | 0 | 0 | 0 | 0 | no |
| doom4-small / copies-8 | 178.4 | 4224 | 229.9, 232.3, 5440, 5499 | 695.6 | 1.652e+04 | 23.75 | 1439123 | 2865594 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |

### Changed-circuit comparisons

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no selected configurations |—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|

### Combined

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doom2-small / baseline | 469.7 | 2934 | 480.6, 477.6, 3054, 3022 | 1443 | 9126 | 6.325 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom2-small / copies-8 | 75.74 | 1053 | 89.62, 89.5, 1249, 1248 | 269.8 | 3775 | 13.99 | 664024 | 1338506 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / baseline | 1352 | 1.019e+04 | 1402, —, 1.077e+04, — | 3600 | 2.752e+04 | 7.645 | 1439123 | 2865594 | — | 0 | 22 | 0 | 0 | 0 | 0 | no |
| doom4-small / copies-8 | 178.4 | 4224 | 229.9, 232.3, 5440, 5499 | 695.6 | 1.652e+04 | 23.75 | 1439123 | 2865594 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
