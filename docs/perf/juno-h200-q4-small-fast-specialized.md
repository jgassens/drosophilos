# DrosophilOS performance comparison

Status: complete.

Headline medians/min/max come only from unprofiled repeats. Primitive spikes are captured for all neurons on node 0 only, adding trace-copy cost; renders do not capture spikes.

## small

### Unchanged-circuit simulator-speed comparisons

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doom2-small / baseline | 469.2 | 906.1 | 480.1, 477.1, 925.2, 919.2 | 1441 | 2801 | 1.943 | 581864 | 1180986 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom2-small / copies-8 | 75.64 | 793.6 | 89.48, 89.38, 934.9, 935.4 | 269.5 | 2837 | 10.53 | 581864 | 1180986 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / baseline | 1351 | 4547 | 1400, 1409, 4718, 4749 | 4215 | 1.426e+04 | 3.382 | 1299675 | 2598258 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / copies-8 | 178.2 | 3903 | 229.6, 232.1, 5010, 5060 | 694.9 | 1.523e+04 | 21.92 | 1299675 | 2598258 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |

### Changed-circuit comparisons

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no selected configurations |—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|

### Combined

| configuration | first-frame latency (neural s) | first-frame latency (wall s) | subsequent frame intervals (neural s / wall s) | total neural s | wall s | wall/neural | neurons | edges | spikes | wrong | missing | duplicates | invalid | faults | timeouts | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| doom2-small / baseline | 469.2 | 906.1 | 480.1, 477.1, 925.2, 919.2 | 1441 | 2801 | 1.943 | 581864 | 1180986 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom2-small / copies-8 | 75.64 | 793.6 | 89.48, 89.38, 934.9, 935.4 | 269.5 | 2837 | 10.53 | 581864 | 1180986 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / baseline | 1351 | 4547 | 1400, 1409, 4718, 4749 | 4215 | 1.426e+04 | 3.382 | 1299675 | 2598258 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
| doom4-small / copies-8 | 178.2 | 3903 | 229.6, 232.1, 5010, 5060 | 694.9 | 1.523e+04 | 21.92 | 1299675 | 2598258 | — | 0 | 0 | 0 | 0 | 0 | 0 | no |
