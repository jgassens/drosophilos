# Performance comparison suite

`python -m drosophilos.bench.perf_campaign` implements the fixed comparison suite in
`docs/perf_campaign.md` §3.  It writes one `<out>.json` full record and one `<out>.md`
comparison table.  Primitive spike totals use `capture_spikes` for every neuron on node 0;
that is intentionally restricted to the short primitive runs because it adds host trace-copy
and memory cost.  Render runs do not collect a full spike trace.

The default suite contains:

- `cells` (separate one-cell ADD, AND, XOR, and MOV streams), fanout, the perspective
  multiplier (pipelined and array variants), and tick, each with at least 16 tokens;
- three 8 x 5 frames of `doom2.c`, using `2,258,0`, and `doom4.c`, using `6,256,0`;
- one and eight copies on the same explicitly selected Torch backend/device/dtype.

The reference-frame coverage gate runs before every small-render timing.  Doom2's turn then
forward sequence changes the rendered state, and its later reference frames differ.  Doom4's
`6,256,0` is the sequence used by the render test: all three frames differ and each contains
at least one of the imp-only palette colours 15, 10, or 9.  A static `0,0,0` Doom2 sequence is
rejected with an instruction to adjust the input sequence.

Run a normal CPU suite with:

```
uv run python -m drosophilos.bench.perf_campaign \
  --levels primitive,small --device cpu --copies 1,8 --repeats 3 \
  --out data/perf/cpu-suite
```

Use `--dry-run` to print selected configurations, `--profile-steps N` for separately recorded
diagnostic runs (never headline medians), and `--primitive-blocks cells` for the quick
one-cell check.  The Markdown has unchanged-circuit simulator, changed-circuit, and combined
sections; repeat medians/min/max for wall time and first-frame latency are retained in JSON.

`historical` selects 40 x 25 Doom2 (`2,258`, two frames) and 24 x 15 Doom4 sprite (`6,256`,
two frames).  It is disabled without `--historical` and is H200-only: it can take hours.

## Local default cost

Not measured in this isolated worktree: `uv` could not obtain the locked Torch dependency
(`mpmath`) because outbound package DNS/network access is unavailable.  The default remains
three repeats with the documented minimum 16 primitive tokens; measure the command above on
the target CPU before treating it as a local ten-minute budget.
