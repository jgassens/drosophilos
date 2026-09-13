# M1 storage primitive comparison (Profile 3, default neuron parameters)

Measured by `drosophilos/lib/latch_alternatives.py` (data in `data/m1/latch_alternatives.json`).
H0 proved a two-neuron loop can hold a bit; M1 asks whether anything cheaper in activity
exists in this neuron model before that loop becomes the production register.

| candidate | neurons | period | member rate | spikes / 100 ms held | survives 5 % / 10 % weight noise (of 100) | minimal reset (both members of one link) | recovery before a 1.4× ignition |
|---|---|---|---|---|---|---|---|
| loop2_1.4x | 2 | 4.6 ms | 217.4 Hz | 44 | 100 / 100 | 1 × 1.5× loop | 15.0 ms |
| loop2_1.05x | 2 | 6.9 ms | 144.9 Hz | 29 | 100 / 95 | 1 × 0.75× loop | 35.0 ms |
| loop2_1.4x_delay10ms | 2 | 5.2 ms | 192.3 Hz | 38 | 100 / 100 | 4 × 0.75× loop | 40.0 ms |
| ring4_1.4x | 4 | 4.7 ms | 212.8 Hz | 86 | 100 / 100 | 4 × 0.75× loop | 10.0 ms |
| ring8_1.4x | 8 | 4.6 ms | 217.4 Hz | 140 | 100 / 100 | not found up to 4 × 3.0× | None ms |

## Reading

- **The two-neuron loop at 1.4× drive is the production register.** It is the cheapest in
  neurons, the most robust to weight noise, the easiest to reset (one 1.5× pulse on both
  members), and the fastest to recover.
- **Lowering the drive lowers the broadcast** (29 vs 44 spikes per 100 ms) but costs
  robustness (5 % of loops die within 100 ms at 10 % weight noise) and doubles recovery.
- **Longer rings do not lower the broadcast**: they end up holding several circulating
  spikes (the ignition pulse and re-entries are not gated by refractoriness once the link
  delay exceeds it), so a 4-ring emits about twice and an 8-ring about three times the
  spikes of the 2-loop, and inhibiting two members of an 8-ring does not stop it.
- **Long synaptic delays (10 ms, Profile 3 only) do not help either**: the loop then carries
  multiple spikes and behaves like a ring.
- **There is no low-activity storage in this neuron model.** Without a slow state variable
  (adaptation, synaptic depression, a second membrane compartment) the only memory longer
  than tau_m = 20 ms is a circulating spike, and its broadcast per unit time is set by the
  regeneration period, ~4.7 ms. Silence cannot encode a value ("silence is not zero"), so an
  explicitly refreshed latch would still have to spike at least once per refresh interval
  and would need a refresh clock, which the self-timed design does not have. A lower-
  activity register therefore requires a model extension, which Profile 2 cannot grant on
  the anatomical substrate and which is recorded here as a Stage H question, not an M1 fix.
- Consequence for Profile 2 embedding (H0's isolation finding): the register's broadcast is
  ~44 spikes per 100 ms per bit; the compiler must budget silenced outputs accordingly.
