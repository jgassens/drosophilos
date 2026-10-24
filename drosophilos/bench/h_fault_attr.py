"""Bounded, aggregate-only attribution of failed H1 additions to incoming host edges.

Scores are temporal associations, not causal estimates. A delivery is a source spike at s
on an embedded edge of delay d, arriving at s+d; only arrivals in [failure-W, failure)
count. Weight is signed embedded quanta, and score is delivered spikes * abs(weight).
Only a short source-spike tail survives between chained additions; no traces are saved.
The compressed NPZ contains full per-edge arrays keyed by endpoint bodyIds, plus model-rule
provenance as a scalar JSON string (no pickle), and can
be passed to full_graph_topology(attribution_in=...). Compact JSON is a report, not a cut file.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..connectome.embed_image import load_attribution
from ..sim.model import ModelRules


def validate_attribution_rules(path: Path, rules: ModelRules) -> None:
    """Refuse scores from different integration rules; allow unlabelled legacy files."""
    load_attribution(path, rules)


def failure_point(ch, rec, trace, reference, reference_rec, end_step: int) -> dict | None:
    """Locate an observable failure using exactly the harness's circuit spike record.

    Success requires valid expected decode AND READY. For a failure, take the earliest of:
    a consumer FAULT spike; producer watchdog TIMEOUT; opposite-to-expected consumer rail
    tap; expected tap earlier (relative to load) than that tap's first isolated spike.
    Rail checks stop at the first of ACCEPT, consumer reset, or terminal observation, so
    teardown is not called corruption. Ties prefer FAULT, TIMEOUT, wrong, then early rail.
    If none exists, the stall point is the terminal observation: READY's spike step if
    present, otherwise the exclusive deadline step returned by run_words. This is a
    detection time, not an inferred last-progress time (latches keep spiking in stalls).
    """
    if rec.status == "valid" and rec.decoded == rec.word and rec.ready_step is not None:
        return None
    ev = trace.events
    ev = ev[(ev["step"] >= rec.load_step) & (ev["step"] <= end_step) & (ev["node"] == 0)]
    candidates = []

    def first(neurons, reason, priority, events=ev):
        steps = events["step"][np.isin(events["neuron"], list(neurons))]
        if len(steps):
            candidates.append((int(steps.min()), priority, reason))

    first(ch.consumer.fault, "fault_spike", 0)
    if ch.producer.watchdog is not None:
        first([ch.producer.watchdog.timeout.u], "timeout_spike", 1)
    rail_end = min(s for s in (rec.accept_step, rec.reset_step, end_step) if s is not None)
    rails = ev[ev["step"] <= rail_end]
    ref = reference.events
    ref = ref[(ref["step"] >= reference_rec.load_step) & (ref["node"] == 0)]
    for bit, taps in enumerate(ch.consumer.rail_taps):
        expected = (rec.word >> bit) & 1
        first([taps[1 - expected]], "wrong_rail", 2, rails)
        ref_steps = ref["step"][ref["neuron"] == taps[expected]]
        if len(ref_steps):
            earliest = int(ref_steps.min()) - reference_rec.load_step + rec.load_step
            first([taps[expected]], "early_rail", 3, rails[rails["step"] < earliest])
    if candidates:
        step, _, reason = min(candidates)
    else:
        step = int(end_step)
        reason = "stall_deadline" if rec.ready_step is None else "stall_terminal"
    return {"step": step, "reason": reason}


class IncomingAttribution:
    """One O(incoming edges + recent source spikes) accumulator for an entire bench run."""

    def __init__(self, m, fg, roles, window_ms: float, params, *, rules: ModelRules = ModelRules()):
        if not np.isfinite(window_ms) or window_ms <= 0:
            raise ValueError("attribute window must be finite and positive")
        self.window_ms = float(window_ms)
        self.rules = rules
        # Integer arrival steps in [T-W/dt, T) are exactly [T-floor(W/dt), T).
        self.window_steps = int(Fraction(str(window_ms)) / Fraction(str(params.dt)))
        topo = fg.topology
        circuit = np.zeros(topo.n, bool)
        circuit[fg.index_map] = True
        hosts = circuit.copy()
        hosts[m.n:] = False
        edge = np.flatnonzero(hosts[topo.dst] & ~circuit[topo.src])
        self.pre, self.post = topo.src[edge], topo.dst[edge]
        self.weight = topo.quanta[edge].astype(np.int64)
        self.delay = topo.delay[edge]
        self.n = topo.n
        self.sources = np.zeros(topo.n, bool)
        self.sources[self.pre] = True
        self.max_delay = int(self.delay.max()) if len(edge) else 0
        # Retain anatomical counts even when some of these embedded weights are zeroed.
        anatomical = hosts[m.post] & ~circuit[m.pre]
        keys = m.pre[anatomical].astype(np.int64) * m.n + m.post[anatomical]
        order = np.argsort(keys)
        keys, starts = np.unique(keys[order], return_index=True)
        counts = np.add.reduceat(m.count[anatomical][order].astype(np.int64), starts) if len(starts) else np.empty(0, np.int64)
        self.synapses = counts[np.searchsorted(keys, self.pre.astype(np.int64) * m.n + self.post)]
        bodies = m.neurons["bodyId"].to_numpy()
        self.pre_body, self.post_body = bodies[self.pre], bodies[self.post]
        self.post_designed = fg.designed_of()[self.post]
        self.roles = roles
        self.labels, self.codes, self.group_failures = {}, {}, {}
        for column in ("type", "superclass"):
            labels = m.neurons[column].fillna("<unknown>").astype(str).to_numpy()[self.pre]
            self.labels[column], self.codes[column] = np.unique(labels, return_inverse=True)
            self.group_failures[column] = np.zeros(len(self.labels[column]), np.int64)
        self.score = np.zeros(len(edge), np.int64)
        self.deliveries = np.zeros(len(edge), np.int64)
        self.edge_failures = np.zeros(len(edge), np.int64)
        self.failures = 0
        self._tail_steps = np.empty(0, np.int64)
        self._tail_sources = np.empty(0, np.int64)

    def observe(self, trace, failure: dict | None, *, end_step: int, fresh: bool) -> dict | None:
        """Aggregate one addition, including preceding arrivals across a chained load boundary.

        Called for successful additions too, to preserve their short tail for the next word.
        Fresh simulators discard the previous tail. No per-addition edge arrays are retained.
        """
        ev = trace.events
        source = self.sources[ev["neuron"]] & (ev["node"] == 0)
        steps, neurons = ev["step"][source], ev["neuron"][source]
        if not fresh:
            steps = np.concatenate((self._tail_steps, steps))
            neurons = np.concatenate((self._tail_sources, neurons))
        result = None
        if failure is not None:
            delivered = np.zeros(len(self.pre), np.int64)
            stop = failure["step"]
            for delay in np.unique(self.delay[self.weight != 0]):
                lo, hi = stop - self.window_steps - int(delay), stop - int(delay)
                hits = np.bincount(neurons[(steps >= lo) & (steps < hi)], minlength=self.n)
                take = (self.delay == delay) & (self.weight != 0)
                delivered[take] = hits[self.pre[take]]
            score = delivered * np.abs(self.weight)
            self.score += score
            self.deliveries += delivered
            active = delivered > 0
            self.edge_failures += active
            self.failures += 1
            for column in self.labels:
                self.group_failures[column][np.unique(self.codes[column][active])] += 1
            result = {**failure, "active_edges": int(active.sum()),
                      "deliveries": int(delivered.sum()), "score": int(score.sum()),
                      "exc_score": int(score[self.weight > 0].sum()),
                      "inh_score": int(score[self.weight < 0].sum())}
        keep = steps >= end_step - self.window_steps - self.max_delay
        self._tail_steps, self._tail_sources = steps[keep], neurons[keep]
        return result

    def report(self, top_n: int = 50) -> dict:
        def totals(score):
            return {"score": int(score.sum()), "exc_score": int(score[self.weight > 0].sum()),
                    "inh_score": int(score[self.weight < 0].sum())}

        tables = {}
        for column, labels in self.labels.items():
            rows = []
            # Integer add.at keeps scores exact, unlike floating weighted bincount.
            grouped = np.zeros((len(labels), 3), np.int64)
            for j, take in enumerate((np.ones(len(self.score), bool), self.weight > 0, self.weight < 0)):
                np.add.at(grouped[:, j], self.codes[column][take], self.score[take])
            for i, label in enumerate(labels):
                rows.append({column: str(label), "score": int(grouped[i, 0]), "exc_score": int(grouped[i, 1]),
                             "inh_score": int(grouped[i, 2]), "failures": int(self.group_failures[column][i])})
            tables[column] = sorted(rows, key=lambda r: (-r["score"], r[column]))
        rank = np.lexsort((self.pre, self.post, -self.synapses, -self.score))
        top = []
        for i in rank[:top_n]:
            top.append({"pre_bodyId": int(self.pre_body[i]), "post_bodyId": int(self.post_body[i]),
                        "post_role": self.roles[self.post_designed[i]],
                        **{col: str(self.labels[col][self.codes[col][i]]) for col in self.labels},
                        "synapses": int(self.synapses[i]), "weight": int(self.weight[i]),
                        "score": int(self.score[i]), "exc_score": int(self.score[i]) if self.weight[i] > 0 else 0,
                        "inh_score": int(self.score[i]) if self.weight[i] < 0 else 0,
                        "deliveries": int(self.deliveries[i]), "failures": int(self.edge_failures[i])})
        return {"window_ms": self.window_ms, "window_steps": self.window_steps,
                "arrival_interval": "[failure_step - window_steps, failure_step)",
                "weight_units": "signed embedded quanta", "failures": self.failures,
                "incoming_edges": len(self.score), "scored_edges": int((self.score > 0).sum()),
                **totals(self.score), "top_edges": top, "by_type": tables["type"],
                "by_superclass": tables["superclass"]}

    def save(self, path: Path) -> None:
        """Compressed arrays and authoritative rule values; exact path, no implicit suffix."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as out:
            np.savez_compressed(out, pre_bodyId=self.pre_body, post_bodyId=self.post_body,
                                score=self.score, weight=self.weight, synapses=self.synapses,
                                deliveries=self.deliveries, failures=self.edge_failures,
                                model_rules=json.dumps(asdict(self.rules)))
