"""Finite-mission TMR availability model, not a neural simulator or runtime voter.

One copy means a logical service containing three tick-kernel replicas. An active
replica stalls with probability h[t] at tick t. Two unavailable replicas fail the
service immediately; one is masked. A stall at t returns before t + R, i.e. R=1
still cannot mask two stalls in the SAME tick. None means no recovery. Recovery
failure leaves that replica permanently unavailable. All healthy outputs are
assumed correct. Three additional controller/log lanes are unrepaired; a static
single writer can fail liveness. Per-copy frailty draws persist through repair.
The default perfect-protection/repair result is a conditional floor, not a
forecast; see docs/stage_d_tmr.md for the fault-containment assumptions.

The analytic solver propagates a finite-state distribution. Monte Carlo samples
individual replica event times, repairs, and whole campaigns independently of
that recurrence. Both support time-varying hazards and explicit common modes.

    python -m drosophilos.bench.tmr_model --recovery-ticks 2 --trials 20000
    python -m drosophilos.bench.tmr_model --profile observed --recovery-ticks none
    python -m drosophilos.bench.tmr_model --sensitivity
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from numbers import Integral
from typing import Sequence

import numpy as np

Rate = float | Sequence[float] | np.ndarray

# docs/stage_d.md section 7 / perf_campaign.md ledger. These are completed-tick
# counts before the first stall, hence zero-based indices of the failed attempt.
# Same seed, different builds: pooled only as an illustrative timing sensitivity.
BASE_STALL_TICKS = (1, 31, 323, 728)
RATE_ROBUST_STALL_TICKS = (450, 535, 599, 654, 864)


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _rates(name: str, value: Rate, ticks: int) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(ticks, float(a))
    if a.shape != (ticks,) or not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
        raise ValueError(f"{name} must be a probability or {ticks} probabilities in [0, 1]")
    a = a.copy()
    a.flags.writeable = False
    return a


def _log_survival(hazard: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log1p(-hazard)


def _first_stall_probability(hazard: np.ndarray) -> float:
    """Marginal chance that a lane stalls at least once during the mission."""
    return -math.expm1(float(_log_survival(hazard).sum()))


@dataclass(frozen=True)
class Model:
    ticks: int = 1000
    copies: int = 100
    hazard: Rate = 5e-5
    recovery_ticks: int | None = 2
    recovery_failure: float = 0.0
    # Three controller/voter/log lanes, with NO repair in this conservative model.
    controller_hazard: Rate = 0.0
    writer_hazard: Rate = 0.0  # static writer: one stall fails service liveness
    # One independent, persistent draw per kernel replica, not per tick/repair.
    # Fraction q has elevated hazard, others zero; preserve marginal first-stall
    # mission probability, not mean instantaneous hazard. q=1 is homogeneous.
    frailty_fraction: float = 1.0
    post_rejoin_multiplier: float = 1.0  # intensity multiplier after first repair
    # Residual FATAL probabilities per logical group per tick, after protection.
    # voter_failure excludes raw controller/writer stalls. watchdog_failure is already
    # demand-weighted, not the failure probability of an individual timer.
    voter_failure: Rate = 0.0
    watchdog_failure: Rate = 0.0
    common_mode: Rate = 0.0  # shock affecting >=2 replicas within one group
    global_common_mode: Rate = 0.0  # one shock shared by ALL copies in a campaign

    def __post_init__(self) -> None:
        _integer("ticks", self.ticks, 1)
        _integer("copies", self.copies, 1)
        if self.recovery_ticks is not None:
            _integer("recovery_ticks", self.recovery_ticks, 1)
        r = np.asarray(self.recovery_failure, dtype=float)
        if r.ndim != 0 or not np.isfinite(r) or not 0 <= r <= 1:
            raise ValueError("recovery_failure must be a probability in [0, 1]")
        object.__setattr__(self, "recovery_failure", float(r))
        q = self.frailty_fraction
        if not np.isfinite(q) or not 0 < q <= 1:
            raise ValueError("frailty_fraction must be in (0, 1]")
        m = self.post_rejoin_multiplier
        if not np.isfinite(m) or m < 0:
            raise ValueError("post_rejoin_multiplier must be finite and >= 0")
        for name in ("hazard", "controller_hazard", "writer_hazard", "voter_failure",
                     "watchdog_failure", "common_mode", "global_common_mode"):
            object.__setattr__(self, name, _rates(name, getattr(self, name), self.ticks))
        self.kernel_hazard()  # validate feasibility of marginal-preserving mixture

    def kernel_hazard(self, *, rejoined: bool = False) -> np.ndarray:
        """Hazard of a susceptible lane; its susceptibility survives every reset."""
        intensity = -_log_survival(self.hazard)
        p = _first_stall_probability(self.hazard)
        q = self.frailty_fraction
        if q < 1:
            if p >= q:
                raise ValueError("frailty_fraction must exceed marginal first-stall probability")
            if p:
                intensity = intensity * (-math.log1p(-p / q) / float(intensity.sum()))
        if rejoined:
            if self.post_rejoin_multiplier == 0:
                return np.zeros(self.ticks)
            intensity = intensity * self.post_rejoin_multiplier
        return -np.expm1(-intensity)

    def fatal_hazard(self) -> np.ndarray:
        """Independent group-level fatal events combined without cancellation."""
        return -np.expm1(sum(_log_survival(x) for x in
                            (self.writer_hazard, self.voter_failure,
                             self.watchdog_failure, self.common_mode)))


def campaign_failure(group_failure: float, copies: int, global_log_survival: float = 0.0) -> float:
    """At least one failed logical copy, plus a campaign-wide common shock."""
    _integer("copies", copies, 1)
    if not math.isfinite(group_failure) or not 0 <= group_failure <= 1:
        raise ValueError("group_failure must be in [0, 1]")
    if math.isnan(global_log_survival) or global_log_survival > 0:
        raise ValueError("global_log_survival must be <= 0")
    if group_failure == 1:
        return 1.0
    return -math.expm1(copies * math.log1p(-group_failure) + global_log_survival)


def unreplicated_failure(hazard: Rate = 5e-5, *, ticks: int = 1000, copies: int = 100) -> float:
    """Stalls only: no replication, no recovery, any missed tick fails the exit."""
    _integer("ticks", ticks, 1)
    _integer("copies", copies, 1)
    return -math.expm1(copies * float(_log_survival(_rates("hazard", hazard, ticks)).sum()))


def _homogeneous_curve(model: Model) -> np.ndarray:
    """Exact discrete-time recurrence; failure is absorbing, not just end downtime.

    State is all healthy, one permanently down, or one down with k more service
    ticks before repair. Repair is applied at the END of a tick, after its failure
    opportunities. Protection shocks then absorb surviving probability mass.
    A repair still pending after the last requested tick does not fail the exit.
    """
    healthy, permanent, failed = 1.0, 0.0, 0.0
    r = model.recovery_ticks
    # An R >= horizon cannot affect service within this mission.
    if r is not None and r >= model.ticks:
        r = None
    pending = np.zeros(0 if r is None else r - 1)
    cdf = np.empty(model.ticks)
    for t, (h, fatal) in enumerate(zip(model.hazard, model.fatal_hazard())):
        live2 = (1 - h) ** 2
        degraded = permanent + float(pending.sum())
        lost = healthy * (3 * h * h - 2 * h**3) + degraded * (2 * h - h * h)
        new_stall = healthy * 3 * h * live2
        next_healthy = healthy * (1 - h) ** 3
        next_permanent = permanent * live2
        if r is None:
            next_permanent += new_stall
        else:
            repairing = new_stall if r == 1 else pending[0] * live2
            next_healthy += repairing * (1 - model.recovery_failure)
            next_permanent += repairing * model.recovery_failure
            if r > 1:
                pending[:-1] = pending[1:] * live2
                pending[-1] = new_stall
        surviving = next_healthy + next_permanent + float(pending.sum())
        failed = min(1.0, failed + lost + surviving * fatal)
        healthy = next_healthy * (1 - fatal)
        permanent = next_permanent * (1 - fatal)
        pending *= 1 - fatal
        cdf[t] = failed
    return cdf


def _lane_curve(model: Model, susceptible: tuple[int, int, int]) -> np.ndarray:
    """Identity-preserving chain: rejoined mask, unavailable lane and countdown.

    Unlike a mean-hazard recurrence, this retains the same frailty draw when a
    lane returns. Only one unavailable lane can exist in a surviving service.
    """
    r = model.recovery_ticks
    if r is not None and r >= model.ticks:
        r = None
    healthy = np.zeros(8)
    healthy[0] = 1
    permanent = np.zeros((8, 3))
    pending = np.zeros((8, 3, 0 if r is None else r - 1))
    bits = (np.arange(8)[:, None] & (1 << np.arange(3))) != 0
    base, renewed = model.kernel_hazard(), model.kernel_hazard(rejoined=True)
    cdf = np.zeros(model.ticks)
    failed = 0.0
    for t, fatal in enumerate(model.fatal_hazard()):
        h = np.where(bits, renewed[t], base[t]) * susceptible
        live = 1 - h
        live3 = live.prod(axis=1)
        live2 = np.stack([live[:, [j for j in range(3) if j != lane]].prod(axis=1)
                          for lane in range(3)], axis=1)
        one = h * live2
        # Explicit products retain tiny double-stall probabilities.
        multi = (h[:, 0] * h[:, 1] * live[:, 2] + h[:, 0] * h[:, 2] * live[:, 1]
                 + h[:, 1] * h[:, 2] * live[:, 0] + h.prod(axis=1))
        lost2 = np.stack([h[:, j] + h[:, k] - h[:, j] * h[:, k]
                         for j, k in ((1, 2), (0, 2), (0, 1))], axis=1)
        degraded = permanent + pending.sum(axis=2)
        lost = float(healthy @ multi + (degraded * lost2).sum())
        new = healthy[:, None] * one
        next_healthy = healthy * live3
        next_permanent = permanent * live2
        if r is None:
            next_permanent += new
        else:
            repairing = new if r == 1 else pending[:, :, 0] * live2
            next_permanent += repairing * model.recovery_failure
            for mask in range(8):
                for lane in range(3):
                    next_healthy[mask | (1 << lane)] += repairing[mask, lane] * (1 - model.recovery_failure)
            if r > 1:
                pending[:, :, :-1] = pending[:, :, 1:] * live2[:, :, None]
                pending[:, :, -1] = new
        surviving = float(next_healthy.sum() + next_permanent.sum() + pending.sum())
        failed = min(1.0, failed + lost + surviving * fatal)
        healthy = next_healthy * (1 - fatal)
        permanent = next_permanent * (1 - fatal)
        pending *= 1 - fatal
        cdf[t] = failed
    return cdf


def analytic(model: Model) -> dict:
    """Exact kernel frailty mixture × unrepaired controller majority survival.

    Static writer and residual protection hazards are service-fatal. Frailty
    draws are independent across physical kernel copies, retained after repair.
    Controller draws are homogeneous and independent of the kernel/writer.
    """
    q = model.frailty_fraction
    if q == 1 and model.post_rejoin_multiplier == 1:
        cdf = _homogeneous_curve(model)
    else:
        cdf = np.zeros(model.ticks)
        # Symmetric lane permutations: solve four configurations, not eight.
        for count in range(4) if q < 1 else (3,):
            weight = math.comb(3, count) * q**count * (1 - q)**(3 - count)
            if weight:
                cdf += weight * _lane_curve(model, tuple(int(i < count) for i in range(3)))
    p_controller = -np.expm1(np.cumsum(_log_survival(model.controller_hazard)))
    controller_cdf = p_controller**2 * (3 - 2 * p_controller)
    cdf = -np.expm1(_log_survival(cdf) + _log_survival(controller_cdf))
    failed = float(cdf[-1])
    global_log = float(_log_survival(model.global_common_mode).sum())
    return {
        "group_failure": failed,
        "controller_group_failure": float(controller_cdf[-1]),
        "campaign_failure": campaign_failure(failed, model.copies, global_log),
        "group_failure_by_tick": cdf,
    }


class _EventSampler:
    """Sample the first Bernoulli event at or after any array of start ticks.

    The integrated hazard turns an exponential draw into an event index. Certain
    events are handled separately to avoid inf-inf on a restart after h[t]=1.
    This skips empty time intervals; it does not approximate Bernoulli draws by
    a continuous-time Poisson process.
    """

    def __init__(self, hazard: np.ndarray):
        self.ticks = len(hazard)
        certain = hazard == 1
        finite = np.where(certain, 0.0, hazard)
        self.prefix = np.r_[0.0, np.cumsum(-_log_survival(finite))]
        indices = np.where(certain, np.arange(self.ticks), self.ticks)
        self.next_certain = np.r_[np.minimum.accumulate(indices[::-1])[::-1], self.ticks]

    def draw(self, starts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        starts = np.minimum(starts, self.ticks)
        target = self.prefix[starts] + rng.exponential(size=starts.shape)
        times = np.searchsorted(self.prefix[1:], target, side="right")
        return np.minimum(times, self.next_certain[starts])


def _wilson(failures: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = failures / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def monte_carlo(model: Model, *, trials: int = 20000, seed: int = 108,
                batch_size: int = 1000) -> dict:
    """Sample full campaigns (trials x copies x three replicas), with bounded memory.

    The next two replica events overlap iff second < first + R. Equality means
    repair has finished. Failed repairs remain down; active replicas can fail
    repeatedly after successful repair. Global shocks are sampled ONCE per
    campaign, not independently for each copy. CI is a binomial Wilson interval
    for campaign failure and cannot measure a 1e-10 residual with small samples.
    """
    _integer("trials", trials, 1)
    _integer("batch_size", batch_size, 1)
    rng = np.random.default_rng(seed)
    stalls = _EventSampler(model.kernel_hazard())
    renewed_stalls = _EventSampler(model.kernel_hazard(rejoined=True))
    controllers = _EventSampler(model.controller_hazard)
    fatal = _EventSampler(model.fatal_hazard())
    global_shocks = _EventSampler(model.global_common_mode)
    failures = group_failures = 0
    for offset in range(0, trials, batch_size):
        count = min(batch_size, trials - offset)
        groups = count * model.copies
        susceptible = (rng.random((groups, 3)) < model.frailty_fraction
                       if model.frailty_fraction < 1 else np.ones((groups, 3), dtype=bool))
        next_stall = stalls.draw(np.zeros((groups, 3), dtype=int), rng)
        next_stall[~susceptible] = model.ticks
        failed_at = fatal.draw(np.zeros(groups, dtype=int), rng)
        if np.any(model.controller_hazard):
            controller_events = controllers.draw(np.zeros((groups, 3), dtype=int), rng)
            failed_at = np.minimum(failed_at, np.partition(controller_events, 1, axis=1)[:, 1])
        while True:
            lane = np.argmin(next_stall, axis=1)
            first = next_stall[np.arange(groups), lane]
            active = np.flatnonzero((first < model.ticks) & (first < failed_at))
            if not len(active):
                break
            second = np.partition(next_stall[active], 1, axis=1)[:, 1]
            if model.recovery_ticks is None:
                back = np.full(len(active), model.ticks)
            else:
                back = np.minimum(first[active] + min(model.recovery_ticks, model.ticks), model.ticks)
                unsuccessful = rng.random(len(active)) < model.recovery_failure
                back[unsuccessful] = model.ticks
            overlap = second < back
            failed_at[active[overlap]] = np.minimum(failed_at[active[overlap]], second[overlap])
            # Retire the earliest event even for a group that has now failed.
            # failed_at prevents further renewal of that group.
            next_stall[active, lane[active]] = model.ticks
            renew = ~overlap & (back < failed_at[active])
            next_stall[active[renew], lane[active[renew]]] = renewed_stalls.draw(back[renew], rng)
        group_failed = failed_at < model.ticks
        group_failures += int(group_failed.sum())
        global_failed = global_shocks.draw(np.zeros(count, dtype=int), rng) < model.ticks
        failures += int(np.count_nonzero(group_failed.reshape(count, model.copies).any(axis=1) | global_failed))
    p = failures / trials
    return {
        "trials": trials,
        "seed": seed,
        "batch_size": batch_size,
        "failures": failures,
        "group_failure": group_failures / (trials * model.copies),
        "campaign_failure": p,
        "standard_error": math.sqrt(p * (1 - p) / trials),
        "confidence_95": _wilson(failures, trials),
    }


def binned_hazard(stall_ticks: Sequence[int], *, copies: int, ticks: int,
                  edges: Sequence[int], normalize_to: float | None = None) -> np.ndarray:
    """First-stall histogram divided by at-risk replica-ticks in each bin.

    Each event is a distinct replica's first failed attempt (zero-based); other
    replicas are right-censored after ticks. A failing attempt contributes one
    exposure. Normalization preserves the mission survival of a constant hazard
    normalize_to, so comparisons isolate timing rather than total failure risk.
    No sampling uncertainty or within-replica frailty is inferred from the bins.
    """
    _integer("copies", copies, 1)
    _integer("ticks", ticks, 1)
    for e in edges:
        _integer("bin edge", e, 0)
    for t in stall_ticks:
        _integer("stall tick", t, 0)
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != ticks or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("edges must increase strictly from 0 to ticks")
    if len(stall_ticks) > copies or any(t >= ticks for t in stall_ticks):
        raise ValueError("need at most one first stall per copy, within the mission")
    observed = np.asarray(stall_ticks, dtype=int)
    h = np.zeros(ticks)
    for a, b in zip(edges, edges[1:]):
        exposure = (copies - len(observed)) * (b - a) + np.clip(observed - a + 1, 0, b - a).sum()
        events = np.count_nonzero((observed >= a) & (observed < b))
        if exposure == 0:
            raise ValueError("bin has no at-risk exposure")
        h[a:b] = events / exposure
    if normalize_to is not None:
        target = _rates("normalize_to", normalize_to, 1)[0]
        if target == 1:
            raise ValueError("normalize_to must be < 1")
        intensity = -_log_survival(h)
        total = float(intensity.sum())
        if total == 0 or not math.isfinite(total):
            raise ValueError("normalization needs nonzero, finite observed cumulative hazard")
        h = -np.expm1(intensity * (ticks * math.log1p(-target) / total))
    return h


def observed_hazard(normalize_to: float | None = 5e-5) -> np.ndarray:
    """Illustrative pooled shape; the two historical builds are not iid cohorts."""
    return binned_hazard(BASE_STALL_TICKS + RATE_ROBUST_STALL_TICKS,
                         copies=200, ticks=1000, edges=(0, 250, 500, 750, 1000),
                         normalize_to=normalize_to)


def cadence_hazard(hazard: Rate, *, source_seconds: float = 5.84,
                   tick_seconds: float = 13.0, time_fraction: float = 1.0) -> np.ndarray:
    """Split integrated hazard into transaction-driven and elapsed-time terms.

    alpha=0 preserves hazard per transaction; alpha=1 scales all exposure with
    cadence, including idle holds. The split is an assumption, not a fit.
    """
    if (not math.isfinite(source_seconds) or source_seconds <= 0
            or not math.isfinite(tick_seconds) or tick_seconds <= 0
            or not math.isfinite(time_fraction) or not 0 <= time_fraction <= 1):
        raise ValueError("positive finite cadence and time_fraction in [0, 1] required")
    h = np.asarray(hazard, dtype=float)
    if not np.all(np.isfinite(h)) or np.any((h < 0) | (h > 1)):
        raise ValueError("hazard must be in [0, 1]")
    scale = 1 - time_fraction + time_fraction * tick_seconds / source_seconds
    return -np.expm1(_log_survival(h) * scale)


def first_stall_estimate(stalls: int, *, copies: int = 100, ticks: int = 1000) -> dict:
    """Campaign first-stall fraction and Wilson 95% interval mapped to constant h.

    Assumes independent copies, one fixed build, and a constant active hazard.
    This does not pool different builds/seeds or infer confidence in a TMR fit.
    """
    _integer("copies", copies, 1)
    _integer("ticks", ticks, 1)
    _integer("stalls", stalls, 0)
    if stalls > copies:
        raise ValueError("stalls must not exceed copies")

    def to_h(p: float) -> float:
        return 1.0 if p == 1 else -math.expm1(math.log1p(-p) / ticks)

    interval = _wilson(stalls, copies)
    return {"mission_failure": stalls / copies, "confidence_95": interval,
            "hazard": to_h(stalls / copies),
            "hazard_confidence_95": tuple(to_h(p) for p in interval)}


def binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """P[X <= successes] for a binomial variate, without a scipy dependency."""
    _integer("trials", trials, 1)
    _integer("successes", successes, 0)
    if successes >= trials:
        return 1.0
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if probability == 0:
        return 1.0
    if probability == 1:
        return 0.0
    term = (1 - probability) ** trials
    total = term
    for observed in range(successes):
        term *= (trials - observed) * probability / ((observed + 1) * (1 - probability))
        total += term
    return min(1.0, total)


def campaign_gate(*, trials: int = 300, target_failure: float = 0.03,
                  floor_failure: float | None = None) -> dict:
    """Exact one-sided gate and floor pass chance for independent campaigns.

    The acceptance count is the largest count whose chance at the target rate is
    at most 5%, so accepting it is a one-sided 95% test of the stated target.
    """
    _integer("trials", trials, 1)
    if not math.isfinite(target_failure) or not 0 < target_failure < 1:
        raise ValueError("target_failure must be in (0, 1)")
    if floor_failure is None:
        floor_failure = analytic(Model())["campaign_failure"]
    if not math.isfinite(floor_failure) or not 0 <= floor_failure <= 1:
        raise ValueError("floor_failure must be in [0, 1]")
    acceptance = max(k for k in range(trials + 1)
                     if binomial_cdf(k, trials, target_failure) <= 0.05)
    return {"trials": trials, "target_failure": target_failure,
            "accept_at_most": acceptance,
            "target_false_accept_probability": binomial_cdf(
                acceptance, trials, target_failure),
            "floor_pass_probability": binomial_cdf(acceptance, trials, floor_failure)}


def sensitivity_table(base: Model | None = None) -> list[dict]:
    """Reproducible assumptions for the design table; all are scenarios, not fits.

    Infeasible frailty scenarios are reported as skipped rows rather than silently
    clamped: preserving a marginal first-stall probability p requires q > p.
    """
    scenarios = [
        ("Ideal R=2 floor at h=5e-5", {}),
        ("Low evidence envelope", {"hazard": 1.5e-5}),
        ("4/100 constant-h estimate", {"hazard": first_stall_estimate(4)["hazard"]}),
        ("5/100 constant-h estimate", {"hazard": first_stall_estimate(5)["hazard"]}),
        ("6/100 including zero_once", {"hazard": first_stall_estimate(6)["hazard"]}),
        ("9/100 old build", {"hazard": first_stall_estimate(9)["hazard"]}),
        ("Half time-driven, 13s", {"hazard": float(cadence_hazard(5e-5, time_fraction=0.5))}),
        ("Fully time-driven, 13s", {"hazard": float(cadence_hazard(5e-5))}),
        ("Wide cadence/evidence envelope", {"hazard": 3e-4}),
        ("1% failed repairs", {"recovery_failure": 0.01}),
        ("10% failed repairs", {"recovery_failure": 0.1}),
        ("No qualified repair: all fail", {"recovery_failure": 1.0}),
        ("Unrepaired controllers, low", {"controller_hazard": 5e-6}),
        ("Unrepaired controllers, illustrative", {"controller_hazard": 2.4e-5}),
        ("Unrepaired controllers, kernel-scale", {"controller_hazard": 5e-5}),
        ("Static writer, 1e-7", {"writer_hazard": 1e-7}),
        ("Static writer, 1e-6", {"writer_hazard": 1e-6}),
        ("Static writer, low kernel-scale", {"writer_hazard": 5e-6}),
        ("Static writer, illustrative kernel-scale", {"writer_hazard": 2.4e-5}),
        ("Static writer, kernel-scale", {"writer_hazard": 5e-5}),
        ("10% persistent susceptible lanes", {"frailty_fraction": 0.1}),
        ("5% persistent susceptible lanes", {"frailty_fraction": 0.05}),
        ("Post-rejoin intensity x2", {"post_rejoin_multiplier": 2.0}),
        ("Joint illustrative assumptions", {"hazard": float(cadence_hazard(5e-5)),
          "recovery_failure": 0.01, "controller_hazard": 2.4e-5,
          "writer_hazard": 2.4e-5, "frailty_fraction": 0.2}),
    ]
    base = Model() if base is None else base
    rows = []
    for label, kwargs in scenarios:
        hazard = _rates("hazard", kwargs.get("hazard", base.hazard), base.ticks)
        frailty_fraction = kwargs.get("frailty_fraction", base.frailty_fraction)
        p_first_stall = _first_stall_probability(hazard)
        row = {"scenario": label, "hazard": float(hazard[0]),
               "f_repair": kwargs.get("recovery_failure", base.recovery_failure),
               "controller_hazard": float(_rates(
                   "controller_hazard", kwargs.get("controller_hazard", base.controller_hazard),
                   base.ticks)[0]),
               "writer_hazard": float(_rates(
                   "writer_hazard", kwargs.get("writer_hazard", base.writer_hazard), base.ticks)[0]),
               "frailty_fraction": frailty_fraction,
               "post_rejoin_multiplier": kwargs.get("post_rejoin_multiplier",
                                                     base.post_rejoin_multiplier)}
        if frailty_fraction < 1 and p_first_stall >= frailty_fraction:
            row.update(campaign_failure=None,
                       note=("skipped: frailty_fraction must exceed marginal first-stall "
                             f"probability (q={frailty_fraction:g}, p={p_first_stall:g})"))
            rows.append(row)
            continue
        model = replace(base, **kwargs)
        row["campaign_failure"] = analytic(model)["campaign_failure"]
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--hazard", type=float, default=5e-5)
    parser.add_argument("--profile", choices=("constant", "observed"), default="constant")
    parser.add_argument("--recovery-ticks", default="2", help="lost service ticks including failure tick, or none")
    parser.add_argument("--recovery-failure", "--f-repair", type=float, default=0.0)
    parser.add_argument("--controller-hazard", type=float, default=0.0)
    parser.add_argument("--writer-hazard", type=float, default=0.0)
    parser.add_argument("--frailty-fraction", type=float, default=1.0)
    parser.add_argument("--post-rejoin-multiplier", type=float, default=1.0)
    parser.add_argument("--tick-seconds", type=float, default=5.84)
    parser.add_argument("--time-fraction", type=float, default=1.0)
    parser.add_argument("--sensitivity", action="store_true", help="print design sensitivity table")
    parser.add_argument("--voter-failure", type=float, default=0.0)
    parser.add_argument("--watchdog-failure", type=float, default=0.0)
    parser.add_argument("--common-mode", type=float, default=0.0)
    parser.add_argument("--global-common-mode", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=20000, help="whole Monte Carlo campaigns; 0 skips sampling")
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args(argv)
    try:
        _integer("trials", args.trials, 0)
        if args.profile == "observed" and args.ticks != 1000:
            raise ValueError("observed profile is defined only for 1000 ticks")
        h = observed_hazard(args.hazard) if args.profile == "observed" else args.hazard
        h = cadence_hazard(h, tick_seconds=args.tick_seconds, time_fraction=args.time_fraction)
        r = None if args.recovery_ticks.lower() == "none" else int(args.recovery_ticks)
        model = Model(ticks=args.ticks, copies=args.copies, hazard=h, recovery_ticks=r,
                      recovery_failure=args.recovery_failure, voter_failure=args.voter_failure,
                      controller_hazard=args.controller_hazard, writer_hazard=args.writer_hazard,
                      frailty_fraction=args.frailty_fraction,
                      post_rejoin_multiplier=args.post_rejoin_multiplier,
                      watchdog_failure=args.watchdog_failure, common_mode=args.common_mode,
                      global_common_mode=args.global_common_mode)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if args.sensitivity:
        print(json.dumps(sensitivity_table(model), indent=2, allow_nan=False))
        return
    exact = analytic(model)
    exact.pop("group_failure_by_tick")
    report = {
        "assumption": "sensitivity only; ideal case is a floor, not a forecast; static frailty; "
                      "healthy results correct; unrepaired controller lanes; static single writer",
        "parameters": vars(args),
        "effective_hazard": (float(model.hazard[0]) if np.all(model.hazard == model.hazard[0])
                             else model.hazard.tolist()),
        "unreplicated_stalls_only": unreplicated_failure(model.hazard, ticks=model.ticks, copies=model.copies),
        "analytic": exact,
        "monte_carlo": monte_carlo(model, trials=args.trials, seed=args.seed) if args.trials else None,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
