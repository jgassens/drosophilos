"""Finite-mission TMR availability model, not a neural simulator or runtime voter.

One copy means a logical service containing three tick-kernel replicas. An active
replica stalls with probability h[t] at tick t. Two unavailable replicas fail the
service immediately; one is masked. A stall at t returns before t + R, i.e. R=1
still cannot mask two stalls in the SAME tick. None means no recovery. Recovery
failure leaves that replica permanently unavailable. All healthy outputs are
assumed correct; see docs/stage_d_tmr.md for the fault-containment assumptions.

The analytic solver propagates a finite-state distribution. Monte Carlo samples
individual replica event times, repairs, and whole campaigns independently of
that recurrence. Both support time-varying hazards and explicit common modes.

    python -m drosophilos.bench.tmr_model --recovery-ticks 2 --trials 20000
    python -m drosophilos.bench.tmr_model --profile observed --recovery-ticks none
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Model:
    ticks: int = 1000
    copies: int = 100
    hazard: Rate = 5e-5
    recovery_ticks: int | None = 2
    recovery_failure: float = 0.0
    # Residual FATAL probabilities per logical group per tick, after protection.
    # voter_failure includes commit controller/log. watchdog_failure is already
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
        for name in ("hazard", "voter_failure", "watchdog_failure", "common_mode", "global_common_mode"):
            object.__setattr__(self, name, _rates(name, getattr(self, name), self.ticks))

    def fatal_hazard(self) -> np.ndarray:
        """Independent group-level fatal events combined without cancellation."""
        return -np.expm1(sum(_log_survival(x) for x in
                            (self.voter_failure, self.watchdog_failure, self.common_mode)))


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


def analytic(model: Model) -> dict:
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
    global_log = float(_log_survival(model.global_common_mode).sum())
    return {
        "group_failure": float(failed),
        "campaign_failure": campaign_failure(float(failed), model.copies, global_log),
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
    stalls = _EventSampler(model.hazard)
    fatal = _EventSampler(model.fatal_hazard())
    global_shocks = _EventSampler(model.global_common_mode)
    failures = group_failures = 0
    for offset in range(0, trials, batch_size):
        count = min(batch_size, trials - offset)
        groups = count * model.copies
        next_stall = stalls.draw(np.zeros((groups, 3), dtype=int), rng)
        failed_at = fatal.draw(np.zeros(groups, dtype=int), rng)
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
            next_stall[active[renew], lane[active[renew]]] = stalls.draw(back[renew], rng)
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--hazard", type=float, default=5e-5)
    parser.add_argument("--profile", choices=("constant", "observed"), default="constant")
    parser.add_argument("--recovery-ticks", default="2", help="lost service ticks including failure tick, or none")
    parser.add_argument("--recovery-failure", type=float, default=0.0)
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
        r = None if args.recovery_ticks.lower() == "none" else int(args.recovery_ticks)
        model = Model(ticks=args.ticks, copies=args.copies, hazard=h, recovery_ticks=r,
                      recovery_failure=args.recovery_failure, voter_failure=args.voter_failure,
                      watchdog_failure=args.watchdog_failure, common_mode=args.common_mode,
                      global_common_mode=args.global_common_mode)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    exact = analytic(model)
    exact.pop("group_failure_by_tick")
    report = {
        "assumption": "independent replica stalls; healthy results correct; repair restores readiness",
        "parameters": vars(args),
        "unreplicated_stalls_only": unreplicated_failure(model.hazard, ticks=model.ticks, copies=model.copies),
        "analytic": exact,
        "monte_carlo": monte_carlo(model, trials=args.trials, seed=args.seed) if args.trials else None,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
