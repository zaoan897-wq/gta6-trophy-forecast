"""Forecast Exophase earn rates for GTA VI trophies at a future target time.

Approach
--------
A trophy's printed rate is earners / players_tracked.  Both parts move:
Exophase refreshes the page in steps, so players_tracked P(t) jumps while
the earners e_i(t) of each trophy grow as tracked players progress through
the game.  This program therefore forecasts the two parts separately and
also fits the rate directly, then lets the case's own history judge which
view has been most trustworthy lately:

  * "last"    - the rate stays at its most recent read;
  * "trend"   - logit(rate) moves linearly in ln(1 + hours since
                release); the slope blends a whole-history fit with a
                recent-window fit and is shrunk toward the trophy's
                tier, and the forecast is anchored at the last read so
                level misfit cannot drag it;
  * "earners" - players_tracked keeps growing at its recent pace with
                the growth rate decaying over the horizon, and each
                trophy's earners follow a power law in P whose
                elasticity is shrunk toward its tier's mean.

The history is first compressed to distinct snapshots (the page sits
unchanged between refreshes, so plateaus must not dominate the fits).
Candidates are scored by an internal backtest: fit on an early part of
the snapshots, predict the final snapshot, and measure the mean pinball
loss on the logit scale at the case's five levels.  Because a backtest
only ever predicts a fraction of the span ahead while the real horizon
is usually longer, the backtest weights are mixed toward an even prior
as the horizon outgrows the span.  The final median is the weighted
mixture of the candidates (in logit space) and the final spread
combines each candidate's own spread with the mixture's disagreement
and a floor derived from backtest residuals.  A history with a single
distinct snapshot cannot be backtested; then the last rate is kept
with a deliberately wide spread, because a refresh the history has
never shown can move every rate at once.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
RATE_LO = 0.01
RATE_HI = 99.99


# ---------------------------------------------------------------- helpers

def clamp_rate(x):
    return min(RATE_HI, max(RATE_LO, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, resid_std, mean_x, mean_y); with a shrunk slope the
    intercept can be refit as mean_y - b * mean_x so the line still
    passes through the data's centre of mass.
    """
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        var = sum(w * (y - my) ** 2 for y, w in zip(ys, ws)) / sw
        return my, 0.0, math.sqrt(max(var, 0.0)), mx, my
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    var = sum(w * (y - a - b * x) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return a, b, math.sqrt(max(var, 0.0)), mx, my


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


# ------------------------------------------------------- history handling

class Snapshot:
    __slots__ = ("hours", "u", "players", "rates")

    def __init__(self, hours, players, rates):
        self.hours = hours            # hours since release midnight (may be < 0)
        self.u = math.log1p(max(hours, -0.99))
        self.players = players
        self.rates = rates


def compress_history(case, release_dt):
    """Collapse runs of identical reads into single snapshots.

    Between Exophase refreshes the page is byte-identical, so keeping
    every read would let a long plateau outweigh the moments that move.
    """
    snaps = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        at = parse_time(read["at"])
        hours = (at - release_dt).total_seconds() / 3600.0
        snaps.append(Snapshot(hours, max(1, read["players_tracked"]),
                              list(read["rates"])))
    return snaps


def snap_weights(snaps):
    """Each snapshot stands for the time until the next one appears."""
    if len(snaps) == 1:
        return [1.0]
    gaps = [snaps[i + 1].hours - snaps[i].hours for i in range(len(snaps) - 1)]
    gaps.append(median_of(gaps) if gaps else 1.0)
    return [max(g, 1e-3) for g in gaps]


def last_known_rates(snaps, n):
    """Most recent non-null rate per trophy, scanning backwards."""
    out = [None] * n
    for snap in reversed(snaps):
        for i in range(n):
            if out[i] is None and snap.rates[i] is not None:
                out[i] = snap.rates[i]
        if all(v is not None for v in out):
            break
    return out


def tier_groups(trophies):
    groups = {}
    for i, t in enumerate(trophies):
        groups.setdefault(t["tier"], []).append(i)
    return groups


def shrink(values, group_of, group_means, counts, k=4.0):
    """Pull each value toward its group mean; sparse data shrinks more."""
    out = []
    for i, v in enumerate(values):
        g = group_means[group_of[i]]
        n = counts[i]
        out.append((n * v + k * g) / (n + k) if n > 0 else g)
    return out


# ------------------------------------------------------------- candidates
# Each candidate maps (snaps, weights, trophies, target_u, horizon_days)
# to (median_logit list, sigma list).  All are pure and deterministic.

def cand_last(snaps, weights, trophies, target_u, horizon_days, vol):
    n = len(trophies)
    last = last_known_rates(snaps, n)
    s = min(2.0, max(0.35, vol * math.sqrt(max(horizon_days, 0.01))))
    med = [logit(r if r is not None else 1.0) for r in last]
    return med, [s] * n


def cand_trend(snaps, weights, trophies, target_u, horizon_days, vol):
    n = len(trophies)
    groups = tier_groups(trophies)
    group_of = [t["tier"] for t in trophies]
    last = last_known_rates(snaps, n)
    u_last = snaps[-1].u
    recent = max(3, (len(snaps) + 1) // 2)
    slopes, slopes_loc, means, resid, counts = [], [], [], [], []
    for i in range(n):
        xs, ys, ws = [], [], []
        for snap, w in zip(snaps, weights):
            r = snap.rates[i]
            if r is not None:
                xs.append(snap.u)
                ys.append(logit(r))
                ws.append(w)
        counts.append(len(xs))
        if len(xs) >= 2:
            _, b, sd, mx, my = wls(xs, ys, ws)
            _, b_loc, _, _, _ = wls(xs[-recent:], ys[-recent:], ws[-recent:])
        elif xs:
            b, b_loc, sd, mx, my = 0.0, 0.0, 0.5, xs[0], ys[0]
        else:
            b, b_loc, sd, mx, my = 0.0, 0.0, 0.5, 0.0, logit(1.0)
        slopes.append(b)
        slopes_loc.append(b_loc)
        means.append((mx, my))
        resid.append(sd)
    tier_slope = {g: median_of([0.5 * (slopes[i] + slopes_loc[i])
                                for i in idx]) for g, idx in groups.items()}
    tier_resid = {g: median_of([resid[i] for i in idx]) for g, idx in groups.items()}
    blended = [0.5 * (slopes[i] + slopes_loc[i]) for i in range(n)]
    blended = shrink(blended, group_of, tier_slope, counts)
    resid = shrink(resid, group_of, tier_resid, counts)
    extra = max(0.0, target_u - u_last)
    med, sig = [], []
    for i in range(n):
        mx, my = means[i]
        line_at_target = (my - blended[i] * mx) + blended[i] * target_u
        anchored = logit(last[i] if last[i] is not None else 1.0) \
            + blended[i] * extra
        med.append(0.55 * anchored + 0.45 * line_at_target
                   if counts[i] else logit(1.0))
        sig.append(min(2.2, max(0.3, resid[i] * (1.0 + 0.35 * extra))))
    return med, sig


def forecast_players(snaps, weights, target_hours):
    """Forecast players_tracked at the target, in log space.

    Two views are averaged: a line of ln P against ln(1 + hours), which
    damps itself, and the recent per-hour growth rate carried forward
    with exponential decay (a growth spurt cannot run for ever, but it
    does not stop at the checkpoint either).  The result never falls
    below the last count and never exceeds 25x it.
    """
    last = snaps[-1]
    ln_last = math.log(last.players)
    a, b, _, _, _ = wls([s.u for s in snaps],
                        [math.log(s.players) for s in snaps], weights)
    target_u = math.log1p(max(target_hours, -0.99))
    view_line = a + b * target_u
    recent = snaps[-(max(3, (len(snaps) + 1) // 2)):]
    if len(recent) >= 2 and recent[-1].hours > recent[0].hours:
        g = (math.log(recent[-1].players) - math.log(recent[0].players)) \
            / (recent[-1].hours - recent[0].hours)
    else:
        g = 0.0
    g = min(max(g, 0.0), 0.05)
    dh = max(target_hours - last.hours, 0.0)
    tau = 60.0
    view_local = ln_last + g * tau * (1.0 - math.exp(-dh / tau))
    ln_star = 0.5 * (view_line + view_local)
    ln_star = min(max(ln_star, ln_last), ln_last + math.log(25.0))
    return ln_star


def cand_earners(snaps, weights, trophies, target_u, horizon_days, vol):
    n = len(trophies)
    groups = tier_groups(trophies)
    group_of = [t["tier"] for t in trophies]
    target_hours = math.expm1(target_u)
    ln_p_star = forecast_players(snaps, weights, target_hours)
    p_star = math.exp(ln_p_star)
    last = last_known_rates(snaps, n)
    elas, means, resid, counts = [], [], [], []
    for i in range(n):
        xs, ys, ws = [], [], []
        for snap, w in zip(snaps, weights):
            r = snap.rates[i]
            if r is not None:
                xs.append(math.log(snap.players))
                ys.append(math.log(r / 100.0 * snap.players + 0.5))
                ws.append(w)
        counts.append(len(xs))
        if len(xs) >= 2:
            _, b, sd, mx, my = wls(xs, ys, ws)
        elif xs:
            b, sd, mx, my = 1.0, 0.5, xs[0], ys[0]
        else:
            b, sd, mx, my = 1.0, 0.5, 0.0, 0.0
        means.append((mx, my))
        elas.append(b)
        resid.append(sd)
    tier_e = {g: median_of([elas[i] for i in idx]) for g, idx in groups.items()}
    tier_r = {g: median_of([resid[i] for i in idx]) for g, idx in groups.items()}
    elas = shrink(elas, group_of, tier_e, counts)
    resid = shrink(resid, group_of, tier_r, counts)
    med, sig = [], []
    for i in range(n):
        # Refit the intercept through the data's centre of mass: with a
        # shrunk slope, the raw intercept (the line's value at ln P = 0,
        # far outside the data) would swing the forecast wildly.
        mx, my = means[i]
        ln_e = (my - elas[i] * mx) + elas[i] * ln_p_star
        e_star = min(max(math.exp(ln_e) - 0.5, 0.0), p_star)
        rate = 100.0 * e_star / p_star if p_star > 0 else 0.0
        if counts[i] == 0:
            rate = last[i] if last[i] is not None else 1.0
        med.append(logit(rate))
        sig.append(min(2.2, max(0.3, resid[i])))
    return med, sig


CANDIDATES = (("last", cand_last), ("trend", cand_trend),
              ("earners", cand_earners))


# -------------------------------------------------------------- backtest

def pinball_mean(pred_med, pred_sig, actual_logit, levels):
    total, cnt = 0.0, 0
    for m, s, y in zip(pred_med, pred_sig, actual_logit):
        if y is None:
            continue
        for tau in levels:
            q = m + norm_ppf(tau) * s
            d = y - q
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            cnt += 1
    return total / cnt if cnt else None


def logit_volatility(snaps):
    """Pooled RMS logit move per sqrt(day) between consecutive snapshots."""
    acc, cnt = [], 0
    for a, b in zip(snaps, snaps[1:]):
        gap_days = max((b.hours - a.hours) / 24.0, 1e-4)
        for ra, rb in zip(a.rates, b.rates):
            if ra is not None and rb is not None:
                acc.append(((logit(rb) - logit(ra)) ** 2) / gap_days)
                cnt += 1
    if cnt == 0:
        return 0.5
    return math.sqrt(sum(acc) / cnt)


def backtest(snaps, weights, trophies, levels):
    """Score each candidate by predicting the final snapshot from less.

    Returns (weights per candidate, pooled RMS logit error of the blend).
    """
    m = len(snaps)
    span = snaps[-1].hours - snaps[0].hours
    actual = [logit(r) if r is not None else None for r in snaps[-1].rates]
    scores = {name: [] for name, _ in CANDIDATES}
    rms_terms = []
    for frac in (0.62, 0.82):
        cut = snaps[0].hours + frac * span
        k = max(1, sum(1 for s in snaps if s.hours <= cut))
        if k >= m:
            continue
        train, tw = snaps[:k], weights[:k]
        h_days = max((snaps[-1].hours - train[-1].hours) / 24.0, 0.01)
        vol = logit_volatility(train)
        preds = {}
        for name, fn in CANDIDATES:
            med, sig = fn(train, tw, trophies, snaps[-1].u, h_days, vol)
            preds[name] = (med, sig)
            sc = pinball_mean(med, sig, actual, levels)
            if sc is not None:
                scores[name].append(sc)
        inv = {nme: 1.0 / max(scores[nme][-1], 1e-6)
               for nme, _ in CANDIDATES if scores[nme]}
        tot = sum(inv.values())
        if tot > 0:
            for i in range(len(trophies)):
                if actual[i] is None:
                    continue
                mix = sum(inv[nme] / tot * preds[nme][0][i] for nme in inv)
                rms_terms.append((actual[i] - mix) ** 2)
    cand_w = {}
    for name, _ in CANDIDATES:
        if scores[name]:
            mean_sc = sum(scores[name]) / len(scores[name])
            cand_w[name] = math.exp(-mean_sc / 0.12)
    tot = sum(cand_w.values())
    if tot <= 0:
        cand_w = {"last": 1.0}
    else:
        cand_w = {k: v / tot for k, v in cand_w.items()}
    rms = math.sqrt(sum(rms_terms) / len(rms_terms)) if rms_terms else 0.8
    return cand_w, rms


# -------------------------------------------------------------- forecast

def forecast_case(case):
    levels = tuple(case.get("levels") or DEFAULT_LEVELS)
    trophies = case["trophies"]
    n = len(trophies)
    release_dt = datetime.fromisoformat(case["game"]["release_date"]) \
        .replace(tzinfo=timezone.utc)
    snaps = compress_history(case, release_dt)
    if not snaps:  # no history at all: neutral wide forecast
        rows = [{"id": t["id"],
                 "q": [round(clamp_rate(logistic_pct(logit(5.0)
                                                     + norm_ppf(tau) * 1.5)), 4)
                       for tau in levels]} for t in trophies]
        return {"trophies": rows}
    weights = snap_weights(snaps)
    target_hours = (parse_time(case["target"]) - release_dt) \
        .total_seconds() / 3600.0
    target_u = math.log1p(max(target_hours, -0.99))
    horizon_days = case["horizon_minutes"] / 1440.0

    if len(snaps) == 1:
        # Degenerate history: no refresh has ever been seen, so the one
        # thing we know is that a refresh can move everything at once.
        last = last_known_rates(snaps, n)
        sigma = min(1.8, max(1.15, 0.55 * math.sqrt(horizon_days)))
        med = [logit(r if r is not None else 1.0) for r in last]
        sig = [sigma] * n
    else:
        vol = logit_volatility(snaps)
        cand_w, bt_rms = backtest(snaps, weights, trophies, levels)
        # Backtests only predict a fraction of the span ahead; when the
        # real horizon is much longer, lean back toward an even prior.
        span_days = max((snaps[-1].hours - snaps[0].hours) / 24.0, 0.05)
        lam = 0.5 * horizon_days / (horizon_days + 2.0 * span_days)
        names = [name for name, _ in CANDIDATES]
        mixed = {name: (1.0 - lam) * cand_w.get(name, 0.0) + lam / len(names)
                 for name in names}
        tot = sum(mixed.values())
        cand_w = {k: v / tot for k, v in mixed.items()}
        preds = {}
        for name, fn in CANDIDATES:
            if cand_w.get(name, 0.0) > 0:
                preds[name] = fn(snaps, weights, trophies, target_u,
                                 horizon_days, vol)
        med, sig = [], []
        bt_floor = min(1.5, max(0.3, bt_rms *
                                math.sqrt(horizon_days / (0.2 * span_days))))
        for i in range(n):
            mix = sum(cand_w[nme] * preds[nme][0][i] for nme in preds)
            var = sum(cand_w[nme] * (preds[nme][1][i] ** 2 +
                                     preds[nme][0][i] ** 2)
                      for nme in preds) - mix * mix
            s = math.sqrt(max(var, 0.0))
            med.append(mix)
            sig.append(min(2.5, max(s, bt_floor, 0.3)))

    rows = []
    for i, t in enumerate(trophies):
        qs = [clamp_rate(logistic_pct(med[i] + norm_ppf(tau) * sig[i]))
              for tau in levels]
        qs = [round(q, 4) for q in qs]
        for j in range(1, len(qs)):  # enforce non-decreasing after rounding
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        payload = json.dumps(result, separators=(",", ":")) + "\n"
        (out_dir / (path.stem + ".out")).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
