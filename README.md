[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zaoan897-wq/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zaoan897-wq/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Entry for make.fun's `gta6-trophy-rarity-forecast-01`: forecast, for each
of GTA VI's 55 PS5 trophies, the quantiles of its Exophase earn rate at
the case's target time. `main.py` is Python 3, standard library only,
deterministic, and runs in well under a second per case.

## Why the rate moves

A printed rate is `earners / players_tracked`. Exophase refreshes the
page in steps: the tracked-player count P jumps at each refresh while
each trophy's earners e_i grow as tracked players progress. A rate can
therefore fall while its earners grow (the denominator outruns it), or
rise for late-game trophies (progression outruns the denominator).

## Model

History is compressed to distinct snapshots first — between refreshes
the page is identical, and plateaus must not dominate the fits. Three
candidate forecasts are computed per trophy:

- **last** — the most recent rate, with a spread from the pooled
  snapshot-to-snapshot logit volatility.
- **trend** — logit(rate) against ln(1 + hours since release); the
  slope blends whole-history and recent-window fits, is shrunk toward
  the trophy's tier, and the forecast is anchored at the last read.
- **earners** — P is extrapolated (a self-damping log-time line averaged
  with the recent growth rate carried forward under exponential decay,
  floored at the last count, capped at 25×), and each trophy's earners
  follow a power law in P with tier-shrunk elasticity; the shrunk
  slope's intercept is refit through the data's centre of mass.

An internal backtest (fit on the first ~62% / ~82% of the span, predict
the final snapshot, score mean pinball loss on the logit scale) weights
the candidates; because backtests predict much shorter horizons than
the real one, the weights are mixed toward an even prior as the horizon
outgrows the span. The final spread is the mixture's own spread floored
by the backtest residual scale grown to the real horizon. A history
with a single distinct snapshot keeps the last rate with a deliberately
wide spread: an unseen refresh can move every rate at once.

Quantiles are `100 · logistic(logit(median) + z_τ · σ)` at the case's
levels, clamped to [0.01, 99.99] and made non-decreasing.
