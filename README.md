# Demand Forecasting — SARIMA vs Machine Learning

Forecasting a strongly seasonal monthly series (airline passengers,
1949–1960 — the classic demand-forecasting benchmark) three ways:
a seasonal-naive baseline, a SARIMA model, and a random forest on
lag features. The workflow — decompose, test stationarity, model,
beat-the-baseline evaluation — is the same one used for product
demand planning.

## Results

Final 12 months (1960) held out for evaluation.

| Model | RMSE | MAPE |
|---|---|---|
| Seasonal naive (repeat last year) | 50.7 | 10.0% |
| Random forest (lag features) | 33.7 | 4.8% |
| **SARIMA(1,1,1)(1,1,1,12)** | **21.2** | **3.7%** |

Stationarity (ADF test): raw series p = 0.99 (not stationary);
after seasonal differencing p = 0.012 (stationary) — which is exactly
why the SARIMA order includes a seasonal difference (D = 1).

## What the pipeline does

1. **Explore** (`pandas`, `matplotlib`): plot the series and run a
   multiplicative seasonal decomposition — trend, yearly seasonality,
   residual.
2. **Test stationarity** (`statsmodels` ADF test) before and after
   seasonal differencing.
3. **Baseline**: seasonal naive — next year = last year, month for
   month. Any real model has to beat this to be worth anything.
4. **SARIMA**: (1,1,1)(1,1,1,12) — one regular difference for the
   trend, one seasonal difference for the yearly cycle.
5. **ML comparison**: random forest on features that are all known at
   forecast time — the value 12 months ago, a trend index, and the
   calendar month (no leakage from the future).
6. **Evaluate** on the held-out year with RMSE and MAPE; full output
   in `results.txt`, plots in `plots/`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python forecast.py
```

## Data

Monthly international airline passengers, 1949–1960 (the classic
Box–Jenkins series), in `data/passengers.csv`.

## Interview notes — be able to explain

- **Why a baseline first?** "The model got 4% error" means nothing
  until you know the dumb method got 10%. Seasonal naive is the bar
  every forecaster has to clear.
- **What does SARIMA(1,1,1)(1,1,1,12) mean?** AR(1), one difference,
  MA(1); then the same again at the seasonal period of 12 months.
  Differencing is what makes a trending, seasonal series stationary
  enough to model.
- **Why did SARIMA beat the random forest here?** Two reasons: the
  dataset is small (144 points), and tree models cannot extrapolate —
  a random forest can never predict a value higher than anything it
  saw in training, which is a real handicap on a series that grows
  every year. SARIMA models the trend explicitly.
- **Why only lag-12 in the ML model?** At forecast time for a full
  year ahead, last month's value is unknown — but the same month last
  year is known. Using lag-1 would leak future information into a
  multi-month forecast.
- **RMSE vs MAPE?** RMSE punishes big misses (in raw units); MAPE is
  a percentage, comparable across series. Report both.
