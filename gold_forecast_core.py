import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_CONFIG = {
    "start_date": "2000-01-01",
    "end_date": None,  # None = today
    "random_state": 42,
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "use_manual_event_flags": True
}


def download_market_data(start_date: str, end_date=None) -> pd.DataFrame:
    tickers = {
        "gold": "GC=F",
        "sp500": "^GSPC",
        "vix": "^VIX",
        "usd_proxy": "UUP",
        "oil": "CL=F",
        "tnx": "^TNX",
        "btc": "BTC-USD"
    }

    frames = []
    for col_name, ticker in tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError(f"Downloaded empty data for ticker: {ticker}")

        close_data = data["Close"]

        # yfinance may return Series or DataFrame depending on version/options
        if isinstance(close_data, pd.DataFrame):
            s = close_data.iloc[:, 0].rename(col_name)
        else:
            s = close_data.rename(col_name)

        frames.append(s)

    df = pd.concat(frames, axis=1).sort_index()
    df = df.ffill().dropna()
    return df


def apply_manual_event_flags(monthly_df: pd.DataFrame) -> pd.DataFrame:
    m = monthly_df.copy()

    # Historical examples (editable)
    m.loc[(m.index >= "2020-03-01") & (m.index <= "2020-06-01"), "global_crisis_flag"] = 1
    m.loc[(m.index >= "2022-02-01") & (m.index <= "2023-12-01"), "war_shock_flag"] = 1
    m.loc[(m.index >= "2021-06-01") & (m.index <= "2023-06-01"), "inflation_shock_flag"] = 1

    return m


def make_monthly_features(daily_df: pd.DataFrame, use_manual_event_flags: bool = True) -> pd.DataFrame:
    df = daily_df.copy()

    for c in ["gold", "sp500", "usd_proxy", "oil", "btc"]:
        df[f"{c}_ret_1d"] = df[c].pct_change()

    monthly = pd.DataFrame(index=df.resample("MS").mean().index)

    for c in ["gold", "sp500", "vix", "usd_proxy", "oil", "tnx", "btc"]:
        monthly[f"{c}_avg"] = df[c].resample("MS").mean()

    for c in ["gold", "sp500", "usd_proxy", "oil", "btc"]:
        monthly[f"{c}_mom"] = monthly[f"{c}_avg"].pct_change()

    for c in ["gold", "sp500", "usd_proxy", "oil", "btc"]:
        monthly[f"{c}_rv"] = df[f"{c}_ret_1d"].resample("MS").std()

    monthly["fear_flag"] = (
        monthly["vix_avg"] >= monthly["vix_avg"].rolling(24, min_periods=6).median() * 1.2
    ).astype(int)

    rolling_max = monthly["sp500_avg"].rolling(12, min_periods=3).max()
    monthly["sp500_drawdown"] = (monthly["sp500_avg"] / rolling_max) - 1.0

    usd_roll = monthly["usd_proxy_avg"].rolling(6, min_periods=3).mean()
    monthly["usd_uptrend"] = (monthly["usd_proxy_avg"] > usd_roll).astype(int)

    tnx_roll = monthly["tnx_avg"].rolling(6, min_periods=3).mean()
    monthly["rates_uptrend"] = (monthly["tnx_avg"] > tnx_roll).astype(int)

    for lag in [1, 2, 3, 6, 12]:
        monthly[f"gold_lag_{lag}"] = monthly["gold_avg"].shift(lag)

    monthly["target_next_gold"] = monthly["gold_avg"].shift(-1)

    monthly["month"] = monthly.index.month
    monthly["quarter"] = monthly.index.quarter
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12.0)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12.0)

    monthly["global_crisis_flag"] = 0
    monthly["war_shock_flag"] = 0
    monthly["inflation_shock_flag"] = 0

    if use_manual_event_flags:
        monthly = apply_manual_event_flags(monthly)

    monthly = monthly.dropna().copy()
    return monthly


def get_feature_columns():
    return [
        "gold_avg", "sp500_avg", "vix_avg", "usd_proxy_avg", "oil_avg", "tnx_avg", "btc_avg",
        "gold_mom", "sp500_mom", "usd_proxy_mom", "oil_mom", "btc_mom",
        "gold_rv", "sp500_rv", "usd_proxy_rv", "oil_rv", "btc_rv",
        "fear_flag", "sp500_drawdown", "usd_uptrend", "rates_uptrend",
        "gold_lag_1", "gold_lag_2", "gold_lag_3", "gold_lag_6", "gold_lag_12",
        "month", "quarter", "month_sin", "month_cos",
        "global_crisis_flag", "war_shock_flag", "inflation_shock_flag"
    ]


def train_model(monthly_df: pd.DataFrame, config: dict):
    features = get_feature_columns()

    X = monthly_df[features]
    y = monthly_df["target_next_gold"]

    split_idx = int(len(monthly_df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=config["random_state"],
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = math.sqrt(mean_squared_error(y_test, pred_test))
    mape = np.mean(np.abs((y_test - pred_test) / y_test)) * 100

    train_pred = model.predict(X_train)
    residual_std = np.std(y_train - train_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}
    return model, features, metrics, residual_std


def build_future_dataframe(last_monthly_row: pd.Series, forecast_start: str, periods: int) -> pd.DataFrame:
    future_idx = pd.date_range(start=forecast_start, periods=int(periods), freq="MS")
    future = pd.DataFrame(index=future_idx)

    exog_cols = [
        "sp500_avg", "vix_avg", "usd_proxy_avg", "oil_avg", "tnx_avg", "btc_avg",
        "sp500_mom", "usd_proxy_mom", "oil_mom", "btc_mom",
        "sp500_rv", "usd_proxy_rv", "oil_rv", "btc_rv",
        "fear_flag", "sp500_drawdown", "usd_uptrend", "rates_uptrend",
        "global_crisis_flag", "war_shock_flag", "inflation_shock_flag"
    ]

    for c in exog_cols:
        future[c] = float(last_monthly_row[c])

    future["month"] = future.index.month
    future["quarter"] = future.index.quarter
    future["month_sin"] = np.sin(2 * np.pi * future["month"] / 12.0)
    future["month_cos"] = np.cos(2 * np.pi * future["month"] / 12.0)

    return future


def apply_scenario_to_future(
    future_df: pd.DataFrame,
    scenario,
    vix_bump_pct: float = 0.0,
    oil_bump_pct: float = 0.0,
    tnx_bump_pct: float = 0.0,
    usd_bump_pct: float = 0.0
) -> pd.DataFrame:
    f = future_df.copy()
    scenario_val = "{}".format(scenario).strip().lower()

    if scenario_val == "crisis":
        f["global_crisis_flag"] = 1
        f["war_shock_flag"] = 1
        f["inflation_shock_flag"] = 1
        f["fear_flag"] = 1
        f["vix_avg"] = f["vix_avg"] * 1.20
        f["oil_avg"] = f["oil_avg"] * 1.10
    elif scenario_val == "risk_on":
        f["global_crisis_flag"] = 0
        f["war_shock_flag"] = 0
        f["inflation_shock_flag"] = 0
        f["fear_flag"] = 0
        f["vix_avg"] = f["vix_avg"] * 0.90
    else:
        pass

    vix_b = float(vix_bump_pct)
    oil_b = float(oil_bump_pct)
    tnx_b = float(tnx_bump_pct)
    usd_b = float(usd_bump_pct)

    f["vix_avg"] = f["vix_avg"] * (1.0 + vix_b / 100.0)
    f["oil_avg"] = f["oil_avg"] * (1.0 + oil_b / 100.0)
    f["tnx_avg"] = f["tnx_avg"] * (1.0 + tnx_b / 100.0)
    f["usd_proxy_avg"] = f["usd_proxy_avg"] * (1.0 + usd_b / 100.0)

    return f


def recursive_forecast(model, features, history_df: pd.DataFrame, future_df: pd.DataFrame, residual_std: float):
    gold_series = history_df["gold_avg"].copy()
    preds, lower, upper = [], [], []

    for dt in future_df.index:
        row = future_df.loc[dt].copy()

        row["gold_avg"] = gold_series.iloc[-1]
        row["gold_mom"] = gold_series.pct_change().iloc[-1] if len(gold_series) > 1 else 0.0

        recent_changes = gold_series.pct_change().dropna().tail(6)
        row["gold_rv"] = recent_changes.std() if len(recent_changes) > 2 else 0.01

        for lag in [1, 2, 3, 6, 12]:
            row[f"gold_lag_{lag}"] = gold_series.iloc[-lag] if len(gold_series) >= lag else gold_series.iloc[0]

        x_row = pd.DataFrame([row])[features]
        yhat = model.predict(x_row)[0]

        preds.append(float(yhat))
        lower.append(float(yhat - 1.96 * residual_std))
        upper.append(float(yhat + 1.96 * residual_std))

        gold_series.loc[dt] = yhat

    out = pd.DataFrame({
        "predicted_gold_avg": preds,
        "predicted_lower_95": lower,
        "predicted_upper_95": upper
    }, index=future_df.index)

    return out


def run_end_to_end(
    forecast_start: str,
    forecast_months: int,
    scenario="baseline",
    vix_bump_pct: float = 0.0,
    oil_bump_pct: float = 0.0,
    tnx_bump_pct: float = 0.0,
    usd_bump_pct: float = 0.0,
    config: dict = None
):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    daily = download_market_data(cfg["start_date"], cfg["end_date"])
    monthly = make_monthly_features(daily, use_manual_event_flags=cfg["use_manual_event_flags"])

    model, features, metrics, residual_std = train_model(monthly, cfg)

    future = build_future_dataframe(monthly.iloc[-1], forecast_start, int(forecast_months))
    future = apply_scenario_to_future(
        future,
        scenario=scenario,
        vix_bump_pct=vix_bump_pct,
        oil_bump_pct=oil_bump_pct,
        tnx_bump_pct=tnx_bump_pct,
        usd_bump_pct=usd_bump_pct
    )

    forecast = recursive_forecast(model, features, monthly, future, residual_std)

    fi = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return {
        "daily": daily,
        "monthly": monthly,
        "forecast": forecast,
        "metrics": metrics,
        "feature_importance": fi
    }