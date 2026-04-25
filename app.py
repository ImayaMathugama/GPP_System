import traceback
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from gold_forecast_core import run_end_to_end


st.set_page_config(page_title="Gold Price Prediction System", layout="wide")
st.title("🪙 Gold Price Prediction System (NumPy + Pandas + ML)")
st.caption("Forecast monthly gold prices with market and crisis scenario factors.")

# Sidebar controls
st.sidebar.header("Forecast Settings")

forecast_start = st.sidebar.text_input("Forecast Start (YYYY-MM-01)", value="2026-05-01")
forecast_months = st.sidebar.slider("Forecast Horizon (months)", min_value=1, max_value=48, value=20)

scenario = st.sidebar.selectbox(
    "Scenario",
    options=["baseline", "crisis", "risk_on"],
    index=0,
    help="baseline: unchanged assumptions, crisis: stress/fear assumptions, risk_on: calmer market assumptions"
)

st.sidebar.markdown("### Custom Macro Shocks (%)")
vix_bump_pct = st.sidebar.slider("VIX bump %", -50.0, 200.0, 0.0, 1.0)
oil_bump_pct = st.sidebar.slider("Oil bump %", -50.0, 200.0, 0.0, 1.0)
tnx_bump_pct = st.sidebar.slider("10Y Yield bump %", -50.0, 200.0, 0.0, 1.0)
usd_bump_pct = st.sidebar.slider("USD proxy bump %", -50.0, 200.0, 0.0, 1.0)

run_btn = st.sidebar.button("Run Forecast", type="primary")

if run_btn:
    with st.spinner("Downloading data, training model, and forecasting..."):
        try:
            result = run_end_to_end(
                forecast_start=forecast_start,
                forecast_months=forecast_months,
                scenario=scenario,
                vix_bump_pct=vix_bump_pct,
                oil_bump_pct=oil_bump_pct,
                tnx_bump_pct=tnx_bump_pct,
                usd_bump_pct=usd_bump_pct
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
            st.stop()

    forecast = result["forecast"].copy()
    monthly = result["monthly"].copy()
    metrics = result["metrics"]
    fi = result["feature_importance"].copy()

    # KPI metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.2f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c3.metric("MAPE %", f"{metrics['MAPE_%']:.2f}")

    # Chart
    st.subheader("Forecast Chart")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly["gold_avg"], label="Historical Gold (monthly avg)")
    ax.plot(forecast.index, forecast["predicted_gold_avg"], "--", label="Forecast")
    ax.fill_between(
        forecast.index,
        forecast["predicted_lower_95"],
        forecast["predicted_upper_95"],
        alpha=0.2,
        label="95% interval"
    )
    ax.set_title("Gold Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Tables
    st.subheader("Forecast Table")
    show_df = forecast.copy()
    show_df.index.name = "month"
    st.dataframe(show_df.round(2), use_container_width=True)

    st.subheader("Top Feature Importances")
    st.dataframe(fi.head(20).round(6), use_container_width=True)

    # Download buttons
    st.subheader("Download Results")
    forecast_csv = show_df.reset_index().to_csv(index=False).encode("utf-8")
    metrics_csv = pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8")
    fi_csv = fi.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast CSV",
        data=forecast_csv,
        file_name="gold_forecast.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Metrics CSV",
        data=metrics_csv,
        file_name="model_metrics.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Feature Importance CSV",
        data=fi_csv,
        file_name="feature_importance.csv",
        mime="text/csv"
    )

    # Quick requested window: May-Dec 2026
    st.subheader("Requested Window: May-Dec 2026")
    req = show_df.loc[(show_df.index >= "2026-05-01") & (show_df.index <= "2026-12-01")]
    if len(req) == 0:
        st.info("No rows in May-Dec 2026 for current settings.")
    else:
        st.dataframe(req.round(2), use_container_width=True)

else:
    st.info("Set parameters in the sidebar and click **Run Forecast**.")