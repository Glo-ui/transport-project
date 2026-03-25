import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Transport Demand Analysis",
    page_icon="🚌",
    layout="wide",
)

st.title("Kenya Public Transport Demand Analysis & Forecasting")
st.markdown(
    "Analyzing public transport statistics from five Kenyan counties "
    "— **Kisumu, Mombasa, Nairobi, Nakuru, and Uasin Gishu** — spanning 2013–2024."
)


# ── Data loading ─────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

    def clean_dataset(file_path, county_name):
        df_raw = pd.read_csv(file_path, header=None)
        headers = df_raw.iloc[1].astype(str).str.strip()

        col_names = ["Year"]
        for i in range(1, len(headers)):
            name = headers[i]
            if name in ("nan", ""):
                name = f"Unknown_{i}"
            col_names.append(name)

        df = df_raw.iloc[3:].copy()
        df.columns = col_names[: len(df.columns)]

        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").str.strip().str.strip('"')
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        df["County"] = county_name

        rename_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "road transport" in cl:
                rename_map[col] = "Road Transport"
            elif "panelvan" in cl or "pick-up" in cl:
                rename_map[col] = "PanelVans, Pick-ups"
            elif cl == "minibuses/matatu":
                rename_map[col] = "MiniBuses/Matatu"
            elif "total motor" in cl:
                rename_map[col] = "Total Motor Cycles"
            elif "matatus" in cl and "0-14" in cl:
                rename_map[col] = "Matatus (0-14 seaters)"
            elif "buses" in cl and "34" in cl:
                rename_map[col] = "Buses (34+ seaters)"
            elif ("mini bus" in cl or "mini bues" in cl) and "15-33" in cl:
                rename_map[col] = "Mini Buses (15-33 seaters)"
            elif "buses and coaches" in cl:
                rename_map[col] = "Buses and Coaches"

        df = df.rename(columns=rename_map)
        keep_cols = [
            "Year", "Road Transport", "PanelVans, Pick-ups", "MiniBuses/Matatu",
            "Total Motor Cycles", "Matatus (0-14 seaters)", "Buses (34+ seaters)",
            "Mini Buses (15-33 seaters)", "Buses and Coaches", "County",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df.reset_index(drop=True)

    datasets = {
        "Kisumu": "Kisumu dataset.csv",
        "Mombasa": "Mombasa Dataset.csv",
        "Nairobi": "Nairobi dataset.csv",
        "Nakuru": "Nakuru Dataset.csv",
        "Uasin Gishu": "Uasin Gishu Dataset.csv",
    }

    frames = []
    for county, filename in datasets.items():
        filepath = os.path.join(DATA_DIR, filename)
        frames.append(clean_dataset(filepath, county))

    return pd.concat(frames, ignore_index=True)


df_combined = load_and_clean_data()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_counties = st.sidebar.multiselect(
    "Select Counties",
    options=df_combined["County"].unique().tolist(),
    default=df_combined["County"].unique().tolist(),
)

year_range = st.sidebar.slider(
    "Year Range",
    min_value=int(df_combined["Year"].min()),
    max_value=int(df_combined["Year"].max()),
    value=(int(df_combined["Year"].min()), int(df_combined["Year"].max())),
)

df_filtered = df_combined[
    (df_combined["County"].isin(selected_counties))
    & (df_combined["Year"].between(year_range[0], year_range[1]))
]

# ── Section 1: Data Overview ─────────────────────────────────
st.header("1. Data Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", df_filtered.shape[0])
col2.metric("Counties", len(selected_counties))
col3.metric("Year Span", f"{year_range[0]}–{year_range[1]}")

with st.expander("View Raw Data"):
    st.dataframe(df_filtered, use_container_width=True)

with st.expander("Descriptive Statistics"):
    st.dataframe(df_filtered.describe().round(0), use_container_width=True)

# ── Section 2: EDA ───────────────────────────────────────────
st.header("2. Exploratory Data Analysis")

# Road Transport trend
st.subheader("Road Transport Volume Over Time")
fig1, ax1 = plt.subplots(figsize=(12, 5))
for county in selected_counties:
    data = df_filtered[df_filtered["County"] == county]
    ax1.plot(data["Year"], data["Road Transport"], marker="o", label=county, linewidth=2)
ax1.set_xlabel("Year")
ax1.set_ylabel("Road Transport (KSh Millions)")
ax1.set_title("Road Transport Volume by County")
ax1.legend(title="County")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# Category selector for detailed plots
st.subheader("Transport Category Trends")
transport_categories = [
    "Total Motor Cycles", "Matatus (0-14 seaters)", "Buses (34+ seaters)",
    "Mini Buses (15-33 seaters)", "PanelVans, Pick-ups", "Buses and Coaches",
]
selected_category = st.selectbox("Select a transport category:", transport_categories)

fig2, ax2 = plt.subplots(figsize=(12, 5))
for county in selected_counties:
    data = df_filtered[df_filtered["County"] == county]
    ax2.plot(data["Year"], data[selected_category], marker="o", label=county, linewidth=2)
ax2.set_xlabel("Year")
ax2.set_ylabel("Count")
ax2.set_title(f"{selected_category} by County")
ax2.legend(title="County")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# Correlation heatmap
st.subheader("Correlation Matrix")
numeric_cols = df_filtered.select_dtypes(include=["number"]).columns
corr = df_filtered[numeric_cols].corr()

fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
            center=0, square=True, ax=ax3)
ax3.set_title("Correlation Matrix of Transportation Categories")
fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# Box plots
st.subheader("Distribution by County")
selected_box = st.selectbox(
    "Select category for box plot:",
    ["Road Transport"] + transport_categories,
    key="box",
)

fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_filtered, x="County", y=selected_box, palette="Set2", ax=ax4)
ax4.set_title(f"{selected_box} Distribution by County")
ax4.tick_params(axis="x", rotation=30)
fig4.tight_layout()
st.pyplot(fig4)
plt.close(fig4)


# ── Section 3: Model Training & Evaluation ───────────────────
st.header("3. Model Training & Evaluation")


@st.cache_data
def train_models(df):
    df = df.sort_values(["County", "Year"]).reset_index(drop=True)

    demand_columns = [
        "Road Transport", "PanelVans, Pick-ups", "MiniBuses/Matatu",
        "Total Motor Cycles", "Matatus (0-14 seaters)", "Buses (34+ seaters)",
        "Mini Buses (15-33 seaters)", "Buses and Coaches",
    ]

    for col in demand_columns:
        df[f"{col}_lag1"] = df.groupby("County")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("County")[col].shift(2)

    df["Road_Transport_growth"] = df.groupby("County")["Road Transport"].pct_change()
    df["Year_index"] = df["Year"] - df["Year"].min()

    le = LabelEncoder()
    df["County_encoded"] = le.fit_transform(df["County"])

    target = "Road Transport"
    feature_cols = ["Year", "Year_index", "County_encoded", "Road_Transport_growth"]
    feature_cols += [c for c in df.columns if "_lag1" in c or "_lag2" in c]

    df_model = df.dropna(subset=feature_cols + [target]).copy()

    X = df_model[feature_cols]
    y = df_model[target]

    train_mask = df_model["Year"] <= 2021
    test_mask = df_model["Year"] >= 2022

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        results[name] = {
            "model": model, "predictions": y_pred,
            "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
        }

    # Predictions detail
    best_name = max(results, key=lambda k: results[k]["r2"])
    pred_df = df_model[test_mask][["Year", "County", "Road Transport"]].copy()
    pred_df["Predicted"] = results[best_name]["predictions"].astype(int)
    pred_df["Error_%"] = (
        (pred_df["Road Transport"] - pred_df["Predicted"]) / pred_df["Road Transport"] * 100
    ).round(1)

    # Feature importance
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Train a model for EACH demand category and generate forecasts
    forecast_years = [2025, 2026, 2027]
    all_forecasts = {}

    for target_cat in demand_columns:
        # Train RF model for this category
        y_cat = df_model[target_cat]
        cat_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        cat_model.fit(X_train, y_cat[train_mask])

        cat_forecasts = []
        for county in df["County"].unique():
            county_data = df[df["County"] == county].sort_values("Year").copy()
            county_enc = le.transform([county])[0]
            for year in forecast_years:
                latest = county_data.iloc[-1]
                second = county_data.iloc[-2] if len(county_data) > 1 else latest
                features = {
                    "Year": year,
                    "Year_index": year - df["Year"].min(),
                    "County_encoded": county_enc,
                    "Road_Transport_growth": (
                        (latest["Road Transport"] - second["Road Transport"])
                        / second["Road Transport"]
                    ) if second["Road Transport"] != 0 else 0,
                }
                for col in demand_columns:
                    features[f"{col}_lag1"] = latest[col]
                    features[f"{col}_lag2"] = second[col]
                X_f = pd.DataFrame([features])[feature_cols]
                prediction = cat_model.predict(X_f)[0]
                cat_forecasts.append({
                    "Year": year, "County": county,
                    "Predicted": int(round(prediction)),
                })
                new_row = latest.copy()
                new_row["Year"] = year
                new_row[target_cat] = prediction
                county_data = pd.concat([county_data, pd.DataFrame([new_row])], ignore_index=True)

        all_forecasts[target_cat] = pd.DataFrame(cat_forecasts)

    return results, best_name, pred_df, importances, all_forecasts, y_test, feature_cols, demand_columns


results, best_name, pred_df, importances, all_forecasts, y_test, feature_cols, demand_columns = train_models(
    df_combined.copy()
)

# Metrics table
st.subheader("Model Comparison")
metrics_data = []
for name, res in results.items():
    metrics_data.append({
        "Model": name,
        "MAE": f"{res['mae']:,.0f}",
        "RMSE": f"{res['rmse']:,.0f}",
        "R² Score": f"{res['r2']:.3f}",
        "MAPE": f"{res['mape']:.1f}%",
    })
st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
st.success(f"Best model: **{best_name}** (R² = {results[best_name]['r2']:.3f})")

# Predicted vs Actual
st.subheader("Predicted vs Actual")
col_a, col_b = st.columns(2)

with col_a:
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    y_pred_best = results[best_name]["predictions"]
    ax5.scatter(y_test, y_pred_best, alpha=0.7, edgecolors="k", linewidth=0.5, s=80)
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    ax5.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    ax5.set_xlabel("Actual (KSh M)")
    ax5.set_ylabel("Predicted (KSh M)")
    ax5.set_title(f"{best_name} — Predicted vs Actual")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

with col_b:
    st.markdown("**Prediction Details**")
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

# Feature importance
st.subheader("Feature Importance (Random Forest)")
fig6, ax6 = plt.subplots(figsize=(10, 6))
importances.head(10).sort_values().plot(kind="barh", color="steelblue", ax=ax6)
ax6.set_title("Top 10 Feature Importances")
ax6.set_xlabel("Importance")
fig6.tight_layout()
st.pyplot(fig6)
plt.close(fig6)

# ── Section 4: Forecasting ───────────────────────────────────
st.header("4. Demand Forecast (2025–2027)")

forecast_category = st.selectbox(
    "Select transport category to forecast:",
    demand_columns,
    key="forecast_cat",
)

forecast_df = all_forecasts[forecast_category]

# Units label
unit_label = "KSh Millions" if forecast_category == "Road Transport" else "Vehicles"

pivot = forecast_df.pivot(index="County", columns="Year", values="Predicted")
st.dataframe(pivot, use_container_width=True)

fig7, ax7 = plt.subplots(figsize=(14, 6))
for county in df_combined["County"].unique():
    hist = df_combined[df_combined["County"] == county]
    fore = forecast_df[forecast_df["County"] == county]
    ax7.plot(hist["Year"], hist[forecast_category], marker="o", linewidth=2,
             label=f"{county} (actual)")
    ax7.plot(fore["Year"], fore["Predicted"], marker="s",
             linewidth=2, linestyle="--", alpha=0.7)
ax7.axvline(x=2024.5, color="gray", linestyle=":", alpha=0.5, label="Forecast boundary")
ax7.set_title(f"{forecast_category}: Historical + Forecast")
ax7.set_xlabel("Year")
ax7.set_ylabel(f"{forecast_category} ({unit_label})")
ax7.legend(title="County", bbox_to_anchor=(1.05, 1), loc="upper left")
ax7.grid(True, alpha=0.3)
fig7.tight_layout()
st.pyplot(fig7)
plt.close(fig7)

# ── Section 5: Temporal Demand Estimation ─────────────────────
st.header("5. Estimated Demand by Time Period")
st.caption(
    "The source data is annual. Monthly, weekly, daily, and hourly estimates are "
    "derived using Kenyan transport seasonality patterns (school terms, holidays, "
    "rainy seasons, rush hours)."
)

# Seasonality weights — based on Kenya school calendar, holidays, weather
MONTHLY_WEIGHTS = np.array([
    0.090,  # Jan - holiday travel tail, back to school
    0.075,  # Feb - short month, relatively low
    0.080,  # Mar - normal
    0.095,  # Apr - Easter, long rains start (higher matatu demand)
    0.078,  # May - long rains, slightly lower
    0.072,  # Jun - cool/dry, relatively calm
    0.070,  # Jul - lowest, mid-year school break lull
    0.082,  # Aug - school resumes, uptick
    0.085,  # Sep - normal
    0.088,  # Oct - short rains begin
    0.090,  # Nov - picking up toward December
    0.095,  # Dec - peak — holiday travel, festivities
])
MONTHLY_WEIGHTS = MONTHLY_WEIGHTS / MONTHLY_WEIGHTS.sum()  # normalize

DAILY_WEIGHTS = np.array([1.10, 1.12, 1.08, 1.05, 1.15, 0.80, 0.70])  # Mon-Sun
DAILY_WEIGHTS = DAILY_WEIGHTS / DAILY_WEIGHTS.sum()

HOURLY_WEIGHTS = np.array([
    0.005, 0.003, 0.002, 0.002, 0.005, 0.020,  # 0-5
    0.060, 0.095, 0.090, 0.065, 0.050, 0.045,  # 6-11
    0.050, 0.055, 0.045, 0.040, 0.055, 0.090,  # 12-17
    0.085, 0.060, 0.040, 0.025, 0.015, 0.008,  # 18-23
])
HOURLY_WEIGHTS = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

temp_col1, temp_col2 = st.columns(2)
with temp_col1:
    temp_category = st.selectbox(
        "Transport category:", demand_columns, key="temp_cat"
    )
with temp_col2:
    temp_granularity = st.selectbox(
        "Time granularity:", ["Monthly", "Weekly", "Daily", "Hourly"], key="temp_gran"
    )

temp_year = st.slider(
    "Select year:", int(df_combined["Year"].min()), 2027, 2024, key="temp_year"
)

# Get annual value per county for selected year
temp_unit = "KSh Millions" if temp_category == "Road Transport" else "Vehicles"


@st.cache_data
def build_temporal(df_src, all_fc, category, year, granularity):
    rows = []
    for county in df_src["County"].unique():
        # Get annual total: from actual data or forecast
        county_actual = df_src[
            (df_src["County"] == county) & (df_src["Year"] == year)
        ]
        if len(county_actual) > 0:
            annual = county_actual[category].values[0]
        else:
            fc = all_fc.get(category)
            if fc is not None:
                fc_row = fc[(fc["County"] == county) & (fc["Year"] == year)]
                annual = fc_row["Predicted"].values[0] if len(fc_row) > 0 else 0
            else:
                annual = 0

        if granularity == "Monthly":
            for m in range(12):
                rows.append({
                    "County": county, "Period": MONTH_NAMES[m],
                    "Value": annual * MONTHLY_WEIGHTS[m], "Order": m,
                })
        elif granularity == "Weekly":
            weekly = annual / 52
            for w in range(1, 53):
                month_idx = min(int((w - 1) / 4.33), 11)
                seasonal = MONTHLY_WEIGHTS[month_idx] * 12
                rows.append({
                    "County": county, "Period": f"W{w}",
                    "Value": weekly * seasonal, "Order": w,
                })
        elif granularity == "Daily":
            daily_base = annual / 365
            for d in range(7):
                rows.append({
                    "County": county, "Period": DAY_NAMES[d],
                    "Value": daily_base * DAILY_WEIGHTS[d] * 7, "Order": d,
                })
        elif granularity == "Hourly":
            daily_avg = annual / 365
            for h in range(24):
                rows.append({
                    "County": county, "Period": f"{h:02d}:00",
                    "Value": daily_avg * HOURLY_WEIGHTS[h] * 24, "Order": h,
                })
    return pd.DataFrame(rows)


temporal_df = build_temporal(df_combined, all_forecasts, temp_category, temp_year, temp_granularity)

if not temporal_df.empty:
    temporal_df = temporal_df.sort_values("Order")

    fig8, ax8 = plt.subplots(figsize=(14, 6))
    for county in temporal_df["County"].unique():
        cdata = temporal_df[temporal_df["County"] == county]
        ax8.plot(cdata["Period"], cdata["Value"], marker="o", linewidth=1.5,
                 label=county, markersize=4)
    ax8.set_title(f"{temp_category} — {temp_granularity} Estimate ({temp_year})", fontsize=14)
    ax8.set_xlabel(temp_granularity.rstrip("ly") + " Period")
    ax8.set_ylabel(f"{temp_category} ({temp_unit})")
    ax8.legend(title="County")
    ax8.grid(True, alpha=0.3)
    if temp_granularity == "Weekly":
        ax8.set_xticks(ax8.get_xticks()[::4])
    ax8.tick_params(axis="x", rotation=45)
    fig8.tight_layout()
    st.pyplot(fig8)
    plt.close(fig8)

    with st.expander("View data table"):
        pivot_temp = temporal_df.pivot(index="Period", columns="County", values="Value")
        pivot_temp = pivot_temp.loc[
            temporal_df.drop_duplicates("Period").sort_values("Order")["Period"]
        ]
        st.dataframe(pivot_temp.round(0), use_container_width=True)
else:
    st.warning("No data available for the selected year.")


# ── Section 6: Comparative Analysis ──────────────────────────
st.header("6. Comparative Analysis — County Rankings & Trends")

comp_category = st.selectbox(
    "Select transport category for comparison:",
    demand_columns,
    key="comp_cat",
)

comp_unit = "KSh Millions" if comp_category == "Road Transport" else "Vehicles"

# Build ranking table: latest actual (2024) vs forecast (2027)
latest_year = int(df_combined["Year"].max())
forecast_year = 2027


@st.cache_data
def build_comparison(df_src, all_fc, category, base_yr, target_yr):
    rows = []
    for county in df_src["County"].unique():
        actual = df_src[(df_src["County"] == county) & (df_src["Year"] == base_yr)]
        val_now = actual[category].values[0] if len(actual) > 0 else 0

        fc = all_fc.get(category)
        fc_row = fc[(fc["County"] == county) & (fc["Year"] == target_yr)] if fc is not None else pd.DataFrame()
        val_future = fc_row["Predicted"].values[0] if len(fc_row) > 0 else 0

        change_pct = ((val_future - val_now) / val_now * 100) if val_now != 0 else 0

        rows.append({
            "County": county,
            f"{base_yr} Actual": int(val_now),
            f"{target_yr} Forecast": int(val_future),
            "Change (%)": round(change_pct, 1),
        })

    comp = pd.DataFrame(rows)
    comp = comp.sort_values(f"{base_yr} Actual", ascending=False).reset_index(drop=True)
    comp.index = comp.index + 1
    comp.index.name = "Rank"
    return comp


comp_df = build_comparison(df_combined, all_forecasts, comp_category, latest_year, forecast_year)

# Display with trend indicators
st.subheader(f"County Rankings — {comp_category}")

def style_trend(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return "color: green"
        elif val < 0:
            return "color: red"
    return ""

styled = comp_df.style.applymap(style_trend, subset=["Change (%)"])
st.dataframe(styled, use_container_width=True)

# Side-by-side bar chart
st.subheader(f"{latest_year} vs {forecast_year} — {comp_category}")

fig9, ax9 = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
width = 0.35
bars1 = ax9.bar(x - width / 2, comp_df[f"{latest_year} Actual"], width,
                label=f"{latest_year} Actual", color="steelblue")
bars2 = ax9.bar(x + width / 2, comp_df[f"{forecast_year} Forecast"], width,
                label=f"{forecast_year} Forecast", color="coral")
ax9.set_xlabel("County")
ax9.set_ylabel(f"{comp_category} ({comp_unit})")
ax9.set_title(f"{comp_category}: Current vs Forecast by County")
ax9.set_xticks(x)
ax9.set_xticklabels(comp_df["County"], rotation=30)
ax9.legend()
ax9.grid(True, alpha=0.3, axis="y")

# Add trend arrows on bars
for i, row in comp_df.iterrows():
    change = row["Change (%)"]
    arrow = "+" if change > 0 else ""
    ax9.annotate(
        f"{arrow}{change:.1f}%",
        xy=(i - 1 + width / 2, row[f"{forecast_year} Forecast"]),
        ha="center", va="bottom", fontsize=9, fontweight="bold",
        color="green" if change > 0 else "red",
    )

fig9.tight_layout()
st.pyplot(fig9)
plt.close(fig9)

# Trend summary across ALL categories
st.subheader("Trend Summary — All Categories")


@st.cache_data
def build_trend_matrix(df_src, all_fc, cats, base_yr, target_yr):
    matrix = {}
    for cat in cats:
        cat_trends = {}
        for county in df_src["County"].unique():
            actual = df_src[(df_src["County"] == county) & (df_src["Year"] == base_yr)]
            val_now = actual[cat].values[0] if len(actual) > 0 else 0
            fc = all_fc.get(cat)
            fc_row = fc[(fc["County"] == county) & (fc["Year"] == target_yr)] if fc is not None else pd.DataFrame()
            val_future = fc_row["Predicted"].values[0] if len(fc_row) > 0 else 0
            change = ((val_future - val_now) / val_now * 100) if val_now != 0 else 0
            cat_trends[county] = round(change, 1)
        matrix[cat] = cat_trends
    return pd.DataFrame(matrix).T


trend_matrix = build_trend_matrix(
    df_combined, all_forecasts, demand_columns, latest_year, forecast_year
)
trend_matrix.index.name = "Category"


def color_cells(val):
    if isinstance(val, (int, float)):
        if val > 5:
            return "background-color: #d4edda; color: #155724"
        elif val < -5:
            return "background-color: #f8d7da; color: #721c24"
        else:
            return "background-color: #fff3cd; color: #856404"
    return ""


st.markdown(
    f"Percentage change from **{latest_year}** to **{forecast_year}** forecast. "
    "Green = growth (>5%), Red = decline (>5%), Yellow = stable."
)
st.dataframe(
    trend_matrix.style.applymap(color_cells).format("{:+.1f}%"),
    use_container_width=True,
)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "*Data Source: Kenya National Bureau of Statistics (KNBS) Economic Surveys, "
    "with county-level estimates derived from Gross County Product weights.*"
)
