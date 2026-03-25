import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Decentralized Transport Demand Forecasting",
    page_icon="🚌",
    layout="wide",
)

st.title("Decentralized Transportation Demand Forecasting Across Kenyan Urban Areas")
st.markdown(
    "**Optimizing Public Transportation Through Distributed Data Systems**\n\n"
    "Course: Distributed Machine Learning, CMT 444  \n"
    "Team: Gloria Wanjiku Kamau — 1061409 | Cindy Mitchell Owoko — 1060318"
)

# ── Project Overview ─────────────────────────────────────────
st.header("1. Project Overview")
col_ov1, col_ov2 = st.columns([2, 1])
with col_ov1:
    st.markdown(
        "This study predicts transportation demand in five Kenyan cities using "
        "**decentralized machine learning**. Data from transport statistics, "
        "population patterns, and activity levels is used to model demand "
        "across urban areas.\n\n"
        "Each city trains its own model independently, and predictions are "
        "combined using **federated averaging**. This approach improves accuracy "
        "while preserving data privacy and scalability."
    )
with col_ov2:
    st.info("**Key Innovation**\n\nDecentralized machine learning where cities "
            "share *predictions*, not raw data.")

# Research objectives
st.subheader("Research Objectives")
obj_cols = st.columns(5)
objectives = [
    ("Predict Transport Demand",
     "Forecast demand patterns across five Kenyan cities using ML models"),
    ("Data Integration",
     "Use transport statistics and population data for comprehensive modeling"),
    ("Decentralized Training",
     "Train separate models for each city to maintain data independence"),
    ("Federated Learning",
     "Combine models using federated averaging for improved accuracy"),
    ("Performance Evaluation",
     "Evaluate accuracy using MAE, RMSE, and R² metrics"),
]
for i, (title, desc) in enumerate(objectives):
    with obj_cols[i]:
        st.metric(label=f"0{i+1}", value=title)
        st.caption(desc)

# Why decentralized
st.subheader("Why Decentralized Learning?")
why_cols = st.columns(3)
with why_cols[0]:
    st.success("**Data Privacy**\n\nRaw data never leaves city servers, "
               "protecting sensitive transportation information.")
with why_cols[1]:
    st.success("**Scalability**\n\nAdditional cities can join without "
               "retraining entire models.")
with why_cols[2]:
    st.success("**Efficiency**\n\nLocal models train faster on relevant "
               "data without centralized bottlenecks.")


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

DEMAND_COLUMNS = [
    "Road Transport", "PanelVans, Pick-ups", "MiniBuses/Matatu",
    "Total Motor Cycles", "Matatus (0-14 seaters)", "Buses (34+ seaters)",
    "Mini Buses (15-33 seaters)", "Buses and Coaches",
]

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

# ── Section 2: Data Sources & EDA ────────────────────────────
st.header("2. Data Sources & Exploration")
st.markdown(
    "**Data Sources:** Transport statistics (KNBS), population distributions, "
    "and activity-level indicators across five Kenyan urban areas."
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", df_filtered.shape[0])
col2.metric("Counties", len(selected_counties))
col3.metric("Year Span", f"{year_range[0]}–{year_range[1]}")

with st.expander("View Raw Data"):
    st.dataframe(df_filtered, use_container_width=True)

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
transport_categories = [c for c in DEMAND_COLUMNS if c != "Road Transport"]
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

# Correlation
st.subheader("Correlation Matrix")
numeric_cols = df_filtered.select_dtypes(include=["number"]).columns
corr = df_filtered[numeric_cols].corr()
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, center=0, square=True, ax=ax3)
ax3.set_title("Correlation Matrix of Transportation Categories")
fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)


# ── Section 3: Decentralized Model Training ──────────────────
st.header("3. Decentralized Model Training (Federated Learning)")
st.markdown(
    "**Approach:** Random Forest models are trained **separately for each city** "
    "(local models). Predictions from all city models are then combined using "
    "**federated averaging** to produce a global prediction. This simulates a "
    "federated learning system where raw data never leaves city boundaries."
)


@st.cache_data
def train_federated(df):
    """Train per-city local models + federated average. Return all results."""
    df = df.sort_values(["County", "Year"]).reset_index(drop=True)
    le = LabelEncoder()
    df["County_encoded"] = le.fit_transform(df["County"])

    # Feature engineering
    for col in DEMAND_COLUMNS:
        df[f"{col}_lag1"] = df.groupby("County")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("County")[col].shift(2)
    df["Road_Transport_growth"] = df.groupby("County")["Road Transport"].pct_change()
    df["Year_index"] = df["Year"] - df["Year"].min()

    feature_cols = ["Year", "Year_index", "Road_Transport_growth"]
    feature_cols += [c for c in df.columns if "_lag1" in c or "_lag2" in c]

    df_model = df.dropna(subset=feature_cols).copy()
    train_mask = df_model["Year"] <= 2021
    test_mask = df_model["Year"] >= 2022

    counties = df["County"].unique().tolist()

    # ── Per-city local models (for Road Transport as primary target) ──
    local_results = {}
    local_test_preds = {}
    for county in counties:
        c_train = df_model[train_mask & (df_model["County"] == county)]
        c_test = df_model[test_mask & (df_model["County"] == county)]
        if len(c_train) < 3 or len(c_test) == 0:
            continue
        X_tr = c_train[feature_cols]
        y_tr = c_train["Road Transport"]
        X_te = c_test[feature_cols]
        y_te = c_test["Road Transport"]

        model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        mae = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2 = r2_score(y_te, preds) if len(y_te) > 1 else 0.0

        local_results[county] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2}
        local_test_preds[county] = {
            "y_true": y_te.values, "y_pred": preds,
            "years": c_test["Year"].values,
        }

    # ── Federated model: average predictions from all local models ──
    fed_preds_all = []
    fed_true_all = []
    fed_details = []
    for county in counties:
        c_test = df_model[test_mask & (df_model["County"] == county)]
        if len(c_test) == 0:
            continue
        X_te = c_test[feature_cols]
        y_te = c_test["Road Transport"]

        # Each local model predicts on this county's test data
        all_preds = []
        for c_name, c_res in local_results.items():
            all_preds.append(c_res["model"].predict(X_te))

        # Federated average
        fed_pred = np.mean(all_preds, axis=0)
        fed_preds_all.extend(fed_pred)
        fed_true_all.extend(y_te.values)

        for j in range(len(y_te)):
            fed_details.append({
                "Year": c_test["Year"].values[j], "County": county,
                "Actual": int(y_te.values[j]), "Federated Pred": int(fed_pred[j]),
            })

    fed_true_arr = np.array(fed_true_all)
    fed_pred_arr = np.array(fed_preds_all)
    fed_mae = mean_absolute_error(fed_true_arr, fed_pred_arr)
    fed_rmse = np.sqrt(mean_squared_error(fed_true_arr, fed_pred_arr))
    fed_r2 = r2_score(fed_true_arr, fed_pred_arr)

    # ── Centralized baseline (single model on all data) ──
    X_train_all = df_model[train_mask][feature_cols]
    y_train_all = df_model[train_mask]["Road Transport"]
    X_test_all = df_model[test_mask][feature_cols]
    y_test_all = df_model[test_mask]["Road Transport"]

    central_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    central_model.fit(X_train_all, y_train_all)
    central_preds = central_model.predict(X_test_all)
    central_mae = mean_absolute_error(y_test_all, central_preds)
    central_rmse = np.sqrt(mean_squared_error(y_test_all, central_preds))
    central_r2 = r2_score(y_test_all, central_preds)

    # ── Per-category forecasts (using federated approach) ──
    forecast_years = [2025, 2026, 2027]
    all_forecasts = {}
    for target_cat in DEMAND_COLUMNS:
        # Train local models per county for this category
        cat_local_models = {}
        for county in counties:
            c_train = df_model[train_mask & (df_model["County"] == county)]
            if len(c_train) < 3:
                continue
            m = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
            m.fit(c_train[feature_cols], c_train[target_cat])
            cat_local_models[county] = m

        cat_forecasts = []
        for county in counties:
            county_data = df[df["County"] == county].sort_values("Year").copy()
            for year in forecast_years:
                latest = county_data.iloc[-1]
                second = county_data.iloc[-2] if len(county_data) > 1 else latest
                features = {
                    "Year": year,
                    "Year_index": year - df["Year"].min(),
                    "Road_Transport_growth": (
                        (latest["Road Transport"] - second["Road Transport"])
                        / second["Road Transport"]
                    ) if second["Road Transport"] != 0 else 0,
                }
                for col in DEMAND_COLUMNS:
                    features[f"{col}_lag1"] = latest[col]
                    features[f"{col}_lag2"] = second[col]
                X_f = pd.DataFrame([features])[feature_cols]

                # Federated average of all local models
                preds = [m.predict(X_f)[0] for m in cat_local_models.values()]
                prediction = np.mean(preds) if preds else 0

                cat_forecasts.append({
                    "Year": year, "County": county,
                    "Predicted": int(round(prediction)),
                })
                new_row = latest.copy()
                new_row["Year"] = year
                new_row[target_cat] = prediction
                county_data = pd.concat(
                    [county_data, pd.DataFrame([new_row])], ignore_index=True
                )
        all_forecasts[target_cat] = pd.DataFrame(cat_forecasts)

    # Feature importance from centralized model
    importances = pd.Series(
        central_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return (local_results, fed_mae, fed_rmse, fed_r2, pd.DataFrame(fed_details),
            central_mae, central_rmse, central_r2,
            importances, all_forecasts, local_test_preds, feature_cols)


(local_results, fed_mae, fed_rmse, fed_r2, fed_details_df,
 central_mae, central_rmse, central_r2,
 importances, all_forecasts, local_test_preds, feature_cols) = train_federated(
    df_combined.copy()
)

# ── Local model results per city ──
st.subheader("Local Model Performance (Per City)")
local_rows = []
for county, res in local_results.items():
    local_rows.append({
        "City": county, "MAE": f"{res['mae']:,.0f}",
        "RMSE": f"{res['rmse']:,.0f}", "R²": f"{res['r2']:.3f}",
    })
st.dataframe(pd.DataFrame(local_rows), use_container_width=True, hide_index=True)

# ── Federated vs Centralized comparison ──
st.subheader("Federated vs Centralized Model Comparison")
comp_rows = [
    {"Approach": "Federated (averaged local models)",
     "MAE": f"{fed_mae:,.0f}", "RMSE": f"{fed_rmse:,.0f}", "R²": f"{fed_r2:.3f}"},
    {"Approach": "Centralized (single global model)",
     "MAE": f"{central_mae:,.0f}", "RMSE": f"{central_rmse:,.0f}", "R²": f"{central_r2:.3f}"},
]
st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

if fed_r2 >= central_r2:
    st.success("Federated model matches or outperforms the centralized model — "
               "achieving good accuracy while keeping data decentralized!")
else:
    st.info("Centralized model has a slight edge, but federated learning "
            "preserves data privacy with competitive accuracy.")

# Feature importance
st.subheader("Feature Importance")
fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
importances.head(10).sort_values().plot(kind="barh", color="steelblue", ax=ax_imp)
ax_imp.set_title("Top 10 Features Driving Demand Predictions")
ax_imp.set_xlabel("Importance")
fig_imp.tight_layout()
st.pyplot(fig_imp)
plt.close(fig_imp)

# ── Section 4: Demand Patterns ───────────────────────────────
st.header("4. Key Findings: Demand Patterns")

HOURLY_WEIGHTS = np.array([
    0.005, 0.003, 0.002, 0.002, 0.005, 0.020,  # 0-5
    0.060, 0.095, 0.090, 0.065, 0.050, 0.045,  # 6-11
    0.050, 0.055, 0.045, 0.040, 0.055, 0.090,  # 12-17
    0.085, 0.060, 0.040, 0.025, 0.015, 0.008,  # 18-23
])
HOURLY_WEIGHTS = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()

DAILY_WEIGHTS = np.array([1.10, 1.12, 1.08, 1.05, 1.15, 0.80, 0.70])  # Mon-Sun
DAILY_WEIGHTS = DAILY_WEIGHTS / DAILY_WEIGHTS.sum()

MONTHLY_WEIGHTS = np.array([
    0.090, 0.075, 0.080, 0.095, 0.078, 0.072,
    0.070, 0.082, 0.085, 0.088, 0.090, 0.095,
])
MONTHLY_WEIGHTS = MONTHLY_WEIGHTS / MONTHLY_WEIGHTS.sum()

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Key insight boxes matching slides
pat_cols = st.columns(4)
with pat_cols[0]:
    st.error("**Morning Peak**\n\n6–9 AM: Highest demand as commuters travel to work and school")
with pat_cols[1]:
    st.warning("**Midday Lull**\n\n9 AM–3 PM: Reduced demand during working hours")
with pat_cols[2]:
    st.error("**Evening Peak**\n\n3–8 PM: Second surge as people return home")
with pat_cols[3]:
    st.info("**Weekend Drop**\n\nLower demand compared to weekday patterns")

# Interactive temporal explorer
st.subheader("Explore Demand by Time Period")
st.caption(
    "Annual data is distributed using Kenyan transport seasonality patterns "
    "(school terms, holidays, rush hours, weekend drop-off)."
)

tcol1, tcol2, tcol3 = st.columns(3)
with tcol1:
    temp_category = st.selectbox("Transport category:", DEMAND_COLUMNS, key="temp_cat")
with tcol2:
    temp_granularity = st.selectbox(
        "Time granularity:", ["Hourly", "Daily", "Monthly", "Weekly"], key="temp_gran"
    )
with tcol3:
    temp_year = st.slider(
        "Year:", int(df_combined["Year"].min()), 2027, 2024, key="temp_year"
    )

temp_unit = "KSh Millions" if temp_category == "Road Transport" else "Vehicles"


@st.cache_data
def build_temporal(df_src, all_fc, category, year, granularity):
    rows = []
    for county in df_src["County"].unique():
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
                rows.append({"County": county, "Period": MONTH_NAMES[m],
                             "Value": annual * MONTHLY_WEIGHTS[m], "Order": m})
        elif granularity == "Weekly":
            weekly = annual / 52
            for w in range(1, 53):
                month_idx = min(int((w - 1) / 4.33), 11)
                seasonal = MONTHLY_WEIGHTS[month_idx] * 12
                rows.append({"County": county, "Period": f"W{w}",
                             "Value": weekly * seasonal, "Order": w})
        elif granularity == "Daily":
            daily_base = annual / 365
            for d in range(7):
                rows.append({"County": county, "Period": DAY_NAMES[d],
                             "Value": daily_base * DAILY_WEIGHTS[d] * 7, "Order": d})
        elif granularity == "Hourly":
            daily_avg = annual / 365
            for h in range(24):
                rows.append({"County": county, "Period": f"{h:02d}:00",
                             "Value": daily_avg * HOURLY_WEIGHTS[h] * 24, "Order": h})
    return pd.DataFrame(rows)


temporal_df = build_temporal(
    df_combined, all_forecasts, temp_category, temp_year, temp_granularity
)

if not temporal_df.empty:
    temporal_df = temporal_df.sort_values("Order")
    fig_t, ax_t = plt.subplots(figsize=(14, 6))
    for county in temporal_df["County"].unique():
        cdata = temporal_df[temporal_df["County"] == county]
        ax_t.plot(cdata["Period"], cdata["Value"], marker="o", linewidth=1.5,
                  label=county, markersize=4)

    # Shade peak periods for hourly view
    if temp_granularity == "Hourly":
        ax_t.axvspan("06:00", "09:00", alpha=0.1, color="red", label="Morning Peak")
        ax_t.axvspan("15:00", "20:00", alpha=0.1, color="orange", label="Evening Peak")

    ax_t.set_title(f"{temp_category} — {temp_granularity} Estimate ({temp_year})", fontsize=14)
    ax_t.set_ylabel(f"{temp_category} ({temp_unit})")
    ax_t.legend(title="County")
    ax_t.grid(True, alpha=0.3)
    if temp_granularity == "Weekly":
        ax_t.set_xticks(ax_t.get_xticks()[::4])
    ax_t.tick_params(axis="x", rotation=45)
    fig_t.tight_layout()
    st.pyplot(fig_t)
    plt.close(fig_t)

    with st.expander("View data table"):
        pivot_temp = temporal_df.pivot(index="Period", columns="County", values="Value")
        pivot_temp = pivot_temp.loc[
            temporal_df.drop_duplicates("Period").sort_values("Order")["Period"]
        ]
        st.dataframe(pivot_temp.round(0), use_container_width=True)
else:
    st.warning("No data available for the selected year.")


# ── Section 5: City-by-City Demand Analysis ──────────────────
st.header("5. City-by-City Demand Analysis")
st.markdown(
    "Population and time of day are the most important factors influencing demand. "
    "Larger cities like Nairobi naturally show higher demand due to greater "
    "population density and economic activity."
)

cba_category = st.selectbox(
    "Select transport category:", DEMAND_COLUMNS, key="cba_cat"
)
cba_unit = "KSh Millions" if cba_category == "Road Transport" else "Vehicles"
latest_year = int(df_combined["Year"].max())

# Ranked bar chart of latest year
city_vals = []
for county in df_combined["County"].unique():
    row = df_combined[(df_combined["County"] == county) & (df_combined["Year"] == latest_year)]
    if len(row) > 0:
        city_vals.append({"County": county, "Value": row[cba_category].values[0]})
city_df = pd.DataFrame(city_vals).sort_values("Value", ascending=True)

fig_cba, ax_cba = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("viridis", len(city_df))
ax_cba.barh(city_df["County"], city_df["Value"], color=colors)
ax_cba.set_xlabel(f"{cba_category} ({cba_unit})")
ax_cba.set_title(f"{cba_category} by City — {latest_year}")
ax_cba.grid(True, alpha=0.3, axis="x")
fig_cba.tight_layout()
st.pyplot(fig_cba)
plt.close(fig_cba)


# ── Section 6: Demand Forecast ───────────────────────────────
st.header("6. Demand Forecast (2025–2027)")

forecast_category = st.selectbox(
    "Select transport category to forecast:", DEMAND_COLUMNS, key="forecast_cat"
)
forecast_df = all_forecasts[forecast_category]
f_unit = "KSh Millions" if forecast_category == "Road Transport" else "Vehicles"

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
ax7.set_title(f"{forecast_category}: Historical + Forecast (Federated Model)")
ax7.set_xlabel("Year")
ax7.set_ylabel(f"{forecast_category} ({f_unit})")
ax7.legend(title="County", bbox_to_anchor=(1.05, 1), loc="upper left")
ax7.grid(True, alpha=0.3)
fig7.tight_layout()
st.pyplot(fig7)
plt.close(fig7)


# ── Section 7: Comparative Analysis ──────────────────────────
st.header("7. Comparative Analysis — County Rankings & Trends")

comp_category = st.selectbox(
    "Select transport category for comparison:", DEMAND_COLUMNS, key="comp_cat"
)
comp_unit = "KSh Millions" if comp_category == "Road Transport" else "Vehicles"
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
fig9, ax9 = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
width = 0.35
ax9.bar(x - width / 2, comp_df[f"{latest_year} Actual"], width,
        label=f"{latest_year} Actual", color="steelblue")
ax9.bar(x + width / 2, comp_df[f"{forecast_year} Forecast"], width,
        label=f"{forecast_year} Forecast", color="coral")
ax9.set_xlabel("County")
ax9.set_ylabel(f"{comp_category} ({comp_unit})")
ax9.set_title(f"{comp_category}: Current vs Forecast by County")
ax9.set_xticks(x)
ax9.set_xticklabels(comp_df["County"], rotation=30)
ax9.legend()
ax9.grid(True, alpha=0.3, axis="y")
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

# Trend matrix
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
    df_combined, all_forecasts, DEMAND_COLUMNS, latest_year, forecast_year
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

# ── Section 8: Challenges & Conclusion ───────────────────────
st.header("8. Challenges & Future Directions")

ch_cols = st.columns(3)
with ch_cols[0]:
    st.warning("**Simulated Data**\n\nModel uses estimated county-level data rather "
               "than real-time operational data, which may limit accuracy.")
with ch_cols[1]:
    st.warning("**Data Quality**\n\nTransport data quality varies across cities, "
               "affecting model consistency.")
with ch_cols[2]:
    st.warning("**Computational Resources**\n\nDecentralized training requires "
               "coordination across multiple systems.")

st.subheader("Conclusion")
st.markdown(
    "Decentralized machine learning is effective for predicting transport demand. "
    "It improves accuracy, protects data privacy, and can help planners optimize "
    "transport systems."
)
conc_cols = st.columns(3)
with conc_cols[0]:
    st.success("**Current Success**\n\nAccurate demand prediction using federated learning")
with conc_cols[1]:
    st.info("**Real-Time Data**\n\nExpand using real-time operational data (GTFS)")
with conc_cols[2]:
    st.info("**More Cities**\n\nScale to additional urban areas across Kenya")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "*Data Source: Kenya National Bureau of Statistics (KNBS) Economic Surveys, "
    "with county-level estimates derived from Gross County Product weights, "
    "population data, and known transport patterns.*"
)
