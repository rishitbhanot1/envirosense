import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier

# ---------------------------
# 1) Dataset upload / load
# ---------------------------
print("📂 Please upload your dataset...")

try:
    from google.colab import files
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)
except Exception:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        df = pd.read_csv(file_path)
    except Exception:
        file_path = "global_aqi_2020_2025.csv"
        print(f"No File Uploaded by User; Using Preloaded File: {file_path}")
        df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully!")

# ---------------------------
# 2) Parse Date (if present)
# ---------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    print("⚠️ 'Date' column not found. Predictions that require exact date will use proxies instead.")

# ---------------------------
# 2b) Fill NaNs using neighbor-median (3 before + 3 after)
# ---------------------------
def fillna_with_neighbor_median(df, cols=None, n=3, groupby=None, sort_by=None, inplace=False):
    window = 2 * n + 1
    df_out = df if inplace else df.copy()
    if cols is None:
        cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    def _apply_block(block):
        if sort_by is not None:
            block = block.sort_values(sort_by)
        for c in cols:
            med = block[c].rolling(window=window, center=True, min_periods=1).median()
            block[c] = block[c].fillna(med)
        if sort_by is not None:
            block = block.sort_index()
        return block
    if groupby is not None:
        df_out = df_out.groupby(groupby, group_keys=False).apply(_apply_block)
    else:
        df_out = _apply_block(df_out)
    return df_out

# Apply neighbor-median fill per City, ordered by Date
pollutants = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity"]
df = fillna_with_neighbor_median(df, cols=pollutants, n=3, groupby="City", sort_by="Date")

# ---------------------------
# 3) CPCB AQI Sub-Index Calculation
# ---------------------------
breakpoints = {
    "PM2.5": [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
    "PM10": [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
    "NO2": [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,1000,401,500)],
    "SO2": [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)],
    "CO": [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)],
    "O3": [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)]
}

def calc_sub_index(pollutant, value):
    if pd.isna(value) or pollutant not in breakpoints:
        return np.nan
    for (BP_LO, BP_HI, I_LO, I_HI) in breakpoints[pollutant]:
        if BP_LO <= value <= BP_HI:
            return ((I_HI - I_LO) / (BP_HI - BP_LO)) * (value - BP_LO) + I_LO
    return np.nan

pollutants = ["PM2.5","PM10","NO2","SO2","CO","O3"]
for p in pollutants:
    if p not in df.columns:
        raise KeyError(f"Missing pollutant column: {p}")
    df[p] = pd.to_numeric(df[p], errors="coerce")
    df[f"{p}_subindex"] = df[p].apply(lambda x: calc_sub_index(p,x))

df["AQI"] = df[[f"{p}_subindex" for p in pollutants]].max(axis=1)
print("\n✅ AQI Computation Done!")

# ---------------------------
# 4) Features
# ---------------------------
feature_numeric = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity"]
for col in feature_numeric:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median(skipna=True))

features = feature_numeric + ["City"]
X_raw = df[features].copy()
y = df["AQI"]

X = pd.get_dummies(X_raw, columns=["City"], drop_first=True)
city_series_lower = df["City"].astype(str).str.lower()

X_train, X_test, y_train, y_test, city_train, city_test = train_test_split(X,y,city_series_lower,test_size=0.2,random_state=42)

# ---------------------------
# 5) Models
# ---------------------------
results = pd.DataFrame(columns=["MAE","RMSE","R²","Overall_Confidence_%"])
per_city_mae = {}

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
}

# compact skill-vs-city-mean confidence + per-city MAE
city_mean_map = pd.Series(y_train.values, index=city_train.index).groupby(city_train).mean().to_dict()
global_median = y_train.median()
baseline_preds = city_test.map(lambda c: city_mean_map.get(c, global_median)).values
mae_baseline = mean_absolute_error(y_test, baseline_preds)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    skill = (mae_baseline - mae) / max(1e-6, mae_baseline)
    conf = float(np.clip(skill * 100.0, 0.0, 100.0))

    results.loc[name] = [round(mae, 3), round(rmse, 3), round(r2, 4), round(conf, 2)]

    y_pred_s = pd.Series(y_pred, index=y_test.index)
    per_city_mae[name] = {
        c: mean_absolute_error(y_test.loc[idx], y_pred_s.loc[idx])
        for c in np.unique(city_test.values)
        for idx in [city_test[city_test == c].index]
        if len(idx) >= 3
    }

print("\n📊 Regression Performance:")
print(results)

# ---------------------------
# 5b) Visualization: R² and Regression Fit
# ---------------------------
plt.figure(figsize=(8,5))
plt.bar(results.index, results["R²"], color="skyblue", edgecolor="black")
plt.ylim(0,1.05)
plt.title("Model R² Scores")
plt.ylabel("R²")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor="k")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect Fit")
    plt.title(f"Regression Fit: {name}")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

# ---------------------------
# 6) Classification for AQI Category
# ---------------------------
def categorize_aqi(aqi):
    if aqi<=50: return "Good"
    elif aqi<=100: return "Satisfactory"
    elif aqi<=200: return "Moderate"
    elif aqi<=300: return "Poor"
    elif aqi<=400: return "Very Poor"
    else: return "Severe"

df["AQI_Category"] = df["AQI"].apply(categorize_aqi)
label_encoder = LabelEncoder()
y_class_series = pd.Series(label_encoder.fit_transform(df["AQI_Category"].astype(str)), index=df.index)

y_train_c = y_class_series.loc[X_train.index]
y_test_c = y_class_series.loc[X_test.index]

clf_models = {
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        random_state=42, eval_metric="mlogloss")
}

for name, clf in clf_models.items():
    clf.fit(X_train,y_train_c)
    y_pred_c = clf.predict(X_test)
    print(f"\n📌 {name} Classification Report:")
    print(classification_report(y_test_c,y_pred_c,target_names=label_encoder.classes_))

# ---------------------------
# 7) AQI Proxy Functions (Context Only)
# ---------------------------
def build_proxy_row(city_name_lower, target_date):
    """Build proxy row for pollutants + AQI with detailed fallback reporting (context only)."""
    city_df = df[df["City"].astype(str).str.lower() == city_name_lower]

    if city_df.empty:
        proxy = {col: df[col].median(skipna=True) for col in feature_numeric}
        proxy["City"] = city_name_input
        proxy["AQI"] = np.nan
        print(f"⚠️ No city data for {city_name_input}. Using global pollutant medians. AQI cannot be proxied.")
        return proxy

    proxy, sources = {}, {}
    month = target_date.month if pd.notna(target_date) else None

    # Pollutant context
    for col in feature_numeric:
        val, source = np.nan, None
        if month is not None:
            subset = city_df.loc[city_df["Date"].dt.month == month, col]
            if not subset.empty:
                mean, std = subset.mean(skipna=True), subset.std(skipna=True)
                val = np.random.normal(mean, std * 0.2)
                source = f"Seasonal Avg ({calendar.month_name[month]})"
        if (pd.isna(val)) and pd.notna(target_date):
            past_week = city_df[(city_df["Date"] < target_date) &
                                (city_df["Date"] >= target_date - pd.Timedelta(days=7))]
            if not past_week.empty:
                val = past_week[col].mean(skipna=True)
                if not np.isnan(val):
                    source = "7-Day Lag"
        if (pd.isna(val)) and month is not None:
            tmp = city_df.dropna(subset=["Date"]).copy()
            tmp["Year"] = tmp["Date"].dt.year
            tmp["Month"] = tmp["Date"].dt.month
            recent = tmp[tmp["Month"] == month]
            if not recent.empty:
                yearly = recent.groupby("Year")[col].mean().reset_index()
                if len(yearly) >= 2:
                    coeffs = np.polyfit(yearly["Year"], yearly[col], deg=1)
                    val = np.polyval(coeffs, target_date.year)
                    source = f"Yearly Trend ({calendar.month_name[month]})"
                else:
                    val = yearly[col].mean(skipna=True)
                    source = f"Yearly Avg ({calendar.month_name[month]})"
        if np.isnan(val):
            val = city_df[col].mean(skipna=True)
            if not np.isnan(val):
                source = "City Mean"
        proxy[col], sources[col] = val, source

    # AQI context
    aqi_val, fallback = np.nan, None
    if month is not None:
        subset = city_df.loc[city_df["Date"].dt.month == month, "AQI"]
        if not subset.empty:
            mean, std = subset.mean(skipna=True), subset.std(skipna=True)
            aqi_val = np.random.normal(mean, std * 0.2)
            fallback = f"Seasonal Avg AQI ({calendar.month_name[month]}) [{subset.min():.1f}–{subset.max():.1f}]"
    if pd.isna(aqi_val) and pd.notna(target_date):
        past = city_df[(city_df["Date"] < target_date) &
                       (city_df["Date"] >= target_date - pd.Timedelta(days=7))]
        if not past.empty:
            aqi_val = past["AQI"].mean(skipna=True)
            fallback = f"7-Day Lag AQI [{past['AQI'].min():.1f}–{past['AQI'].max():.1f}]"
    if pd.isna(aqi_val) and month is not None:
        tmp = city_df.dropna(subset=["Date"]).copy()
        tmp["Year"] = tmp["Date"].dt.year
        tmp["Month"] = tmp["Date"].dt.month
        recent = tmp[tmp["Month"] == month]
        if not recent.empty:
            yearly = recent.groupby("Year")["AQI"].mean().reset_index()
            if len(yearly) >= 2:
                coeffs = np.polyfit(yearly["Year"], yearly["AQI"], deg=1)
                aqi_val = np.polyval(coeffs, target_date.year)
                fallback = f"Yearly Trend AQI ({calendar.month_name[month]})"
            else:
                aqi_val = yearly["AQI"].mean(skipna=True)
                fallback = f"Yearly Avg AQI ({calendar.month_name[month]})"
    if pd.isna(aqi_val):
        aqi_val = city_df["AQI"].mean(skipna=True)
        fallback = "City Mean AQI"

    proxy["AQI"] = aqi_val
    proxy["City"] = city_name_input

    # Context report
    print(f"\n🔎 Proxy Context for {city_name_input} on {target_date.date() if pd.notna(target_date) else 'N/A'}:")
    print("➡️ Pollutant sources:")
    for col in feature_numeric:
        print(f"   - {col}: {proxy[col]:.2f} ({sources[col]})")
    print(f"\n➡️ AQI Context: {aqi_val:.2f} ({fallback})\n")

    return proxy

# ---------------------------
# 8) Prediction flow
# ---------------------------
city_name_input = input("Enter the City Name for AQI Prediction: ").strip()
date_input = input("Enter the Date (YYYY-MM-DD) for Prediction: ").strip()

city_name_lower = city_name_input.strip().lower()
target_date = pd.to_datetime(date_input, errors="coerce") if date_input else pd.NaT

if ("Date" in df.columns) and pd.notna(target_date):
    row_exact = df[
        (df["City"].astype(str).str.lower() == city_name_lower)
        & (df["Date"].dt.date == target_date.date())
    ]
else:
    row_exact = pd.DataFrame()

if not row_exact.empty:
    base_row = row_exact.iloc[0]
    print(f"\n🔎 Found exact entry for {city_name_input} on {date_input}. Using its pollutant readings.")
else:
    print(f"\nℹ️ No exact entry for {city_name_input} on {date_input}. Showing proxy context (not used for prediction)...")
    base_row = pd.Series(build_proxy_row(city_name_lower, target_date))

X_pred = pd.DataFrame([base_row[feature_numeric + ["City"]]])
X_pred = pd.get_dummies(X_pred, columns=["City"], drop_first=True)
X_pred = X_pred.reindex(columns=X.columns, fill_value=0)

def categorize_aqi_value(aqi_val):
    if aqi_val <= 50: return "Good"
    elif aqi_val <= 100: return "Satisfactory"
    elif aqi_val <= 200: return "Moderate"
    elif aqi_val <= 300: return "Poor"
    elif aqi_val <= 400: return "Very Poor"
    else: return "Severe"

print(f"\n🔎 Predicting AQI for {city_name_input} on {date_input}...\n")

predictions_summary = []
for name, model in models.items():
    y_hat = float(model.predict(X_pred)[0])
    overall_mae = float(results.loc[name, "MAE"])
    city_mae = per_city_mae.get(name, {}).get(city_name_lower, overall_mae)
    denom = max(1e-6, y.mean())
    city_conf = float(max(0.0, min(100.0, 100.0 - (city_mae / denom) * 100.0)))
    cat = categorize_aqi_value(y_hat)
    predictions_summary.append({
        "Model": name,
        "Predicted_AQI": y_hat,
        "City_MAE_used": round(city_mae, 3),
        "Confidence_%": round(city_conf, 2),
        "Category": cat
    })
    print(f"📌 Model: {name}")
    print(f"   ➤ Predicted AQI: {y_hat:.2f}")
    print(f"   ➤ AQI Category: {cat}")
    print(f"   ➤ Confidence (city-aware): {city_conf:.2f}% (city MAE = {city_mae:.3f})\n")

print("📌 Classifier category predictions (with probabilities):")
for name, clf in clf_models.items():
    try:
        class_pred = clf.predict(X_pred)[0]
        class_name = label_encoder.inverse_transform([int(class_pred)])[0]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_pred)[0]
            prob_map = dict(zip(label_encoder.classes_, (probs * 100).round(2)))
        else:
            prob_map = {}
    except Exception as e:
        class_name = "N/A"
        prob_map = {}
        print(f"⚠️ Classifier {name} prediction error: {e}")
    print(f"\n{name}:")
    print(f"   ➤ Predicted AQI Category: {class_name}")
    if prob_map:
        print(f"   ➤ Probabilities: {prob_map}")

best_model = max(predictions_summary, key=lambda x: x["Confidence_%"])
print("\n🏆 Best Model Recommendation:")
print(f"👉 {best_model['Model']} with {best_model['Confidence_%']:.2f}% confidence.")
print(f"   ➤ Predicted AQI: {best_model['Predicted_AQI']:.2f}")
print(f"   ➤ AQI Category: {best_model['Category']}")

print("\n✅ Prediction complete.")
