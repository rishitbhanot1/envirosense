import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Google Colab
    from google.colab import files
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)
except Exception:
    # Local Jupyter / script (tkinter may fail on headless servers)
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv")]
        )
        df = pd.read_csv(file_path)
    except Exception:
        # Fallback: try an expected path (helpful for automated runs)
        file_path = "/mnt/data/global_air_quality_data_10000.csv"
        print(f"Could not open dialog; trying fallback path: {file_path}")
        df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully!")
print(df.head())

# ---------------------------
# 2) Parse Date (if present)
# ---------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    print("⚠️ 'Date' column not found. Predictions that require exact date will use medians instead.")

# ---------------------------
# 3) CPCB AQI breakpoints & sub-index calculation
# ---------------------------
breakpoints = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
             (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
            (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
            (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
           (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
           (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)]
}

def calc_sub_index(pollutant, value):
    """Calculate pollutant sub-index using CPCB breakpoints."""
    if pd.isna(value):
        return np.nan
    if pollutant not in breakpoints:
        return np.nan
    for (BP_LO, BP_HI, I_LO, I_HI) in breakpoints[pollutant]:
        if BP_LO <= value <= BP_HI:
            return ((I_HI - I_LO) / (BP_HI - BP_LO)) * (value - BP_LO) + I_LO
    return np.nan

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Ensure pollutant numeric conversion and create subindexes
for p in pollutants:
    if p not in df.columns:
        raise KeyError(f"Required pollutant column '{p}' not found in dataset.")
    df[p] = pd.to_numeric(df[p], errors="coerce")
    df[f"{p}_subindex"] = df[p].apply(lambda x: calc_sub_index(p, x))

df["AQI"] = df[[f"{p}_subindex" for p in pollutants]].max(axis=1)

print("\n✅ AQI Computation Done!")
print(df[["City"] + pollutants + ["AQI"]].head())

# ---------------------------
# 4) Prepare features (median imputation + one-hot City)
# ---------------------------
feature_numeric = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]
# Ensure numeric columns exist, if not fill with global median of available columns or create column
for col in feature_numeric:
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found — creating with global median = 0")
        df[col] = np.nan

# Convert to numeric and fill medians (robust)
for col in feature_numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    median_val = df[col].median(skipna=True)
    if np.isnan(median_val):
        median_val = 0.0
    df[col] = df[col].fillna(median_val)

# Prepare X and y
features = feature_numeric + ["City"]
X_raw = df[feature_numeric + ["City"]].copy()
y = df["AQI"].copy()

# One-hot encode City (drop_first avoids multicollinearity; it's okay for tree models too)
X = pd.get_dummies(X_raw, columns=["City"], drop_first=True)

# Keep a lowercase city series (for per-city metrics and matching)
city_series_lower = df["City"].astype(str).str.lower()

# ---------------------------
# 5) Train/test split (preserve city_series for per-city MAE)
# ---------------------------
X_train, X_test, y_train, y_test, city_train, city_test = train_test_split(
    X, y, city_series_lower, test_size=0.2, random_state=42
)

# ---------------------------
# 6) Regression training + per-city MAE
# ---------------------------
results = pd.DataFrame(columns=["MAE", "RMSE", "R²", "Overall_Confidence_%"])
per_city_mae = {}  # model_name -> {city_lower: mae}

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
}

denom_for_conf = max(1e-6, y_test.mean())

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    overall_conf = float(max(0.0, min(100.0, 100.0 - (mae / denom_for_conf) * 100.0)))

    results.loc[name] = [round(mae, 3), round(rmse, 3), round(r2, 4), round(overall_conf, 2)]

    # compute per-city MAE for cities with >= 3 samples in test set
    per_city_mae[name] = {}
    y_pred_s = pd.Series(y_pred, index=y_test.index)
    # city_test is Series containing lowercase city names (with same index)
    for c in np.unique(city_test.values):
        idx = city_test[city_test == c].index
        if len(idx) >= 3:
            per_city_mae[name][c] = mean_absolute_error(y_test.loc[idx], y_pred_s.loc[idx])

    print(f"✅ {name} trained: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.4f}, OverallConf={overall_conf:.2f}%")

print("\n📊 Regression Model Performance Comparison")
print(results)

# Plot R²
plt.figure(figsize=(9,5))
sns.barplot(x=results.index, y=results["R²"].astype(float), palette="viridis")
plt.title("R² Score Comparison", fontsize=14, fontweight="bold")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 7) Classification (AQI categories)
# ---------------------------
def categorize_aqi(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

df["AQI_Category"] = df["AQI"].apply(categorize_aqi)

# Label encode categories (for classifier training)
label_encoder = LabelEncoder()
y_class_series = pd.Series(label_encoder.fit_transform(df["AQI_Category"].astype(str)), index=df.index)

# Use the same X_train/X_test split indices for classification labels
y_train_c = y_class_series.loc[X_train.index]
y_test_c = y_class_series.loc[X_test.index]

clf_models = {
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        random_state=42, use_label_encoder=False, eval_metric="mlogloss")
}

for name, clf in clf_models.items():
    clf.fit(X_train, y_train_c)
    y_pred_c = clf.predict(X_test)
    print(f"\n📌 {name} Classification Report:")
    print(classification_report(y_test_c, y_pred_c, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test_c, y_pred_c)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

print("\n✅ Data Processing & Training Completed Successfully!\n")

# ---------------------------
# 8) Prediction flow: prompt user for city + date, build proxy if needed
# ---------------------------
def smart_city_month_median(city_df, month, col):
    """Return city+month median, fallback to city median, then global median."""
    v = np.nan
    try:
        if "Date" in city_df.columns and city_df["Date"].notna().any() and month is not None:
            v = city_df.loc[city_df["Date"].dt.month == month, col].median(skipna=True)
            if not np.isnan(v):
                return v
        # city overall median
        v = city_df[col].median(skipna=True)
        if not np.isnan(v):
            return v
    except Exception:
        pass
    # global median fallback
    return df[col].median(skipna=True)

# Input from user
city_name_input = input("Enter the City Name for AQI Prediction: ").strip()
date_input = input("Enter the Date (YYYY-MM-DD) for Prediction: ").strip()

# Normalize name for matching
city_name_lower = city_name_input.strip().lower()
target_date = pd.to_datetime(date_input, errors="coerce") if date_input.strip() != "" else pd.NaT

# Try exact matching first (city + date)
if ("Date" in df.columns):
    if pd.notna(target_date):
        row_exact = df[(df["City"].astype(str).str.lower() == city_name_lower) & (df["Date"].dt.date == target_date.date())]
    else:
        row_exact = pd.DataFrame()
else:
    row_exact = pd.DataFrame()

if not row_exact.empty:
    base_row = row_exact.iloc[0]
    print(f"\n🔎 Found exact entry for {city_name_input} on {date_input}. Using its pollutant readings.")
else:
    # Build proxy row from medians
    print(f"\nℹ️ No exact entry for {city_name_input} on {date_input}. Building proxy row from historical medians.")
    city_df = df[df["City"].astype(str).str.lower() == city_name_lower]
    month = target_date.month if pd.notna(target_date) else None

    proxy = {}
    for col in feature_numeric:
        # get city+month median -> city median -> global median
        if not city_df.empty:
            proxy[col] = smart_city_month_median(city_df, month, col)
        else:
            proxy[col] = df[col].median(skipna=True)
    proxy["City"] = city_name_input  # preserve original case for one-hot
    base_row = pd.Series(proxy)

# Build X_pred and one-hot encode City -> align with training X columns
X_pred = pd.DataFrame([base_row[feature_numeric + ["City"]]])
X_pred = pd.get_dummies(X_pred, columns=["City"], drop_first=True)
X_pred = X_pred.reindex(columns=X.columns, fill_value=0)

# Category helper
def categorize_aqi_value(aqi_val):
    if aqi_val <= 50: return "Good"
    elif aqi_val <= 100: return "Satisfactory"
    elif aqi_val <= 200: return "Moderate"
    elif aqi_val <= 300: return "Poor"
    elif aqi_val <= 400: return "Very Poor"
    else: return "Severe"

print(f"\n🔎 Predicting AQI for {city_name_input} on {date_input}...\n")

# Regression model predictions + city-aware confidence
predictions_summary = []

for name, model in models.items():
    y_hat = float(model.predict(X_pred)[0])

    overall_mae = float(results.loc[name, "MAE"])
    # attempt to get per-city MAE (keys are lowercase city names)
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
    print(f"   ➤ Confidence (city-aware): {city_conf:.2f}% (using city MAE = {city_mae:.3f})\n")

# Classifier predictions for AQI category and probabilities
print("📌 Classifier category predictions (with probabilities):")
for name, clf in clf_models.items():
    # if classifier not trained for some reason, skip
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

# Best-model recommendation (highest confidence)
best_model = max(predictions_summary, key=lambda x: x["Confidence_%"])
print("\n🏆 Best Model Recommendation:")
print(f"👉 {best_model['Model']} with {best_model['Confidence_%']:.2f}% confidence.")
print(f"   ➤ Predicted AQI: {best_model['Predicted_AQI']:.2f}")
print(f"   ➤ AQI Category: {best_model['Category']}")

# Done
print("\n✅ Prediction complete.")
