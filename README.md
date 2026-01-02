
---

# 🌍 EnviroSense

*Smart Air Quality Prediction & Insights*

![status](https://img.shields.io/badge/status-deprioritized-red) [![version](https://img.shields.io/badge/version-v0.3-blue)](https://github.com/rishitbhanot1/envirosense/releases/tag/tiramisu) [![version](https://img.shields.io/badge/script-tiramisu-orange)](https://github.com/rishitbhanot1/envirosense/blob/main/scripts/v3_tiramisu.py) [![Python](https://img.shields.io/badge/Python-3.9%2B-violet)](https://www.python.org/) [![USP](https://img.shields.io/badge/USP-AQI%20Auto--Computation%20✅-purple)](https://github.com/rishitbhanot1/envirosense?tab=readme-ov-file#-features)
 [![issues](https://img.shields.io/github/issues/rishitbhanot1/envirosense)](https://github.com/rishitbhanot1/envirosense/issues)



---

## 📖 Overview

**EnviroSense** is a machine learning-powered platform designed to **predict Air Quality Index (AQI)** for cities worldwide.
It processes pollutant data, applies **standardized AQI breakpoints (CPCB/US EPA)**, and uses multiple models (Linear Regression, Decision Tree, Random Forest, XGBoost) to provide accurate AQI predictions along with confidence levels.

The tool is aimed at:
✔️ Environmental researchers
✔️ Policy makers
✔️ Smart city planners
✔️ Everyday citizens who care about clean air

---

## Project Status

This project is **completed** and **no longer actively maintained**.  
Development concluded during pre-release, and the repository has since been **deprioritized**. No further updates are planned and no additional changes are expected.

---

## ✨ Features

* 📂 **Dataset ingestion**: Upload your own AQI dataset.
* ⚗️ **Data preprocessing**: Missing values handled, pollutants standardized.
* 📊 **AQI computation**: Based on pollutants (PM2.5, PM10, NO₂, SO₂, CO, O₃).
* 🔑 **CPCB Breakpoints**: Using pollutant breakpoints and official computing standards from Central Pollution Control Board.
* ❤️ **Differentiator**: Backend self computation of AQI, so you can use the raw pollutant dataset even without the AQI to train.
* 🧠 **Multiple ML Models**:

  * Linear Regression
  * Decision Tree
  * Random Forest Regressor
  * XGBoostRegressor
    
* ✅ **Model Evaluation**: MAE, RMSE, R², confidence percentage.
* 🔮 **Prediction mode**: Enter **city** + **date** → Get AQI predictions across models.
* 🏆 **Best model recommendation** with confidence score.
* 🎨 **Visualizations**: R² comparison, confusion matrices, performance metrics.

---

## 🛠️ Tech Stack

* **Languages**: Python 3.9+
* **Libraries**:

  * Data: `pandas`, `numpy`
  * Visualization: `matplotlib`, `seaborn`
  * ML Models: `scikit-learn`, `xgboost`
  * UI/File Handling: `tkinter`, `google.colab` (for Colab support)
  * Date Handling: `calendar` 

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rishitbhanot1/envirosense.git
cd envirosense
```

### 2️⃣ Install Requirements

```bash
pip install -r model_requirements.txt
```

### 3️⃣ Run the Script

```bash
python envirosense.py
```

---

## 📂 Dataset Format

The dataset should include at least:

| Date       | City     | PM2.5 | PM10 | NO2 | SO2 | CO  | O3 | Temperature | Humidity |
| ---------- | -------- | ----- | ---- | --- | --- | --- | -- | ----------- | -------- |
| 2025-08-01 | Madrid   | 22    | 55   | 18  | 9   | 0.6 | 35 | 28          | 62       |
| 2025-08-02 | New York | 30    | 70   | 25  | 12  | 0.8 | 42 | 30          | 58       |

---

## 📊 Sample Output

### 🔎 Predictions for *Madrid* on *2025-08-30*:

```
📌 Model: Random Forest
   ➤ Predicted AQI: 62.40
   ➤ Confidence: 91.45%
   ➤ AQI Category: Satisfactory

📌 Model: XGBoost
   ➤ Predicted AQI: 59.20
   ➤ Confidence: 89.87%
   ➤ AQI Category: Satisfactory

🏆 Best Model Recommendation:
👉 Random Forest with 91.45% confidence.
   ➤ Predicted AQI: 62.40
   ➤ AQI Category: Satisfactory
```

---

## 📈 Visualizations

* 📊 **R² Score Comparison** (Bar Plot)
* 🔥 **Confusion Matrix** for AQI categories
* 📉 **Error metrics** (MAE, RMSE)

---

## 🧩 Roadmap

* [ ] 🌐 Live AQI API integration
* [ ] 📱 Web Dashboard (Streamlit/Flask)
* [ ] 📌 Geo-mapping of AQI predictions
* [ ] 🤖 Hyperparameter optimization for models

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo, make your changes, and submit a pull request.

---

💡 *Breathe Easy. Predict Smart. Choose EnviroSense.* 🌿

---
