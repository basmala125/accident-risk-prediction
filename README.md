# 🚗 Accident Risk Prediction

A machine learning project that predicts road accident risk based on road, environmental, and situational features. Built as part of my graduation project, using the [Kaggle Playground Series S5E10](https://www.kaggle.com/competitions/playground-series-s5e10) dataset.

## 🔍 Overview

This project trains a tuned **XGBoost Regressor** to estimate accident risk from a set of road and environmental conditions, and deploys the model through an interactive **Streamlit** web app for real-time predictions.

## 🧠 Model

- **Algorithm:** XGBoost Regressor (`XGBRegressor`)
- **Task:** Regression (predicting a continuous accident risk score)
- **Tuning:** Hyperparameter-tuned (learning rate, max depth, n_estimators, etc.)

## 📊 Features Used

| Feature | Description |
|---|---|
| `num_lanes` | Number of lanes on the road |
| `curvature` | Road curvature (0–1) |
| `speed_limit` | Posted speed limit |
| `num_reported_accidents` | Historical reported accidents on the road |
| `road_signs_present` | Whether road signs are present |
| `public_road` | Whether the road is public |
| `holiday` | Whether it's a holiday |
| `school_season` | Whether school is in session |
| `road_type` | Urban / Rural / Highway |
| `lighting` | Daylight / Dim / Night |
| `weather` | Clear / Rainy / Foggy |
| `time_of_day` | Morning / Afternoon / Evening |

## 🗂️ Project Structure

```
accident-risk-project/
├── app.py                          # Streamlit web app
├── final_tuned_xgboost_model.pkl   # Trained & tuned XGBoost model
├── feature_columns.pkl             # Feature columns used by the model (post-encoding)
├── category_values.pkl             # Reference values for categorical features
├── requirements.txt                # Python dependencies
└── README.md
```

## ⚙️ Setup & Usage

1. **Clone the repository**
```bash
git clone https://github.com/basmala125/accident-risk-prediction.git
cd accident-risk-prediction
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 🖥️ How It Works

1. The user inputs road, weather, and situational details through the web interface.
2. Categorical features are one-hot encoded to match the model's training format.
3. The trained XGBoost model predicts an accident risk score.
4. The result is displayed instantly in the app.

## 🛠️ Tech Stack

- Python
- XGBoost
- Pandas
- Streamlit
- scikit-learn

## 📌 Notes

This project was developed as part of an ongoing machine learning graduation project, focused on applying regression techniques to real-world tabular data with mixed categorical and numerical features.

## 📄 License

This project is for educational purposes.
