# Filipino Restaurant Dish Demand Forecasting

An end-to-end machine learning system that predicts daily serving demand for 16 Filipino dishes at a restaurant, based on day of week, weather, and holiday status — helping minimize food waste and avoid shortages.

Built for **IPT 101 – Integrative Programming and Technologies** (Task Performance #4), Pamantasan ng Lungsod ng Valenzuela, BSIT 3-2.

---

## Overview

Restaurants often over- or under-prepare dishes because demand is hard to predict day to day. This project builds a supervised regression model trained on historical sales data, then serves it through a REST API and a simple web interface — so a restaurant owner can get a same-day forecast for a single dish or the full 16-dish menu at once.

- **Target variable:** `quantity_sold` (servings per dish per day)
- **Dataset:** 509 rows of March & April 2026 lunch-service sales records
- **Input features:** day of week, weather condition, public holiday flag, month, day of month, week of month, dish name (30 features after one-hot encoding)

## Features

- 🔮 Single-dish prediction — pick a dish, day, and weather, get a servings forecast
- 📋 Full daily menu planning — one click predicts all 16 dishes plus a total servings count
- ⚡ REST API with interactive Swagger docs
- ✅ Input validation with clear error messages
- 📊 Model interpretation — feature importance, permutation importance, residual analysis, and more

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript |
| Backend API | FastAPI (Python) |
| ML | scikit-learn, pandas, numpy |
| Model persistence | joblib |

## Model Development

Seven regression algorithms were trained and compared: Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and K-Nearest Neighbors. Each was evaluated with a 70/30 train-test split, then validated with 10-fold cross-validation, then tuned via Grid Search CV on the top two performers.

| Model | Baseline R² | Optimized R² | Notes |
|---|---|---|---|
| **Gradient Boosting** | ~0.54 | ~0.60 | Best performer — highest accuracy |
| **Random Forest** | ~0.51 | ~0.58 | Most stable across CV folds (lowest std) |
| Decision Tree | ~0.38 | — | Overfits without tuning |
| KNN | ~0.34 | — | Sensitive to feature scaling |
| Ridge / Linear / Lasso | ~0.13–0.15 | — | Miss non-linear demand patterns |

The final model is selected automatically at training time — whichever of the two tuned ensemble models (Gradient Boosting or Random Forest) scores highest on the test set.

**Key finding:** dish identity is by far the strongest predictor of demand (confirmed by both built-in feature importance and permutation importance), with weather as a secondary factor.

Full methodology, all evaluation tables, and 5 model-interpretation techniques are documented in [`docs/ML_Documentation_Report.pdf`](docs/ML_Documentation_Report.pdf).

## System Architecture

```
Frontend (index.html)  →  FastAPI backend (app.py)  →  Trained model (best_model.pkl)
     vanilla JS               validation, feature            scaler.pkl
     fetch() calls            engineering, scaling            label_columns.pkl
```

1. User submits a prediction request through the form
2. FastAPI validates the dish/day/weather values
3. Input is one-hot encoded and aligned to the training column order
4. `StandardScaler` transforms the row using the scaler fitted during training
5. The model predicts a raw value, which is rounded to a whole number of servings

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the HTML frontend |
| GET | `/health` | API health check + model info |
| POST | `/predict` | Single dish prediction |
| POST | `/predict-batch` | Predict all 16 dishes at once |
| GET | `/dishes` | List valid dishes, days, and weather options |
| GET | `/model-info` | Model details and performance metrics |
| GET | `/docs` | Interactive Swagger API documentation |

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib fastapi uvicorn openpyxl
```

### Run it

```bash
# 1. (Optional) Retrain the model — best_model.pkl etc. are already included
python train_and_save.py

# 2. Start the server
uvicorn app:app --reload
```

Then open:
- **App:** http://127.0.0.1:8000
- **API docs:** http://127.0.0.1:8000/docs

> **Note:** `index.html` must be opened through the running server (the URL above), not double-clicked directly from disk — it calls the API using a relative path that only resolves when FastAPI is serving the page.

## Project Structure

```
├── app.py                      # FastAPI backend
├── index.html                  # Frontend UI
├── train_and_save.py           # Model training pipeline
├── best_model.pkl              # Trained model
├── scaler.pkl                  # Fitted StandardScaler
├── label_columns.pkl           # Feature column order
├── evaluation_results.csv      # All model scores (baseline, CV, optimized)
├── *.png                       # EDA and model interpretation charts
└── docs/
    └── ML_Documentation_Report.pdf
```

## Team

- De los Reyes, Carlo M. — Group Leader
- Enoria, Jorich Maricar C.
- Mercado, Vesti Sahm Angielo
- Santiago, Ishmael Luis C.

BSIT 3-2 | IPT 101 | Pamantasan ng Lungsod ng Valenzuela
