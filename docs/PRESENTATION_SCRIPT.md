# Presentation Script & Guide
## Maricopa Restaurant Dish Demand Forecasting
### IPT 101 — Task Performance #4 | BSIT 3-2

---

## PRE-PRESENTATION CHECKLIST

Before you start, make sure these are ready:

- [ ] `train_and_save.py` has been run — `best_model.pkl`, `scaler.pkl`, `label_columns.pkl` exist in the project folder
- [ ] FastAPI server is running: `uvicorn app:app --reload`
- [ ] Browser is open at `http://127.0.0.1:8000`
- [ ] Swagger docs tab open at `http://127.0.0.1:8000/docs`
- [ ] `evaluation_results.csv` is accessible for reference
- [ ] Code editor (VS Code) is open showing `train_and_save.py` and `app.py`

---

## PART 1 — INTRODUCTION (1–2 minutes)

**[Speaker: Group Leader — De los Reyes, Carlo M.]**

> "Good day, Sir Villones. We are Group [your group number], and we will be presenting our Task Performance #4 — the Development and Integration of a Machine Learning Model."

> "Our project is called **Maricopa Restaurant Dish Demand Forecasting**. The goal is to help the Maricopa Restaurant predict how many servings of each Filipino dish they will sell during lunch service on a given day — so they can minimize food waste and avoid shortages."

**Key points to state out loud:**
- Client problem: The restaurant doesn't know in advance how many servings to prepare.
- Our solution: A regression ML model that takes in the day of the week, weather, holiday status, and date — and predicts quantity sold per dish.
- Target variable: `quantity_sold` (number of servings)
- Dataset: 509 rows from March & April 2026 sales records
- 16 Filipino dishes, 7 days, 3 weather types

---

## PART 2 — ML WORKFLOW / MODEL DEVELOPMENT (5–7 minutes)

**[Open `train_and_save.py` in VS Code]**

### 2.1 — Data Preprocessing

> "Let me walk you through how we prepared the data before training."

Point to the relevant code sections as you explain each step:

| Step | What to say |
|------|-------------|
| **Load data** | "We loaded two Excel files — one for March 2026 and one for April 2026 — and combined them using `pd.concat()`, giving us 509 training rows." |
| **Date features** | "We converted the `date` column to datetime and extracted `month`, `day_of_month`, and `week_of_month`. This helps the model learn temporal patterns — for example, that week 1 of April might have different demand than week 4." |
| **Drop columns** | "We dropped `meal_period` because every single row was 'Lunch' — zero variance, no information. We also dropped the original `date` column after extracting features from it." |
| **One-hot encoding** | "We applied one-hot encoding to `dish_name` (16 dishes), `day_of_week` (7 days), and `weather` (3 types — Sunny, Cloudy, Rainy). This converts text categories into binary numbers that ML models can process. The final feature matrix X has **30 columns**." |
| **StandardScaler** | "We fitted the `StandardScaler` on `X_train` only, then used it to transform both `X_train` and `X_test` separately. This is critical — if we scaled using the test data, the model would 'peek' at test information during training, which is called **data leakage**." |
| **Train-test split** | "We split the data 70/30 — 356 rows for training, 153 rows for testing — using `random_state=42` for reproducibility." |

---

### 2.2 — Algorithm Selection

> "We selected 7 regression algorithms to compare, ranging from simple linear models to powerful ensemble methods."

| Model | Say this |
|-------|----------|
| **Linear Regression** | "Our baseline — assumes a straight-line relationship between features and servings." |
| **Ridge Regression** | "Adds L2 regularization — handles correlated one-hot features well." |
| **Lasso Regression** | "Adds L1 regularization — can zero out irrelevant dish or day columns entirely." |
| **Decision Tree** | "Non-linear model — learns if-else rules like 'if dish = Sinigang AND weather = Rainy, then more servings.'" |
| **Random Forest** | "Ensemble of 100 Decision Trees — averages predictions to reduce overfitting." |
| **Gradient Boosting** | "Sequentially builds trees, each correcting the previous one's errors — typically the highest accuracy." |
| **K-Nearest Neighbors** | "Instance-based — predicts by averaging the 5 most similar historical days." |

---

### 2.3 — Training Procedure

> "Each model was trained the same way for a fair comparison."

Point to the training loop in `train_and_save.py`:

> "We called `model.fit(X_train_scaled, y_train)` for each model, then `model.predict(X_test_scaled)` to generate predictions. We computed three metrics: **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error), and **R²** (coefficient of determination)."

---

## PART 3 — EVALUATION RESULTS (4–5 minutes)

**[Open `evaluation_results.csv` or refer to the printed documentation]**

### 3.1 — Baseline Results

> "After training all 7 models with default hyperparameters, here are the results sorted by R² score:"

Read/show the top 3:
- **Gradient Boosting**: Best performer — R² ≈ 0.54, MAE ≈ 1.8 servings
- **Random Forest**: Close second — R² ≈ 0.51, MAE ≈ 1.9 servings
- **Decision Tree**: R² ≈ 0.38 — overfits without tuning
- Linear models (Ridge, Linear, Lasso): R² around 0.13–0.15 — they miss the non-linear patterns in dish demand

---

### 3.2 — Cross-Validation

> "To check if the models are reliable and not just lucky on one test split, we applied **10-Fold Cross-Validation** across the full dataset."

Key point to emphasize:
> "Random Forest had the **lowest R² Standard Deviation (~0.07)**, meaning it is the most stable and consistent model across different data splits. A low standard deviation means it won't surprise you with wildly different results on real-world data."

---

### 3.3 — Hyperparameter Optimization (Grid Search)

> "We applied Grid Search with 5-Fold Cross-Validation to the two best models."

| Model | Baseline R² | Optimized R² | Improvement |
|-------|-------------|--------------|-------------|
| Random Forest | ~0.51 | ~0.58 | +0.07 |
| Gradient Boosting | ~0.54 | ~0.60 | +0.06 |

> "Grid Search tested 72 Random Forest configurations and 54 Gradient Boosting configurations. The best parameters were automatically selected based on highest mean CV R² score."

Best RF params example: `n_estimators=200, max_depth=10, min_samples_split=5, max_features='sqrt'`

Best GB params example: `n_estimators=200, learning_rate=0.1, max_depth=5`

---

### 3.4 — Final Model Selection

> "The final selected model is the **Optimized Gradient Boosting Regressor** — or Optimized Random Forest if it scores higher at runtime. The exact winner is determined when `train_and_save.py` is run."

Justify with 5 criteria:
1. **Highest R²** — explains the most variance in quantity_sold
2. **Lowest MAE** — average error is less than 2 servings per day, which is practical for a restaurant
3. **CV Stability** — consistent across all 10 folds
4. **Ensemble Method** — more robust than a single decision tree
5. **Restaurant Applicability** — MAE < 2 servings/day means the owner can confidently plan their daily prep

---

### 3.5 — Model Interpretation

> "We applied 5 interpretation techniques to understand WHY the model makes its predictions:"

| Technique | Key Finding |
|-----------|-------------|
| Feature Importance (Built-in) | Dish name features dominate — the specific dish is the strongest predictor of demand |
| Permutation Importance | Confirms dish identity as critical; weather shows moderate importance |
| Learning Curve | Model still improves at full data size — more months of data would further improve accuracy |
| Noise Sensitivity | Sinigang and soup dishes are most sensitive — record dish names accurately |
| Residual Analysis | Residuals are approximately normally distributed with mean near 0 — no systematic bias |

---

## PART 4 — INTEGRATION / SYSTEM DEVELOPMENT (4–5 minutes)

**[Switch to browser at `http://127.0.0.1:8000`]**

### 4.1 — System Architecture

> "The system follows a three-layer architecture:"

1. **Frontend** — `index.html` (vanilla HTML/CSS/JavaScript) — the user interface
2. **Backend API** — `app.py` (FastAPI, Python) — handles requests and runs predictions
3. **ML Layer** — `best_model.pkl`, `scaler.pkl`, `label_columns.pkl` — the serialized model, loaded once at server startup

---

### 4.2 — Live Demo: HTML Frontend

> "Let me show the system live."

**[On the browser at `http://127.0.0.1:8000`]**

Steps to demo:
1. Select a dish — e.g., *Sinigang na Salmon sa Miso*
2. Select day — e.g., *Friday*
3. Select weather — e.g., *Rainy*
4. Set is_holiday — e.g., *No*
5. Fill in month, day_of_month, week_of_month
6. Click **Predict**
7. Show the result — servings count and recommendation

> "The frontend collects these inputs and sends a POST request to `/predict` on our FastAPI server. The server one-hot encodes the input, scales it using the same scaler from training, runs `model.predict()`, and returns the predicted servings as a JSON response — which the frontend then displays."

---

### 4.3 — Data Flow (narrate while pointing at screen)

> "Here is what happens step by step when you click Predict:"

1. User fills the form and clicks Predict
2. JavaScript sends a `POST /predict` request with a JSON payload
3. FastAPI validates the input — returns HTTP 422 if dish/day/weather is invalid
4. `app.py` builds the one-hot encoded feature row (30 columns)
5. Aligns the row to the exact column order from `label_columns.pkl`
6. Applies `scaler.transform()` — same scaler used during training
7. Calls `model.predict()` → raw float (e.g., 21.4 servings)
8. Rounds to integer, minimum 1 serving
9. Returns JSON with prediction, recommendation, and model info
10. Frontend displays the result to the user

---

### 4.4 — Live Demo: API Endpoints

**[Switch to `http://127.0.0.1:8000/docs`]**

Show and briefly demo each endpoint:

| Endpoint | Demo action |
|----------|-------------|
| `GET /health` | Click → show JSON with `status: running` and model type |
| `GET /dishes` | Click → show the list of 16 valid dishes, 7 days, 3 weather types |
| `GET /model-info` | Click → show model name, R² score, MAE, feature count |
| `POST /predict` | Live test — fill in a dish, show prediction response |
| `POST /predict-batch` | Live test — send all 16 dishes, show all predictions + total servings |

> "The batch prediction endpoint is especially useful — the restaurant owner can get the full day's forecast for all 16 dishes at once."

**Invalid input test:**
> "If we send an invalid dish name — say, 'InvalidDish' — the API returns HTTP 422 with a clear validation error message. Input validation is handled at the API level."

---

### 4.5 — Model Files

> "Three files are generated by `train_and_save.py` and loaded by `app.py` at startup:"

| File | Purpose |
|------|---------|
| `best_model.pkl` | The trained model — used for `model.predict()` |
| `scaler.pkl` | Fitted StandardScaler — ensures test inputs are scaled identically to training |
| `label_columns.pkl` | List of 30 feature column names in training order — ensures column alignment |

> "These are serialized using `joblib` and loaded once when the server starts, so every prediction request is fast."

---

## PART 5 — DOCUMENTATION SUMMARY (1–2 minutes)

> "The documentation covers all deliverables of this laboratory activity:"

- **Section 1**: Problem statement, preprocessing pipeline, algorithm selection, and training procedure
- **Section 2**: Baseline results, cross-validation, hyperparameter optimization, final model selection, and 5 model interpretation techniques
- **Section 3**: System architecture, data flow diagram, API endpoint table, model file format, setup instructions, and system testing checklist

> "The `evaluation_results.csv` file contains all exact model scores — baseline, CV, and optimized — generated automatically when `train_and_save.py` is run."

---

## PART 6 — CLOSING (30 seconds)

> "To summarize: we built a complete end-to-end machine learning system — from data preprocessing and model training, through hyperparameter optimization, to a live web application with a REST API and a user-friendly frontend."

> "The final model predicts dish demand with an average error of under 2 servings per day, which is practical and actionable for the restaurant owner."

> "That concludes our presentation. We're open for questions."

---

## COMMON Q&A — EXPECTED QUESTIONS

**Q: Why did you use Gradient Boosting as the final model instead of Random Forest?**
> A: Gradient Boosting achieved a slightly higher R² score on the test set (≈0.60 vs ≈0.58 after tuning). However, the system automatically selects whichever scores higher at runtime — so if Random Forest wins on a given run, it becomes the final model.

**Q: Why StandardScaler? Why not MinMaxScaler?**
> A: StandardScaler is preferred for models sensitive to feature scale because it normalizes based on mean and standard deviation, making it more robust to outliers than MinMaxScaler. For tree-based models like RF and GB, scaling doesn't affect predictions — but it's applied consistently across all 7 models for a fair comparison.

**Q: What does R² = 0.60 actually mean?**
> A: It means the model explains 60% of the variance in the quantity_sold values. The remaining 40% is variance that the available features (day, weather, holiday, dish) cannot fully explain — for example, unpredictable events like a school field trip or a catering order.

**Q: Why only March and April data?**
> A: Those were the available sales records provided for this project. The learning curve analysis confirms the model would improve with more months of data.

**Q: How does the batch prediction endpoint work?**
> A: It accepts a list of dish input objects and runs `model.predict()` for each one, returning all 16 predictions plus a grand total of servings — giving the restaurant owner a complete daily forecast in one API call.

**Q: What happens if the model files don't exist when the server starts?**
> A: `app.py` raises a `FileNotFoundError` immediately at startup and prints a clear message telling you to run `train_and_save.py` first. The server will not start without the model files.

---

## RUNNING COMMANDS (quick reference)

```bash
# Step 1 — Train the model (run once)
python train_and_save.py

# Step 2 — Start the web server
uvicorn app:app --reload

# Access points
# Frontend:  http://127.0.0.1:8000
# API docs:  http://127.0.0.1:8000/docs
# Health:    http://127.0.0.1:8000/health
```

---

*Group Leader: De los Reyes, Carlo M. | Members: Enoria, Jorich Maricar C. · Mercado, Vesti Sahm Angielo · Santiago, Ishmael Luis C.*
*Section: BSIT 3-2 | IPT 101 | May 22, 2026*
