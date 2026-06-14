# 🛡️ Insurance Claims Fraud Detection

A binary classification system that predicts whether an insurance claim is
likely fraudulent, with a Streamlit dashboard for interactive predictions.

```
Dataset → Cleaning → Feature Engineering → ML Model → Fraud Score → Streamlit Dashboard
```

---

## 📁 Project Structure

```
fraud_project/
├── data/
│   └── insurance_claims.csv      # synthetic dataset (5,000 rows)
├── models/                        # created after training
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   ├── model_columns.pkl
│   ├── needs_scaling.pkl
│   └── model_name.pkl
├── plots/                          # created after training (EDA charts)
├── app/
│   └── app.py                      # Streamlit dashboard
├── generate_data.py                # creates the synthetic dataset
├── train_model.py                  # cleaning + feature engineering + training
├── requirements.txt
├── Procfile                         # for Railway/Heroku-style deployment
└── README.md
```

---

## 🚀 How to Run (in Antigravity / locally)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
This creates `data/insurance_claims.csv` with 5,000 rows (~24% fraud rate).
```bash
python generate_data.py
```

### 3. Train the models
This runs EDA, cleaning, feature engineering, trains Logistic Regression,
Random Forest, and XGBoost, evaluates them, and saves the best one to `models/`.
```bash
python train_model.py
```

You'll see printed metrics (Accuracy, Precision, Recall, F1, ROC AUC) for
all three models, plus the winning model. EDA plots are saved to `plots/`.

### 4. Run the dashboard
```bash
streamlit run app/app.py
```
This opens a browser window where you can enter claim details and get:
- **Fraud Risk Score** (e.g. 87%)
- **Classification** (High Risk / Low Risk)

---

## 📊 Dataset Schema

| Column              | Description                                  |
|---------------------|-----------------------------------------------|
| `claim_amount`      | Total amount claimed ($)                      |
| `age`               | Age of the policyholder                       |
| `policy_type`       | Basic / Premium / Comprehensive               |
| `hospital_charges`  | Hospital charges associated with the claim ($)|
| `annual_premium`    | Annual policy premium ($)                     |
| `claim_history`     | No_History / Minor / Major prior claim history|
| `previous_claims`   | Number of previous claims filed               |
| `fraud_reported`    | Target: 1 = Fraud, 0 = Not Fraud              |

> Note: This is a **synthetic** dataset generated with realistic fraud
> patterns (e.g. high claim-to-premium ratio, inflated hospital charges,
> many previous claims, major claim history). You can swap this file with
> a real Kaggle dataset as long as you update column names in
> `train_model.py` and `app/app.py` to match.

---

## 🔧 Feature Engineering

| New Feature              | Formula                                    |
|---------------------------|---------------------------------------------|
| `claim_to_premium_ratio`  | `claim_amount / (annual_premium + 1)`        |
| `charges_to_claim_ratio`  | `hospital_charges / (claim_amount + 1)`      |
| `high_previous_claims`    | `1 if previous_claims >= 3 else 0`           |

---

## 🤖 Models Trained

1. **Logistic Regression** (with feature scaling, `class_weight='balanced'`)
2. **Random Forest** (`class_weight='balanced'`)
3. **XGBoost** (`scale_pos_weight` tuned for class imbalance)

The script automatically picks the model with the **highest ROC AUC** and
saves it for the dashboard to use.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

---

## 🌐 Deployment

### Railway
1. Push this folder to a GitHub repo.
2. Create a new Railway project from the repo.
3. Railway will detect the `Procfile` and run:
   ```
   streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0
   ```

### Render
1. Push to GitHub.
2. Create a new **Web Service** on Render.
3. Build command: `pip install -r requirements.txt`
4. Start command:
   ```
   streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0
   ```

---

## ⚠️ Important Notes

- **Re-run `train_model.py` after any change to `generate_data.py`** — the
  saved model, scaler, and column list must match what `app.py` expects.
- If you swap in a real Kaggle dataset, run:
  ```python
  print(df.columns.tolist())
  print(df.dtypes)
  ```
  first, and update the column names in `train_model.py` and `app/app.py`
  accordingly. The one-hot encoded column names depend on the exact
  category strings (e.g. `policy_type_Premium`), so always check
  `models/model_columns.pkl` after training:
  ```python
  import joblib
  print(joblib.load("models/model_columns.pkl"))
  ```

---

## 📈 Sample Output

```
Fraud Risk Score = 87%
Classification = High Risk
```
