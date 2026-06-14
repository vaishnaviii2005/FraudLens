"""
Phase 3-6: Data Analysis, Cleaning, Feature Engineering, Model Training
Run this script to train and save the fraud detection model.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to file instead of displaying
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ----------------------------------------------------------------
# Phase 3: Load & Explore
# ----------------------------------------------------------------
df = pd.read_csv("data/insurance_claims.csv")

print("=" * 50)
print("DATA INFO")
print("=" * 50)
print(df.info())
print("\n" + "=" * 50)
print("DESCRIBE")
print("=" * 50)
print(df.describe())
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# Visualizations (saved to plots/ folder)
plt.figure(figsize=(8, 5))
df["claim_amount"].hist(bins=50)
plt.title("Claim Amount Distribution")
plt.xlabel("Claim Amount")
plt.savefig("plots/claim_amount_distribution.png")
plt.close()

plt.figure(figsize=(6, 4))
df["fraud_reported"].value_counts().plot(kind="bar")
plt.title("Fraud vs Non-Fraud")
plt.xticks([0, 1], ["Not Fraud", "Fraud"], rotation=0)
plt.savefig("plots/fraud_vs_nonfraud.png")
plt.close()

plt.figure(figsize=(8, 5))
df["age"].hist(bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.savefig("plots/age_distribution.png")
plt.close()

print("\nPlots saved to plots/ folder")

# ----------------------------------------------------------------
# Phase 4: Data Cleaning
# ----------------------------------------------------------------
# Fill missing numeric values with median
for col in ["hospital_charges", "annual_premium"]:
    df[col] = df[col].fillna(df[col].median())

df["previous_claims"] = df["previous_claims"].fillna(0)

# Encode categorical columns
df = pd.get_dummies(df, columns=["policy_type", "claim_history"], drop_first=True)

# ----------------------------------------------------------------
# Phase 5: Feature Engineering
# ----------------------------------------------------------------
df["claim_to_premium_ratio"] = df["claim_amount"] / (df["annual_premium"] + 1)
df["charges_to_claim_ratio"] = df["hospital_charges"] / (df["claim_amount"] + 1)
df["high_previous_claims"] = (df["previous_claims"] >= 3).astype(int)

# ----------------------------------------------------------------
# Phase 6: Model Training & Evaluation
# ----------------------------------------------------------------
X = df.drop(columns=["fraud_reported"])
y = df["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve

def evaluate(model, X_test_eval, y_test_eval, name):
    preds = model.predict(X_test_eval)
    probs = model.predict_proba(X_test_eval)[:, 1]
    acc = accuracy_score(y_test_eval, preds)
    prec = precision_score(y_test_eval, preds)
    rec = recall_score(y_test_eval, preds)
    f1 = f1_score(y_test_eval, preds)
    auc = roc_auc_score(y_test_eval, probs)
    print(f"\n--- {name} ---")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(auc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test_eval, preds))
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": confusion_matrix(y_test_eval, preds).tolist()
    }


print("\n" + "=" * 50)
print("MODEL TRAINING & EVALUATION (WITH HYPERPARAMETER TUNING)")
print("=" * 50)

# Model 1: Logistic Regression
print("\nTuning Logistic Regression...")
lr_grid = {
    "C": [0.01, 0.1, 1.0, 10.0]
}
lr_cv = GridSearchCV(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    lr_grid,
    cv=3,
    scoring="roc_auc"
)
lr_cv.fit(X_train_scaled, y_train)
lr = lr_cv.best_estimator_
metrics_lr = evaluate(lr, X_test_scaled, y_test, f"Logistic Regression (Best C={lr_cv.best_params_['C']})")

# Model 2: Random Forest
print("\nTuning Random Forest...")
rf_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10]
}
rf_cv = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    rf_grid,
    cv=3,
    scoring="roc_auc"
)
rf_cv.fit(X_train, y_train)
rf = rf_cv.best_estimator_
metrics_rf = evaluate(rf, X_test, y_test, f"Random Forest (Best Params={rf_cv.best_params_})")

# Model 3: XGBoost
print("\nTuning XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}
xgb_cv = GridSearchCV(
    XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=42),
    xgb_grid,
    cv=3,
    scoring="roc_auc"
)
xgb_cv.fit(X_train, y_train)
xgb = xgb_cv.best_estimator_
metrics_xgb = evaluate(xgb, X_test, y_test, f"XGBoost (Best Params={xgb_cv.best_params_})")

# ----------------------------------------------------------------
# Pick best model by ROC AUC and save everything needed for the app
# ----------------------------------------------------------------
results = {
    "Logistic Regression": (lr, metrics_lr["roc_auc"], True, metrics_lr),
    "Random Forest": (rf, metrics_rf["roc_auc"], False, metrics_rf),
    "XGBoost": (xgb, metrics_xgb["roc_auc"], False, metrics_xgb)
}

best_name = max(results, key=lambda k: results[k][1])
best_model, best_auc, needs_scaling, best_metrics = results[best_name]

print("\n" + "=" * 50)
print(f"BEST MODEL: {best_name} (ROC AUC = {round(best_auc, 4)})")
print("=" * 50)

joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/model_columns.pkl")
joblib.dump(needs_scaling, "models/needs_scaling.pkl")
joblib.dump(best_name, "models/model_name.pkl")

# Save comparison metrics of all models for frontend
all_metrics = {
    "Logistic Regression": metrics_lr,
    "Random Forest": metrics_rf,
    "XGBoost": metrics_xgb
}
joblib.dump(all_metrics, "models/model_metrics.pkl")

# Extract and save feature importances for all three models so frontend can display them
feature_importances = {}

# Logistic Regression importance (using coefficients)
lr_imp = np.abs(lr.coef_[0])
feature_importances["Logistic Regression"] = pd.Series(lr_imp, index=X.columns).to_dict()

# Random Forest importance
rf_imp = rf.feature_importances_
feature_importances["Random Forest"] = pd.Series(rf_imp, index=X.columns).to_dict()

# XGBoost importance
xgb_imp = xgb.feature_importances_
feature_importances["XGBoost"] = pd.Series(xgb_imp, index=X.columns).to_dict()

joblib.dump(feature_importances, "models/feature_importances.pkl")

# Plot feature importance for best model
if best_name == "Logistic Regression":
    importance = pd.Series(lr_imp, index=X.columns)
elif best_name == "Random Forest":
    importance = pd.Series(rf_imp, index=X.columns)
else:
    importance = pd.Series(xgb_imp, index=X.columns)

importance = importance.sort_values(ascending=False)
plt.figure(figsize=(8, 6))
importance.head(10).plot(kind="barh")
plt.title(f"Top Feature Importances - {best_name}")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# Generate ROC curve plot for frontend to display
plt.figure(figsize=(8, 6))
for name, (model_obj, _, scaled_status, _) in results.items():
    test_data = X_test_scaled if scaled_status else X_test
    probs = model_obj.predict_proba(test_data)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {round(results[name][1], 4)})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curves.png")
plt.close()

print("\nModel, scaler, and columns saved to models/ folder")
print("Done. You can now run the Streamlit app: streamlit run app/app.py")
