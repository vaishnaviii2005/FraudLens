"""
Generates a synthetic insurance claims dataset (5000 rows)
matching the simple schema:
Claim Amount, Age, Policy Type, Hospital Charges, Claim History,
Number of Previous Claims -> Fraud (0/1)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

policy_types = ["Basic", "Premium", "Comprehensive"]
claim_history_types = ["No_History", "Minor", "Major"]

# Base features
age = np.random.randint(18, 75, N)
policy_type = np.random.choice(policy_types, N, p=[0.4, 0.35, 0.25])
annual_premium = np.where(
    policy_type == "Basic", np.random.normal(800, 100, N),
    np.where(policy_type == "Premium", np.random.normal(1500, 200, N),
             np.random.normal(2500, 300, N))
)
annual_premium = np.clip(annual_premium, 300, None)

previous_claims = np.random.poisson(1, N)
claim_history = np.random.choice(claim_history_types, N, p=[0.5, 0.35, 0.15])

hospital_charges = np.random.gamma(shape=2, scale=2000, size=N)
claim_amount = hospital_charges * np.random.uniform(0.8, 1.5, N) + np.random.normal(0, 500, N)
claim_amount = np.clip(claim_amount, 100, None)

# --- Fraud logic: create realistic patterns ---
fraud_score = (
    0.000015 * claim_amount
    + 0.00002 * hospital_charges
    + 0.35 * (previous_claims >= 3).astype(int)
    + 0.3 * (claim_history == "Major").astype(int)
    + 0.25 * (claim_amount / (annual_premium + 1) > 5).astype(int)
    + 0.2 * (hospital_charges / (claim_amount + 1) > 1.2).astype(int)  # inflated charges
    - 0.1 * (age > 50).astype(int)
    + np.random.normal(0, 0.15, N)  # noise
)

# Normalize to probability and sample binary outcome
fraud_prob = 1 / (1 + np.exp(-(fraud_score - 0.55) * 6))
fraud = np.random.binomial(1, fraud_prob)

df = pd.DataFrame({
    "claim_amount": np.round(claim_amount, 2),
    "age": age,
    "policy_type": policy_type,
    "hospital_charges": np.round(hospital_charges, 2),
    "annual_premium": np.round(annual_premium, 2),
    "claim_history": claim_history,
    "previous_claims": previous_claims,
    "fraud_reported": fraud
})

# introduce a few missing values to practice cleaning
for col in ["hospital_charges", "annual_premium"]:
    idx = np.random.choice(df.index, size=int(0.01 * N), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("data/insurance_claims.csv", index=False)
print("Dataset created: data/insurance_claims.csv")
print(df["fraud_reported"].value_counts(normalize=True))
print(df.head())
