import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.ensemble import IsolationForest
import numpy as np

def load_data(filepath):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None

def generate_profile_report(df, title="Data Quality Report"):
    """Generates a ydata-profiling report."""
    profile = ProfileReport(df, title=title, minimal=True) # minimal for faster generation
    return profile

def detect_anomalies(df):
    """Detects anomalies in numerical columns using Isolation Forest and identifies common data quality issues."""
    anomalies = []

    # 1. Missing Values
    for col in df.columns:
        if df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            anomalies.append({
                "type": "Missing Values",
                "column": col,
                "description": f"{missing_count} ({missing_percentage:.2f}%) missing values found.",
                "sample_values": df[df[col].isnull()].index.tolist()[:3] # Get first 3 indices
            })

    # 2. Outliers (Numerical Columns) using Isolation Forest
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df[col].nunique() > 1: # Ensure there's variation
            # Handle NaNs for Isolation Forest
            temp_col_data = df[col].dropna().values.reshape(-1, 1)
            if len(temp_col_data) > 1: # Isolation Forest needs at least 2 samples
                try:
                    iso_forest = IsolationForest(contamination='auto', random_state=42)
                    iso_forest.fit(temp_col_data)
                    outlier_scores = iso_forest.decision_function(temp_col_data)
                    # Lower score indicates higher likelihood of anomaly
                    outlier_indices = np.where(outlier_scores < np.percentile(outlier_scores, 5))[0] # Top 5% most anomalous

                    if len(outlier_indices) > 0:
                        original_indices = df[col].dropna().iloc[outlier_indices].index.tolist()
                        anomalies.append({
                            "type": "Outlier (Numerical)",
                            "column": col,
                            "description": f"Potential outliers detected using Isolation Forest. Score below 5th percentile.",
                            "sample_values": df.loc[original_indices, col].head(3).tolist(),
                            "original_indices": original_indices[:3]
                        })
                except ValueError as e:
                    print(f"Could not run IsolationForest on {col}: {e}")
                    pass # Skip if column isn't suitable

    # 3. Invalid Email Format (Regex based - simple example)
    if 'email' in df.columns:
        invalid_emails = df[~df['email'].astype(str).str.contains(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=False)]
        if not invalid_emails.empty:
            anomalies.append({
                "type": "Invalid Format (Email)",
                "column": "email",
                "description": f"{len(invalid_emails)} rows with invalid email format found.",
                "sample_values": invalid_emails['email'].head(3).tolist(),
                "original_indices": invalid_emails.index.tolist()[:3]
            })

    # Add more data quality checks as needed (e.g., duplicates, inconsistent categories, out-of-range dates)

    return anomalies