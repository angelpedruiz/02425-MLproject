# Data preparation of S&P 500 financial metrics: cleaning and transformation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

### DATA CLEANING ###

# Load the dataset
data = pd.read_csv(os.path.join(script_dir, "..", "data", "financials.csv"))
print("Original data shape:", data.shape)
print(data.head())

# Display basic information about the dataset
print("\nData Info:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Missing values analysis
missing_values = data.isnull().sum()
print("\nMissing values in each column:\n", missing_values)

# Add attribute for if the company gives dividends
df_1 = data.copy()
df_1['Gives_Dividends'] = np.where(df_1['Dividend Yield'] > 0, 'Yes', 'No')
print("\nDividend distribution:")
print(df_1['Gives_Dividends'].value_counts())

# Check for zero values
zero_values = (data == 0).sum()
print("\nZero values in each column:\n", zero_values)

# Swap mislabeled columns (52 Week Low and High are reversed)
df_2 = df_1.copy()
bad_rows = df_2[df_2['52 Week Low'] > df_2['52 Week High']]
print(f"\nRows with swapped 52 Week Low/High: {len(bad_rows)}")

df_2.loc[df_2['52 Week Low'] > df_2['52 Week High'], ['52 Week Low', '52 Week High']] = \
    df_2.loc[df_2['52 Week Low'] > df_2['52 Week High'], ['52 Week High', '52 Week Low']].values

# Verify swap was successful
bad_rows_after = df_2[df_2['52 Week Low'] > df_2['52 Week High']]
print(f"Rows with 52 Week Low > 52 Week High after swap: {len(bad_rows_after)}")

# Replace empty strings with NaN
df_2.replace('', np.nan, inplace=True)

# Drop rows with missing numeric values
numeric_cols = df_2.select_dtypes(include='number').columns
df_cleaned = df_2.dropna(subset=numeric_cols)

print(f"\nData shape after cleaning: {df_cleaned.shape}")
print("Missing values after cleaning:", df_cleaned[numeric_cols].isnull().sum().sum())

# Save cleaned data
df_cleaned.to_csv(os.path.join(script_dir, "..", "data", "financials_cleaned.csv"), index=False)
print("\nSaved cleaned data to financials_cleaned.csv")

### SUMMARY STATISTICS ###

# Kurtosis and Skewness for continuous variables
continuous_vars = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
kurtosis = df_cleaned[continuous_vars].kurtosis()
skewness = df_cleaned[continuous_vars].skew()
print("\nKurtosis:\n", kurtosis)
print("\nSkewness:\n", skewness)

# Outliers detection using IQR method
outlier_summary = {}

for col in continuous_vars:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df_cleaned[(df_cleaned[col] < lower_fence) | (df_cleaned[col] > upper_fence)]

    outlier_summary[col] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence,
        "Num Outliers": len(outliers),
        "Pct Outliers": round(100 * len(outliers) / len(df_cleaned), 2)
    }

# Convert to DataFrame for a nice view
outlier_df = pd.DataFrame(outlier_summary).T
print("\nOutlier Summary (IQR method):\n", outlier_df)

### LOG TRANSFORMATION ###

# List of skewed columns to log-transform
skewed_cols = ["Price", "Price/Earnings", "Price/Book", "Market Cap", "EBITDA", "Price/Sales"]

# Make a copy for transformation
df_transformed = df_cleaned.copy()

# Apply signed log transformation: sign(x) * log(1 + |x|)
# This preserves the sign of negative values while applying log transformation
for col in skewed_cols:
    df_transformed[col] = np.sign(df_transformed[col]) * np.log1p(np.abs(df_transformed[col]))

# Check skewness and kurtosis after transformation
print("\nSkewness after log transformation:\n", df_transformed[skewed_cols].skew())
print("\nKurtosis after log transformation:\n", df_transformed[skewed_cols].kurtosis())

# Outlier detection on log-transformed data
outlier_summary_log = {}

for col in skewed_cols:
    Q1 = df_transformed[col].quantile(0.25)
    Q3 = df_transformed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = df_transformed[(df_transformed[col] < lower_fence) | (df_transformed[col] > upper_fence)]

    outlier_summary_log[col] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence,
        "Num Outliers": len(outliers),
        "Pct Outliers": round(100 * len(outliers) / len(df_transformed), 2)
    }

outlier_df_log = pd.DataFrame(outlier_summary_log).T
print("\nOutlier Summary on Log-transformed data (IQR method):\n", outlier_df_log)

# Save the cleaned + log-transformed dataset
df_transformed.to_csv(os.path.join(script_dir, "..", "data", "financials_log_transformed.csv"), index=False)
print("\nSaved log-transformed data to financials_log_transformed.csv")

print(f"\nFinal transformed data shape: {df_transformed.shape}")
print(df_transformed.head())