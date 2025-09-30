# Data preparation of S&P 500 financial metrics: cleaning and transformation

import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*80)
print("DATA PREPARATION PIPELINE - COMPREHENSIVE REPORT")
print("="*80)

### DATA CLEANING ###

print("\n" + "="*80)
print("STEP 1: DATA LOADING & INITIAL EXPLORATION")
print("="*80)

# Load the dataset
data = pd.read_csv(os.path.join(script_dir, "..", "data", "financials.csv"))
print(f"\n✓ Loaded data from financials.csv")
print(f"  Shape: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"  Columns: {', '.join(data.columns.tolist())}")

# Data types
print(f"\n  Data types:")
for dtype_name, cols in data.columns.to_series().groupby(data.dtypes).groups.items():
    print(f"    - {dtype_name}: {len(cols)} columns")

# Missing values analysis
missing_values = data.isnull().sum()
total_missing = missing_values.sum()
print(f"\n  Missing values: {total_missing} total ({100*total_missing/(data.shape[0]*data.shape[1]):.2f}% of all values)")
if total_missing > 0:
    print("\n  Columns with missing values:")
    for col, count in missing_values[missing_values > 0].items():
        pct = 100 * count / len(data)
        print(f"    - {col}: {count} ({pct:.1f}%)")

# Summary statistics for numeric columns
numeric_cols_initial = data.select_dtypes(include=['float64', 'int64']).columns
print(f"\n  Initial summary statistics (numeric columns):")
print(f"    Number of numeric columns: {len(numeric_cols_initial)}")
summary_stats = data[numeric_cols_initial].describe()
for col in numeric_cols_initial:
    if col in summary_stats.columns:
        print(f"\n    {col}:")
        print(f"      Mean: {summary_stats[col]['mean']:.2f}")
        print(f"      Std: {summary_stats[col]['std']:.2f}")
        print(f"      Min: {summary_stats[col]['min']:.2f}")
        print(f"      Max: {summary_stats[col]['max']:.2f}")
        print(f"      Q1/Median/Q3: {summary_stats[col]['25%']:.2f} / {summary_stats[col]['50%']:.2f} / {summary_stats[col]['75%']:.2f}")

print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

# Add attribute for if the company gives dividends
df_1 = data.copy()
df_1['Gives_Dividends'] = np.where(df_1['Dividend Yield'] > 0, 'Yes', 'No')
dividend_counts = df_1['Gives_Dividends'].value_counts()
print(f"\n✓ Added 'Gives_Dividends' feature")
print(f"  Yes: {dividend_counts.get('Yes', 0)} companies ({100*dividend_counts.get('Yes', 0)/len(df_1):.1f}%)")
print(f"  No: {dividend_counts.get('No', 0)} companies ({100*dividend_counts.get('No', 0)/len(df_1):.1f}%)")

print("\n" + "="*80)
print("STEP 3: DATA CLEANING")
print("="*80)

# Swap mislabeled columns (52 Week Low and High are reversed)
df_2 = df_1.copy()
bad_rows = df_2[df_2['52 Week Low'] > df_2['52 Week High']]
print(f"\n✓ Fixing swapped '52 Week Low' and '52 Week High' columns")
print(f"  Rows with reversed values: {len(bad_rows)} ({100*len(bad_rows)/len(df_2):.1f}%)")

df_2.loc[df_2['52 Week Low'] > df_2['52 Week High'], ['52 Week Low', '52 Week High']] = \
    df_2.loc[df_2['52 Week Low'] > df_2['52 Week High'], ['52 Week High', '52 Week Low']].values

# Verify swap was successful
bad_rows_after = df_2[df_2['52 Week Low'] > df_2['52 Week High']]
print(f"  Rows still reversed after fix: {len(bad_rows_after)}")

# Replace empty strings with NaN
df_2.replace('', np.nan, inplace=True)

# Drop rows with missing numeric values
numeric_cols = df_2.select_dtypes(include='number').columns
rows_before = len(df_2)
df_cleaned = df_2.dropna(subset=numeric_cols)
rows_removed = rows_before - len(df_cleaned)

print(f"\n✓ Removed rows with missing numeric values")
print(f"  Rows removed: {rows_removed} ({100*rows_removed/rows_before:.1f}%)")
print(f"  Final cleaned dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

# Save cleaned data
df_cleaned.to_csv(os.path.join(script_dir, "..", "data", "financials_cleaned.csv"), index=False)
print(f"\n✓ Saved cleaned data to: financials_cleaned.csv")

print("\n" + "="*80)
print("STEP 4: DATA ANALYSIS (PRE-TRANSFORMATION)")
print("="*80)

# Identify continuous variables
continuous_vars = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"\n✓ Identified {len(continuous_vars)} continuous variables")
print(f"  Variables: {', '.join(continuous_vars)}")

# Comprehensive summary statistics
print(f"\n  Summary statistics (cleaned data):")
summary_cleaned = df_cleaned[continuous_vars].describe()
for col in continuous_vars:
    print(f"\n  {col}:")
    print(f"    Count: {int(summary_cleaned[col]['count'])}")
    print(f"    Mean: {summary_cleaned[col]['mean']:.2f}")
    print(f"    Std: {summary_cleaned[col]['std']:.2f}")
    print(f"    Min: {summary_cleaned[col]['min']:.2f}")
    print(f"    Max: {summary_cleaned[col]['max']:.2f}")
    print(f"    Range: {summary_cleaned[col]['max'] - summary_cleaned[col]['min']:.2f}")
    print(f"    Q1/Median/Q3: {summary_cleaned[col]['25%']:.2f} / {summary_cleaned[col]['50%']:.2f} / {summary_cleaned[col]['75%']:.2f}")
    print(f"    IQR: {summary_cleaned[col]['75%'] - summary_cleaned[col]['25%']:.2f}")

    # Coefficient of variation
    cv = (summary_cleaned[col]['std'] / summary_cleaned[col]['mean']) * 100 if summary_cleaned[col]['mean'] != 0 else 0
    print(f"    Coefficient of Variation: {cv:.2f}%")

# Analyze skewness
skewness = df_cleaned[continuous_vars].skew()
print(f"\n  Skewness analysis:")
print(f"    All features:")
for col in continuous_vars:
    skew_val = skewness[col]
    interpretation = "highly right-skewed" if skew_val > 1 else "slightly right-skewed" if skew_val > 0.5 else "symmetric" if abs(skew_val) <= 0.5 else "slightly left-skewed" if skew_val > -1 else "highly left-skewed"
    print(f"      {col}: {skew_val:.2f} ({interpretation})")

highly_skewed = skewness[abs(skewness) > 1].sort_values(ascending=False)
print(f"\n  Highly skewed features (|skew| > 1): {len(highly_skewed)} out of {len(continuous_vars)}")
if len(highly_skewed) > 0:
    print(f"    Features requiring transformation:")
    for col, skew_val in highly_skewed.items():
        print(f"      - {col}: {skew_val:.2f}")

# Analyze kurtosis
kurtosis = df_cleaned[continuous_vars].kurtosis()
print(f"\n  Kurtosis analysis (excess kurtosis):")
for col in continuous_vars:
    kurt_val = kurtosis[col]
    interpretation = "heavy-tailed" if kurt_val > 3 else "light-tailed" if kurt_val < -1 else "normal-like"
    print(f"    {col}: {kurt_val:.2f} ({interpretation})")

# Count negative values in each column
print(f"\n  Negative values analysis:")
has_negatives = False
for col in continuous_vars:
    neg_count = (df_cleaned[col] < 0).sum()
    if neg_count > 0:
        has_negatives = True
        pct = 100 * neg_count / len(df_cleaned)
        print(f"    - {col}: {neg_count} ({pct:.1f}%)")
if not has_negatives:
    print(f"    No negative values found in any column")

# Outlier detection using IQR method
print(f"\n  Outlier detection (IQR method):")
for col in continuous_vars:
    Q1 = summary_cleaned[col]['25%']
    Q3 = summary_cleaned[col]['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
    if outliers > 0:
        pct = 100 * outliers / len(df_cleaned)
        print(f"    {col}: {outliers} outliers ({pct:.1f}%)")
        print(f"      Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

print("\n" + "="*80)
print("STEP 5: SIGNED LOG TRANSFORMATION")
print("="*80)

# Apply signed log transformation to ALL continuous variables
# Formula: sign(x) * log(1 + |x|)
# This preserves the sign of negative values while applying log transformation

df_transformed = df_cleaned.copy()

print(f"\n✓ Applying signed log transformation to all {len(continuous_vars)} continuous variables")
print(f"  Formula: sign(x) * log(1 + |x|)")

for col in continuous_vars:
    df_transformed[col] = np.sign(df_transformed[col]) * np.log1p(np.abs(df_transformed[col]))

print(f"\n  Transformation complete!")

print("\n" + "="*80)
print("STEP 6: POST-TRANSFORMATION ANALYSIS")
print("="*80)

# Comprehensive summary statistics after transformation
print(f"\n  Summary statistics (transformed data):")
summary_transformed = df_transformed[continuous_vars].describe()
for col in continuous_vars:
    print(f"\n  {col}:")
    print(f"    Mean: {summary_transformed[col]['mean']:.2f}")
    print(f"    Std: {summary_transformed[col]['std']:.2f}")
    print(f"    Min: {summary_transformed[col]['min']:.2f}")
    print(f"    Max: {summary_transformed[col]['max']:.2f}")
    print(f"    Range: {summary_transformed[col]['max'] - summary_transformed[col]['min']:.2f}")
    print(f"    Q1/Median/Q3: {summary_transformed[col]['25%']:.2f} / {summary_transformed[col]['50%']:.2f} / {summary_transformed[col]['75%']:.2f}")

# Check skewness after transformation
skewness_after = df_transformed[continuous_vars].skew()
kurtosis_after = df_transformed[continuous_vars].kurtosis()
highly_skewed_after = skewness_after[abs(skewness_after) > 1].sort_values(ascending=False)

print(f"\n✓ Skewness reduction summary:")
print(f"  Before transformation: {len(highly_skewed)} features with |skew| > 1")
print(f"  After transformation: {len(highly_skewed_after)} features with |skew| > 1")
print(f"  Improvement: {len(highly_skewed) - len(highly_skewed_after)} features normalized")

# Detailed skewness comparison
print(f"\n  Detailed skewness comparison (Before → After):")
for col in continuous_vars:
    before = skewness[col]
    after = skewness_after[col]
    improvement = abs(before) - abs(after)
    status = "✓ Improved" if improvement > 0 else "✗ Worsened" if improvement < 0 else "= Unchanged"
    print(f"    {col}: {before:.2f} → {after:.2f} (Δ {improvement:+.2f}) {status}")

if len(highly_skewed_after) > 0:
    print(f"\n  Remaining highly skewed features after transformation:")
    for col, skew_val in highly_skewed_after.items():
        original_skew = skewness[col]
        print(f"    - {col}: {original_skew:.2f} → {skew_val:.2f}")

# Kurtosis comparison
print(f"\n  Kurtosis comparison (Before → After):")
for col in continuous_vars:
    before = kurtosis[col]
    after = kurtosis_after[col]
    improvement = abs(before) - abs(after)
    print(f"    {col}: {before:.2f} → {after:.2f} (Δ {improvement:+.2f})")

# Check normality improvement
print(f"\n  Distribution normalization effectiveness:")
print(f"    Features with improved skewness: {sum(abs(skewness_after[col]) < abs(skewness[col]) for col in continuous_vars)}/{len(continuous_vars)}")
print(f"    Features with improved kurtosis: {sum(abs(kurtosis_after[col]) < abs(kurtosis[col]) for col in continuous_vars)}/{len(continuous_vars)}")
print(f"    Features now symmetric (|skew| < 0.5): {sum(abs(skewness_after) < 0.5)}/{len(continuous_vars)}")

# Outlier comparison after transformation
print(f"\n  Outlier analysis (post-transformation):")
for col in continuous_vars:
    Q1 = summary_transformed[col]['25%']
    Q3 = summary_transformed[col]['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_after = ((df_transformed[col] < lower_bound) | (df_transformed[col] > upper_bound)).sum()

    # Compare with before
    Q1_before = summary_cleaned[col]['25%']
    Q3_before = summary_cleaned[col]['75%']
    IQR_before = Q3_before - Q1_before
    lower_bound_before = Q1_before - 1.5 * IQR_before
    upper_bound_before = Q3_before + 1.5 * IQR_before
    outliers_before = ((df_cleaned[col] < lower_bound_before) | (df_cleaned[col] > upper_bound_before)).sum()

    if outliers_after > 0 or outliers_before > 0:
        pct_after = 100 * outliers_after / len(df_transformed)
        pct_before = 100 * outliers_before / len(df_cleaned)
        change = outliers_before - outliers_after
        print(f"    {col}: {outliers_before} ({pct_before:.1f}%) → {outliers_after} ({pct_after:.1f}%) (Δ {change:+d})")

print("\n" + "="*80)
print("STEP 7: CORRELATION ANALYSIS")
print("="*80)

# Correlation analysis before and after transformation
print(f"\n  Correlation matrix changes:")
corr_before = df_cleaned[continuous_vars].corr()
corr_after = df_transformed[continuous_vars].corr()

# Find strongest correlations
print(f"\n  Top 10 strongest correlations (transformed data):")
corr_pairs = []
for i in range(len(continuous_vars)):
    for j in range(i+1, len(continuous_vars)):
        corr_pairs.append((continuous_vars[i], continuous_vars[j], abs(corr_after.iloc[i,j]), corr_after.iloc[i,j]))
corr_pairs.sort(key=lambda x: x[2], reverse=True)

for var1, var2, abs_corr, corr in corr_pairs[:10]:
    print(f"    {var1} ↔ {var2}: {corr:.3f}")

# Compare correlation changes
print(f"\n  Largest correlation changes (|Δ| > 0.1):")
for i in range(len(continuous_vars)):
    for j in range(i+1, len(continuous_vars)):
        diff = abs(corr_after.iloc[i,j] - corr_before.iloc[i,j])
        if diff > 0.1:
            print(f"    {continuous_vars[i]} ↔ {continuous_vars[j]}: {corr_before.iloc[i,j]:.3f} → {corr_after.iloc[i,j]:.3f} (Δ {diff:.3f})")

print("\n" + "="*80)
print("STEP 8: SAVING RESULTS")
print("="*80)

# Save the cleaned + log-transformed dataset
output_path = os.path.join(script_dir, "..", "data", "financials_log_transformed.csv")
df_transformed.to_csv(output_path, index=False)
print(f"\n✓ Saved transformed data to: financials_log_transformed.csv")
print(f"  Shape: {df_transformed.shape[0]} rows, {df_transformed.shape[1]} columns")
print(f"  All continuous variables transformed using: sign(x) * log(1 + |x|)")

print("\n" + "="*80)
print("EXECUTIVE SUMMARY - DATA PREPARATION REPORT")
print("="*80)

print(f"\n1. DATA OVERVIEW:")
print(f"   - Original dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"   - Final dataset: {df_transformed.shape[0]} rows, {df_transformed.shape[1]} columns")
print(f"   - Rows removed: {data.shape[0] - df_transformed.shape[0]} ({100*(data.shape[0] - df_transformed.shape[0])/data.shape[0]:.1f}%)")
print(f"   - Continuous features: {len(continuous_vars)}")

print(f"\n2. DATA QUALITY ISSUES ADDRESSED:")
print(f"   - Missing values: {total_missing} cells ({100*total_missing/(data.shape[0]*data.shape[1]):.2f}% of dataset)")
print(f"   - Swapped columns: 52 Week Low/High reversed in {len(bad_rows)} rows ({100*len(bad_rows)/len(df_2):.1f}%)")
print(f"   - Rows dropped due to missing values: {rows_removed} ({100*rows_removed/rows_before:.1f}%)")

print(f"\n3. FEATURE ENGINEERING:")
print(f"   - Created 'Gives_Dividends' binary feature")
print(f"   - Companies with dividends: {dividend_counts.get('Yes', 0)} ({100*dividend_counts.get('Yes', 0)/len(df_1):.1f}%)")
print(f"   - Companies without dividends: {dividend_counts.get('No', 0)} ({100*dividend_counts.get('No', 0)/len(df_1):.1f}%)")

print(f"\n4. DISTRIBUTION CHARACTERISTICS (PRE-TRANSFORMATION):")
print(f"   - Highly skewed features (|skew| > 1): {len(highly_skewed)}/{len(continuous_vars)}")
print(f"   - Features with negative values: {sum((df_cleaned[col] < 0).sum() > 0 for col in continuous_vars)}")
total_outliers_before = sum((((df_cleaned[col] < (df_cleaned[col].quantile(0.25) - 1.5*(df_cleaned[col].quantile(0.75)-df_cleaned[col].quantile(0.25)))) | (df_cleaned[col] > (df_cleaned[col].quantile(0.75) + 1.5*(df_cleaned[col].quantile(0.75)-df_cleaned[col].quantile(0.25))))).sum()) for col in continuous_vars)
print(f"   - Total outliers detected: {total_outliers_before}")

print(f"\n5. TRANSFORMATION APPLIED:")
print(f"   - Method: Signed Log Transformation")
print(f"   - Formula: sign(x) * log(1 + |x|)")
print(f"   - Applied to: All {len(continuous_vars)} continuous variables")
print(f"   - Rationale: Handles negative values while reducing skewness")

print(f"\n6. TRANSFORMATION EFFECTIVENESS:")
print(f"   - Features normalized (|skew| reduced below 1): {len(highly_skewed) - len(highly_skewed_after)}")
print(f"   - Remaining highly skewed: {len(highly_skewed_after)}/{len(continuous_vars)}")
print(f"   - Features with improved skewness: {sum(abs(skewness_after[col]) < abs(skewness[col]) for col in continuous_vars)}/{len(continuous_vars)}")
print(f"   - Features now symmetric (|skew| < 0.5): {sum(abs(skewness_after) < 0.5)}/{len(continuous_vars)}")
total_outliers_after = sum((((df_transformed[col] < (df_transformed[col].quantile(0.25) - 1.5*(df_transformed[col].quantile(0.75)-df_transformed[col].quantile(0.25)))) | (df_transformed[col] > (df_transformed[col].quantile(0.75) + 1.5*(df_transformed[col].quantile(0.75)-df_transformed[col].quantile(0.25))))).sum()) for col in continuous_vars)
print(f"   - Outlier reduction: {total_outliers_before} → {total_outliers_after} ({total_outliers_before - total_outliers_after} fewer outliers)")

print(f"\n7. OUTPUT FILES GENERATED:")
print(f"   - financials_cleaned.csv (cleaned, untransformed)")
print(f"   - financials_log_transformed.csv (cleaned + sign log transformed)")

print(f"\n8. RECOMMENDATIONS FOR ANALYSIS:")
print(f"   - Use financials_log_transformed.csv for modeling")
print(f"   - All continuous variables have been normalized")
print(f"   - Consider {len(highly_skewed_after)} remaining skewed features for further processing if needed")
print(f"   - Strong correlations identified - consider multicollinearity in linear models")

print("\n" + "="*80)
print("DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)