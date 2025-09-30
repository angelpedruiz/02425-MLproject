# Data exploration of S&P 500 financial metrics

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

### DATA CLEANING ###

# %% Load the dataset
data = pd.read_csv("../data/financials.csv")
print(data.head())



# %% Display basic information about the dataset
print(data.info())

# %% Summary statistics
print(data.describe())



# %% Missing values analysis
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)


# %% Add atribute for if the company gives dividends
df_1 = data.copy()
df_1['Gives_Dividends'] = np.where(df_1['Dividend Yield'] > 0, 'Yes', 'No')
print(df_1[['Dividend Yield', 'Gives_Dividends']].head())

# %% Data visualization for dividend distribution
sns.countplot(x='Gives_Dividends', data=df_1)
plt.title('Distribution of Companies Giving Dividends')
plt.show()
print(df_1['Gives_Dividends'].value_counts())



# %% For missing value attddributes, check for 0 values
zero_values = (data == 0).sum()
print("Zero values in each column:\n", zero_values)


# %% Handle missing values by dropping rows with missing numeric values
df_2 = df_1.copy()

# Select numeric columns
numeric_cols = df_2.select_dtypes(include='number').columns

# Remove rows with missing values in numeric columns
df_clean = df_2.dropna(subset=numeric_cols)

# Optional: check that missing values are gone
print(df_clean[numeric_cols].isnull().sum())



# %% Analyse new dataframe
print(df_2.describe())


# %% Swap misslabelled columns
df_3 = df_2.copy()
bad_rows = df_3[df_3['52 Week Low'] > df_3['52 Week High']]
df_3.loc[df_3['52 Week Low'] > df_3['52 Week High'], ['52 Week Low', '52 Week High']] = \
df_3.loc[df_3['52 Week Low'] > df_3['52 Week High'], ['52 Week High', '52 Week Low']].values

# Check if the swap was successful for all rows (there should be no rows where 52 Week Low > 52 Week High)
bad_rows_after = df_3[df_3['52 Week Low'] > df_3['52 Week High']]
print("Number of rows with 52 Week Low > 52 Week High after swap:", len(bad_rows_after))

# %% Save cleaned data
df_3.to_csv("../data/financials_cleaned.csv", index=False)
df_cleaned = pd.read_csv("../data/financials_cleaned.csv")

# %% Check cleaned data info
print(df_cleaned.info())
# check for missing values
print(df_cleaned.isnull().sum())

# Drop rows with missing values in numeric columns

# Replace empty strings with NaN
df_2.replace('', np.nan, inplace=True)

# Now drop rows with missing numeric values
numeric_cols = df_2.select_dtypes(include='number').columns
df_clean = df_2.dropna(subset=numeric_cols)

# Check
print(df_clean[numeric_cols].isnull().sum())

# %% Save cleaned data
df_clean.to_csv("../data/financials_cleaned.csv", index=False)
df_cleaned = pd.read_csv("../data/financials_cleaned.csv")



### SUMMARY STATISTICS ###

# %% Kurtosis and Skewness for continuous variables

continuous_vars = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
kurtosis = df_cleaned[continuous_vars].kurtosis()
skewness = df_cleaned[continuous_vars].skew()
print("Kurtosis:\n", kurtosis)
print("Skewness:\n", skewness)


# %% Outliers detection using IQR method

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

# %% Optional: visualize a few distributions with boxplots
for col in ["Price", "Price/Earnings", "Price/Book", "Market Cap", "EBITDA"]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_cleaned[col])
    plt.title(f"Boxplot of {col} (with outliers)")
    plt.show()
# %% Log-transform skewed variables (log1p handles zeros safely)
log_transformed = df_cleaned.copy()

skewed_cols = ["Price", "Price/Earnings", "Price/Book", "Market Cap", "EBITDA", "Price/Sales"]

for col in skewed_cols:
    # Only apply if column has positive values
    # log1p = log(1 + x), avoids issues with 0 values
    log_transformed[f"log_{col}"] = np.log1p(log_transformed[col].clip(lower=0))

# Check new skewness after log transform
log_vars = [f"log_{col}" for col in skewed_cols]
print("Skewness after log-transform:\n", log_transformed[log_vars].skew())
print("Kurtosis after log-transform:\n", log_transformed[log_vars].kurtosis())


# %% Outlier detection on log-transformed data
outlier_summary_log = {}

for col in log_vars:
    Q1 = log_transformed[col].quantile(0.25)
    Q3 = log_transformed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = log_transformed[(log_transformed[col] < lower_fence) | (log_transformed[col] > upper_fence)]

    outlier_summary_log[col] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence,
        "Num Outliers": len(outliers),
        "Pct Outliers": round(100 * len(outliers) / len(log_transformed), 2)
    }

outlier_df_log = pd.DataFrame(outlier_summary_log).T
print("\nOutlier Summary on Log-transformed data (IQR method):\n", outlier_df_log)


# %% Compare distributions before vs after log-transform

cols_to_compare = ["Price", "Price/Earnings", "Price/Book", "Market Cap", "EBITDA", "Price/Sales"]

for col in cols_to_compare:
    plt.figure(figsize=(12, 4))

    # Raw distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df_cleaned[col], bins=50, kde=True)
    plt.title(f"Raw {col} Distribution")

    # Log distribution
    plt.subplot(1, 2, 2)
    sns.histplot(log_transformed[f"log_{col}"], bins=50, kde=True, color="orange")
    plt.title(f"Log-transformed {col} Distribution")

    plt.tight_layout()
    plt.show()



# %% Save the log-transformed dataset
log_transformed.to_csv("../data/financials_log_transformed.csv", index=False)

# Optional: reload to check
df_transformed = pd.read_csv("../data/financials_log_transformed.csv")
print(df_transformed.head())

# List of skewed columns you want to log-transform
skewed_cols = ["Price", "Price/Earnings", "Price/Book", "Market Cap", "EBITDA", "Price/Sales"]

# Make a copy for transformation
df_transformed = df_cleaned.copy()

# Apply log1p (handles zeros) and REPLACE the original columns
for col in skewed_cols:
    df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))

# Check skewness and kurtosis again
print("Skewness after replacement:\n", df_transformed[skewed_cols].skew())
print("Kurtosis after replacement:\n", df_transformed[skewed_cols].kurtosis())

# Save the cleaned + log-transformed dataset
df_transformed.to_csv("../data/financials_log_transformed.csv", index=False)
print("Saved dataset with log-transformed columns replacing originals.")


# %%
