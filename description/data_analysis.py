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

# %%

