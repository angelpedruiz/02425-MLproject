# Correlation analysis and distribution analysis of S&P 500 financial variables (log-transformed data)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "..", "data", "financials_log_transformed.csv"))
output_dir = os.path.join(script_dir, "..", "results", "statistical_analysis_transformed")
os.makedirs(output_dir, exist_ok=True)

# Define which columns were log-transformed
log_transformed_cols = ["Price", "Price/Earnings", "Market Cap", "EBITDA", "Price/Sales", "Price/Book"]

cols = ["Price", "Price/Earnings", "Dividend Yield", "Earnings/Share", "52 Week Low",
        "52 Week High", "Market Cap", "EBITDA", "Price/Sales", "Price/Book"]

# Histograms + Boxplots by attribute
for col in cols:
    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # Check if column was log-transformed
    label = f"log({col})" if col in log_transformed_cols else col
    title_suffix = " (log-transformed)" if col in log_transformed_cols else ""

    sns.histplot(data[col].dropna(), kde=False, bins=30, color='skyblue', ax=axes[0])
    axes[0].set_title(f"Histogram of {col}{title_suffix}")
    axes[0].set_xlabel(label)
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=data[col].dropna(), color='lightgreen', ax=axes[1])
    axes[1].set_title(f"Boxplot of {col}{title_suffix}")
    axes[1].set_xlabel(label)

    plt.tight_layout()
    figure = col.replace(' ', '_').replace('/', '').lower() + "hist_box.png"
    plt.savefig(os.path.join(output_dir, figure))
    plt.close()

# P/E distribution (already log-transformed in the data)
pe_positive = data[data["Price/Earnings"] > 0]["Price/Earnings"]

plt.figure(figsize=(5,3))
sns.histplot(pe_positive, kde=True, bins=30, color='orange')
plt.title("Distribution of P/E (log-transformed, positive only)")
plt.xlabel("log(P/E) ratio")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pe_hist.png"))
plt.close()

plt.figure(figsize=(5,3))
sns.histplot(np.log(pe_positive), kde=True, bins=30, color='orange')
plt.title("Distribution of log(log(P/E)) (positive only)")
plt.xlabel("log(log(P/E)) ratio")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_pe_hist.png"))
plt.close()

# Correlation analysis
plt.figure(figsize=(10,8))
corr_matrix = data[cols].corr(method='pearson')

# Create labels with log notation where applicable
tick_labels = [f"log({col})" if col in log_transformed_cols else col for col in cols]

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=tick_labels,
            yticklabels=tick_labels)
plt.title("Correlation Heatmap of Financial Attributes (log-transformed)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()