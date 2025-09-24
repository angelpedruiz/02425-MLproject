# Correlation analysis and distribution analysis of S&P 500 financial variables

# Correlation analysis and distribution analysis of S&P 500 financial variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

data = pd.read_csv("C:\\Users\\joluf\\OneDrive\\Documentos\\GitHub\\02425-MLproject\\data\\financials.csv")
output_dir = os.path.join(os.path.dirname(_file_), "..", "Plots")

# Columns '52 Week Low' and '52 Week High' data are reversed
data["52 Week Low"], data["52 Week High"] = data["52 Week High"].copy(), data["52 Week Low"].copy()

# Drop rows with missing values
data = data.dropna(subset=["Price/Earnings", "Price/Book"])

cols = ["Price", "Price/Earnings", "Dividend Yield", "Earnings/Share", "52 Week Low",
        "52 Week High", "Market Cap", "EBITDA", "Price/Sales", "Price/Book"]

# Histograms + Boxplots by attribute
for col in cols:
    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    sns.histplot(data[col].dropna(), kde=False, bins=30, color='skyblue', ax=axes[0])
    axes[0].set_title(f"Histogram of {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=data[col].dropna(), color='lightgreen', ax=axes[1])
    axes[1].set_title(f"Boxplot of {col}")
    axes[1].set_xlabel(col)

    plt.tight_layout()
    figure = col.replace(' ', '_').replace('/', '').lower() + "hist_box.png"
    plt.savefig(os.path.join(output_dir, figure))
    plt.close()

# P/E distribution with log transformation
pe_positive = data[data["Price/Earnings"] > 0]["Price/Earnings"]

plt.figure(figsize=(5,3))
sns.histplot(pe_positive, kde=True, bins=30, color='orange')
plt.title("Distribution of P/E (positive only)")
plt.xlabel("P/E ratio")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pe_hist.png"))
plt.close()

plt.figure(figsize=(5,3))
sns.histplot(np.log(pe_positive), kde=True, bins=30, color='orange')
plt.title("Distribution of P/E (positive only)")
plt.xlabel("log(P/E) ratio")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_pe_hist.png"))
plt.close()

# Correlation analysis
plt.figure(figsize=(10,8))
corr_matrix = data[cols].corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=cols,
            yticklabels=cols)
plt.title("Correlation Heatmap of Financial Attributes")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()