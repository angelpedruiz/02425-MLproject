# Principal Component Analysis on log-transformed data to identify key financial indicators and variance patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "..", "data", "financials_log_transformed.csv"))
output_dir = os.path.join(script_dir, "..", "results", "pca_transformed")
os.makedirs(output_dir, exist_ok=True)


# Consider only positive P/E ratios
data = data[data["Price/Earnings"] > 0].copy()
features = ["Price", "Dividend Yield", "Earnings/Share", "52 Week Low",
            "52 Week High", "Market Cap", "EBITDA", "Price/Sales", "Price/Book"]

# Target
target = np.log(data["Price/Earnings"])

# Standardize
X = data[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=len(features))
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# Variance plots
plt.figure(figsize=(6,4))
plt.bar(range(1, len(explained_variance) + 1 ), explained_variance*100, color='gold')
plt.xticks(range(1, len(explained_variance) + 1))
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title("Scree Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scree_plot.png"))
plt.close()

cum_var = np.cumsum(explained_variance) * 100
plt.figure(figsize=(6,4))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', color='steelblue')
plt.xticks(range(1, len(cum_var) + 1))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.title("Cumulative Variance Explained")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cumulative_variance.png"))
plt.close()

# PC loadings
pc1_load = pd.Series(pca.components_[0], index=features)
pc1_order =  pc1_load.abs().sort_values(ascending=False).index
plt.figure(figsize=(6,4))
sns.barplot(x=pc1_load[pc1_order], y=pc1_order, orient='h', color='cornflowerblue')
plt.xlabel("Loading (PC1)")
plt.title("PC1 Principal Component Loadings")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PC1_loadings.png"))

pc2_load = pd.Series(pca.components_[1], index=features)
pc2_order =  pc2_load.abs().sort_values(ascending=False).index
plt.figure(figsize=(6,4))
sns.barplot(x=pc2_load[pc2_order], y=pc2_order, orient='h', color='cornflowerblue')
plt.xlabel("Loading (PC2)")
plt.title("PC2 Principal Component Loadings")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PC2_loadings.png"))

# Scatter plot
pc1_vals = X_pca[:, 0]
pc2_vals = X_pca[:, 1]
plt.figure(figsize=(6,5))
scatter = plt.scatter(pc1_vals, pc2_vals, c=target, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Log(P/E)')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Data projected onto first 2 principal components")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_scatter.png"))
plt.close()