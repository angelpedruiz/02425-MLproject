# Principal Component Analysis to identify key financial indicators and variance patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

data = pd.read_csv("C:\\Users\\joluf\\OneDrive\\Documentos\\GitHub\\02425-MLproject\\data\\financials.csv")
output_dir = os.path.join(os.path.dirname(__file__), "..", "Plots")

# Columns '52 Week Low' and '52 Week High' data are reversed
data["52 Week Low"], data["52 Week High"] = data["52 Week High"].copy(), data["52 Week Low"].copy()

# Drop rows with missing values
data = data.dropna(subset=["Price/Earnings", "Price/Book"])

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

pc3_load = pd.Series(pca.components_[2], index=features)
pc3_order =  pc3_load.abs().sort_values(ascending=False).index
plt.figure(figsize=(6,4))
sns.barplot(x=pc3_load[pc3_order], y=pc3_order, orient='h', color='cornflowerblue')
plt.xlabel("Loading (PC3)")
plt.title("PC3 Principal Component Loadings")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PC3_loadings.png"))

pc4_load = pd.Series(pca.components_[3], index=features)
pc4_order =  pc4_load.abs().sort_values(ascending=False).index
plt.figure(figsize=(6,4))
sns.barplot(x=pc4_load[pc4_order], y=pc4_order, orient='h', color='cornflowerblue')
plt.xlabel("Loading (PC4)")
plt.title("PC4 Principal Component Loadings")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PC4_loadings.png"))

# 2D Scatter plots
pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
pc_vals = [X_pca[:,i] for i in range(4)]
for i,j in pairs:
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(pc_vals[i], pc_vals[j], c=target, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Log(P/E)')
    plt.xlabel(f"PC{i+1}")
    plt.ylabel(f"PC{j+1}")
    plt.title(f"Scatter: PC{i+1} vs PC{j+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_pc{i+1}_pc{j+1}.png"))
    plt.close()

# 3D Scatter plots
triples = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
from mpl_toolkits.mplot3d import Axes3D 
for i,j,k in triples:
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(pc_vals[i], pc_vals[j], pc_vals[k], c=target, cmap='viridis', alpha=0.7)
    ax.set_xlabel(f"PC{i+1}")
    ax.set_ylabel(f"PC{j+1}")
    ax.set_zlabel(f"PC{k+1}")
    ax.set_title(f"3D Scatter: PC{i+1}, PC{j+1}, PC{k+1}")
    fig.colorbar(p, ax=ax, label="Log(P/E)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_pc{i+1}_pc{j+1}_pc{k+1}.png"))
    plt.close()