# Financial Data Analysis Project

Machine Learning analysis of S&P 500 financial metrics for DTU course 02450.

## Overview

Analysis of financial data with focus on Price/Earnings ratio prediction and classification using three main components:

- **data_analysis.py**: Data exploration, attribute analysis, and preprocessing of financial metrics
- **pca.py**: Principal Component Analysis to identify key financial indicators  
- **statistical_analysis.py**: Correlation analysis and distribution analysis of financial variables

## Dataset

**financials.csv**: S&P 500 companies with 14 financial attributes including Price, P/E ratio, Market Cap, EBITDA, sector classifications, and performance metrics.

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Data Analysis**: `python description/data_analysis.py`
   - Identifies discrete vs continuous financial metrics
   - Handles missing values and outliers in financial data
   - Provides summary statistics for all financial indicators

2. **PCA Analysis**: `python description/pca.py`
   - Determines which financial metrics explain most variance
   - Creates 2D visualization of companies in principal component space
   - Shows feature importance for financial analysis

3. **Statistical Analysis**: `python description/statistical_analysis.py`
   - Analyzes distributions of financial ratios and metrics
   - Creates correlation heatmap of financial variables
   - Identifies redundant financial indicators

## Objectives

- **Regression**: Predict P/E ratios from other financial metrics
- **Classification**: Binary classification of P/E ratios (high/low)

Results and visualizations saved in `results/` directory.
