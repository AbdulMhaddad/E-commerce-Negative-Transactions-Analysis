Business Intelligence for E-Commerce: Negative Transaction Analysis
📝 Project Overview

This project analyzes an e-commerce dataset to uncover insights around negative transactions (returns, cancellations, or errors). The goal is to simulate a real-world business intelligence workflow: from raw data → cleaning → feature engineering → predictive modeling → visualization.

The project also implements a machine learning model to predict negative transactions, helping e-commerce owners proactively identify problem areas.


❓ Business Problem

E-commerce companies face challenges like:

Transactions with negative quantities or revenues.

Identifying customers or products driving negative transactions.

Understanding patterns over time to reduce operational losses.

This project addresses these challenges using Python-based analysis and XGBoost classification.



📂 Project Workflow
1. Data Cleaning & Preparation

Removed duplicates and missing values.

Converted InvoiceDate to datetime and CustomerID to integer.

Checked for outliers in Quantity and UnitPrice.

2. Negative Transaction Analysis

Counted and visualized negative quantities.

Analyzed top customers and products with negative transactions.

Categorized negative transactions as Return, Discount/Other, or Correction/Error.

Tracked negative transactions over time with line and bar charts.

3. Feature Engineering & Modeling

Created Revenue and NetRevenue columns (set negative quantities to 0 for NetRevenue).

Adjusted inventory quantities for negative transactions.

Extracted date features: InvoiceMonth, InvoiceDay, InvoiceWeekday.

Reduced high-cardinality categorical features (CustomerID and StockCode) to top 100 + “Other”.

One-hot encoded categorical features: CustomerID_reduced, StockCode_reduced, Country.

4. Predictive Modeling with XGBoost

Target variable: NegativeFlag (1 if Quantity < 0, else 0).

Train/Test Split: 80/20 stratified split.

XGBoost Classifier Parameters:

n_estimators=100, max_depth=6, learning_rate=0.1

Handled class imbalance with scale_pos_weight.

Evaluated using accuracy, confusion matrix, and classification report.

Plotted feature importance to identify drivers of negative transactions.



📊 Key Insights

Certain customers and products are consistently associated with negative transactions.

Negative transactions show seasonal trends, with spikes at specific periods.

Feature importance from XGBoost provides actionable insights to reduce operational risks




🛠️ Tech Stack

Python Libraries: pandas, numpy, matplotlib, seaborn, xgboost, lightgbm, scipy, scikit-learn

Visualization Tools: matplotlib, seaborn

Machine Learning: XGBoost Classifier
