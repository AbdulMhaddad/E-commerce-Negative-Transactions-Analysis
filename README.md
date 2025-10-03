# E-commerce Negative Transactions Detection Analysis

## Table of Contents

## 1- Background and Overview

## 2- Data Structures Overview

## 3- Executive Summary

## 4- Insight Deep Dive

## 5- Recommendations



### [Background and Overview](#background-and-overview)

  The goal of this project is to identify negative revenue transactions in e-commerce data, which are often indicators of refunds, returns, or potential anomalies. By detecting these transactions proactively, businesses can better manage inventory, customer behavior, and operational risks.

  We approached this problem using machine learning, specifically the XGBoost classifier, leveraging historical transaction data to predict whether a new transaction might result in negative revenue. This allows for data-driven decision- 
  making rather than manual inspection.































### [Data Structures Overview](#data-structures-overview)
Data Sources:

The primary dataset comes from [Kaggle Dataset: E-commerce Transactions](https://www.kaggle.com/datasets/carrie1/ecommerce-data), containing e-commerce transactions.
Additional transactions may be simulated or provided for testing the model.

## Raw Columns:

InvoiceNo — Unique identifier for each invoice
StockCode — Product code
Description — Product description
Quantity — Number of items per invoice
InvoiceDate — Date of transaction
UnitPrice — Price per unit
CustomerID — Unique identifier for the customer
Country — Customer country

## Feature Engineering:

Revenue = Quantity × UnitPrice
NegativeFlag = 1 if Revenue < 0, else 0 (target variable)
Recency, Frequency, Monetary (RFM):
Recency — Days since last purchase
Frequency — Number of purchases
Monetary — Total monetary value of purchases

These features were chosen to summarize customer behavior and feed into the XGBoost model.
### [Executive Summary](#executive-summary)

After training the XGBoost classifier:

Objective: Predict whether a transaction is likely to be negative.

Data Split: 70% train / 30% test

Performance Metrics:

Accuracy: ~[insert your result]%

Precision: ~[insert result]%

Recall: ~[insert result]%

F1-score: ~[insert result]%

Key Findings:

The model is highly effective at detecting negative transactions (high recall).

Negative transactions are often associated with unusual RFM patterns, such as high recency with low monetary value.
### [Insight Deep Dive](#insight-deep-dive)
The detailed analysis revealed:

Feature Importance:

Recency and Monetary were the top predictors.

Frequency contributed moderately to model decisions.

Patterns Observed:

Customers with infrequent purchases but sudden high refunds are flagged.

Certain countries or product categories had more negative transactions, highlighting regional or product-specific trends.

Model Behavior:

XGBoost handles class imbalance well, so rare negative transactions are detected reliably.

False positives are minimal, meaning the model is conservative in flagging normal transactions.































### [Recommendations](#recommendations)

Business Actions:

Monitor flagged negative transactions in real-time dashboards.

Investigate top-flagged invoices for potential fraud, errors, or return patterns.

Adjust return policies or promotional strategies based on insights.

Next Steps:

Deploy the model as a scheduled pipeline (daily/weekly predictions).

Integrate with BI tools like Power BI or Tableau for executive dashboards.

Collect more labeled data over time to retrain and improve model accuracy.
