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

The primary dataset comes from Kaggle, containing e-commerce transactions.

Additional transactions may be simulated or provided for testing the model.

Raw Columns:

InvoiceNo — Unique identifier for each invoice

StockCode — Product code

Description — Product description

Quantity — Number of items per invoice

InvoiceDate — Date of transaction

UnitPrice — Price per unit

CustomerID — Unique identifier for the customer

Country — Customer country

Feature Engineering:

Revenue = Quantity × UnitPrice

NegativeFlag = 1 if Revenue < 0, else 0 (target variable)

Recency, Frequency, Monetary (RFM):

Recency — Days since last purchase

Frequency — Number of purchases

Monetary — Total monetary value of purchases

These features were chosen to summarize customer behavior and feed into the XGBoost model.
### [Executive Summary](#executive-summary)
### [Insight Deep Dive](#insight-deep-dive)
 sdasdasd































### [Recommendations](#recommendations)
