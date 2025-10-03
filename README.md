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

InvoiceDate — Date of 

UnitPrice — Price per unit

CustomerID — Unique identifier for the customer

Country — Customer country

## Feature Engineering:

Revenue = Quantity × UnitPrice

NegativeFlag = 1 if Revenue < 0, else 0 (target variable

Recency, Frequency, Monetary (RFM):

Recency — Days since last purchase

Frequency — Number of purchases

Monetary — Total monetary value of purchases

These features were chosen to summarize customer behavior and feed into the XGBoost model


# 📉 Negative Quantity Transactions Over Time

This visualization highlights fluctuations in negative quantity transactions throughout 2011. These transactions may represent returns, data entry errors, or inventory adjustments.


## 📊 Graph
<img width="1000" height="500" alt="Negative Quantity Transcation over time" src="https://github.com/user-attachments/assets/b591fa04-fbd5-467a-8b5e-da773967d8a5" />

## 🔍 Key Observation
- A significant spike occurs around **September 2011**, where the count exceeds **200** negative transactions.
- This anomaly could indicate a system issue, seasonal trend, or operational event.

> The graph plots the count of negative quantity transactions over time, with the x-axis representing dates in 2011 and the y-axis showing transaction counts.

---

# 🔁 Top 10 Customers by Number of Returns

This chart highlights the top 10 customers with the highest number of product returns in 2011, based on transactional data.

## 📊 Graph
<img width="1000" height="500" alt="top 10 customers by number of returs" src="https://github.com/user-attachments/assets/d79658c5-a447-4e1f-8164-f85c9caf1ae7" />


## 🔍 Key Insights
- **Customer 14911** had the highest number of returns, exceeding **200**.
- Return counts decrease progressively across the remaining customers.
- This pattern may indicate varying levels of satisfaction, product issues, or purchasing behavior.

## 🧠 Interpretation
Understanding which customers return the most items can help:
- Identify potential service or product quality issues
- Target high-return customers for follow-up or support
- Improve inventory and fulfillment strategies

---

Feel free to use this chart for further analysis or reporting.



# 📦 Top 10 Products by Number of Returns

This chart displays the ten products with the highest number of returns in 2011, based on transactional data. Each product is identified by a unique code or label.

## 📊 Visualization
<img width="1000" height="500" alt="top 10 products by nunber of returens" src="https://github.com/user-attachments/assets/98d8773a-449d-49fd-8a1c-f1ee55e35ce8" />


## 🔍 Key Insights
- **Product 22423** had the highest number of returns, followed by **M** and **POST**.
- The number of returns gradually declines across the remaining products.
- This pattern may reflect product quality issues, customer dissatisfaction, or inventory challenges.

## 🧠 Interpretation
Analyzing return frequency by product helps:
- Identify items with potential defects or usability concerns
- Guide quality control and supplier evaluation
- Inform product design and customer support strategies

---

This chart can be used to support deeper product performance analysis or operational decision-making.










































### [Executive Summary](#executive-summary)

After training the XGBoost classifier:

Objective: Predict whether a transaction is likely to be negative.

Data Split: 70% train / 30% test

Performance Metrics:

Accuracy: 100%

Precision: 100%

Recall: 100%

F1-score: 100%

Key Findings:

The model perfectly classified negative and non-negative transactions in the test set, as reflected in the confusion matrix: all 78,547 non-negative transactions were correctly identified, and all 1,774 negative transactions were correctly predicted.

A precision of 1.00 indicates that every predicted negative transaction was indeed negative, with no false positives.

A recall of 1.00 shows that the model captured all actual negative transactions, with no false negatives.

The F1-score, combining precision and recall, confirms excellent balance between accuracy and completeness.

These results demonstrate that the feature set and preprocessing were highly effective for this classification task, providing strong confidence in the model’s predictive power.

Such high performance may also indicate the dataset is well-structured, possibly with distinct patterns separating negative transactions from others, which the XGBoost algorithm captured efficiently.

The model is highly effective at detecting negative transactions (high recall).

Negative transactions are often associated with unusual RFM patterns, such as high recency with low monetary value



# 🤖 Classification Report & Confusion Matrix

This visualization presents the performance metrics of a binary classification model, evaluated on a dataset of 80,321 instances.

## 📊 Visualization
<img width="1000" height="500" alt="classification report" src="https://github.com/user-attachments/assets/88087795-729a-4eae-97da-f6851f8d25c0" />


## ✅ Key Metrics
- **Precision, Recall, F1-Score, Accuracy:** All equal to **1.00** for both classes (0 and 1)
- **Support:** Class 0 has 78,547 instances; Class 1 has 1,774
- **Confusion Matrix:**  




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
