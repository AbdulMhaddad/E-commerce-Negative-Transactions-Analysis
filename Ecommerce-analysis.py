import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Files\Portfilo\Project 1\data.csv (1)\data.csv", 
                 encoding="ISO-8859-1")


#EDA processing

print(df.head())


print("\n Info of the Data Set \n:")
print(df.info())


print(" Describtion of the DataSet")
print(df.describe())


print("Columns of the DataSet")
print(df.columns.tolist())


print("Missing Vules of the Dataset")
print(df.isnull().sum())

duplicates = df[df.duplicated()]
print("Duplicate Rows in the dataset")
print(duplicates)

###cleaning 

df_cleaning = df.copy()

print(f"Shape before removing duplicates: {df.shape}")
df_cleaning =df_cleaning.drop_duplicates()

print(f"Shape after removing duplicates: {df_cleaning.shape}")

print(f"\nShape before removing missing values: {df_cleaning.shape}")

# Remove rows with any missing values
df_cleaning = df_cleaning.dropna()

# Check missing values after
print("\nMissing values per column after cleaning:")
print(df_cleaning.isnull().sum())

print(f"\nShape after removing missing values: {df_cleaning.shape}")



# Convert InvoiceDate to datetime
df_cleaning['InvoiceDate'] = pd.to_datetime(df_cleaning['InvoiceDate'], errors='coerce')

# Optionally, convert CustomerID to Int (after dropping missing CustomerIDs)
df_cleaning = df_cleaning.dropna(subset=['CustomerID'])
df_cleaning['CustomerID'] = df_cleaning['CustomerID'].astype(int)

# Check updated types
print(df_cleaning.dtypes)



# ===============================
# 8. Check for Outliers
# ===============================

numeric_cols = ['Quantity', 'UnitPrice']

for col in numeric_cols:
    Q1 = df_cleaning[col].quantile(0.25)
    Q3 = df_cleaning[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_count = df_cleaning[(df_cleaning[col] < lower_bound) | (df_cleaning[col] > upper_bound)].shape[0]
    print(f"{col} - Number of outliers: {outlier_count}")



#analysis part
df_analysis = df_cleaning.copy()

# ===============================
# 9. Negative Value Analysis
# ===============================

# Count negative Quantity and Revenue
# ===============================
# 9. Negative Quantity Analysis
# ===============================

# Count negative Quantity
neg_quantity_count = (df_analysis['Quantity'] < 0).sum()
print(f"Negative Quantity Count: {neg_quantity_count}")

# Sample negative Quantity transactions
print("\nSample Negative Quantity Transactions:")
print(df_analysis[df_analysis['Quantity'] < 0].head())




# ===============================
# 10. Negative Quantity by Customer
# ===============================

neg_by_customer = df_analysis[df_analysis['Quantity'] < 0].groupby('CustomerID')['Quantity'].count().sort_values(ascending=False)
print("Top 10 Customers with Negative Quantities:")
print(neg_by_customer.head(10))


# ===============================
# 11. Negative Quantity by Product
# ===============================

neg_by_product = df_analysis[df_analysis['Quantity'] < 0].groupby('StockCode')['Quantity'].count().sort_values(ascending=False)
print("Top 10 Products with Negative Quantities:")
print(neg_by_product.head(10))


# ===============================
# 12. Negative Quantity Over Time
# ===============================

# Ensure InvoiceDate is datetime
df_analysis['InvoiceDate'] = pd.to_datetime(df_analysis['InvoiceDate'])

# Filter negative quantities
neg_df = df_analysis[df_analysis['Quantity'] < 0]

# Group by invoice date
neg_over_time = neg_df.groupby(neg_df['InvoiceDate'].dt.date)['Quantity'].count()

# Plot
plt.figure(figsize=(12,6))
neg_over_time.plot()
plt.title("Negative Quantity Transactions Over Time")
plt.xlabel("Date")
plt.ylabel("Count of Negative Quantities")
plt.show()

# ===============================
# Summary of Negative Quantity Over Time Figure
# ===============================

# - Overall Trend: Negative quantity transactions occur throughout the year
#   but are concentrated on specific days.
# - Spikes: Certain days have significant peaks, likely caused by returns,
#   invoice cancellations, or bulk adjustments.
# - Flat Periods: Many days show very few or no negative transactions,
#   representing normal operations.
# - Pattern Insight: Negatives are not evenly distributed; they tend to
#   cluster around specific events or customers/products.
# - Business Implication: Negative quantities are episodic rather than
#   constant, helping focus analysis on high-impact days and transactions.



# ===============================
# 13. Categorize Negative Quantities
# ===============================

def categorize_negative(row):
    if row['InvoiceNo'].startswith('C'):
        return 'Return'
    elif row['StockCode'] in ['D', 'M', 'POST']:
        return 'Discount/Other'
    else:
        return 'Correction/Error'

df_analysis['NegativeType'] = df_analysis.apply(
    lambda row: categorize_negative(row) if row['Quantity'] < 0 else None, axis=1
)

# Check the distribution of negative types
neg_type_counts = df_analysis['NegativeType'].value_counts()
print("Distribution of Negative Quantity Types:")
print(neg_type_counts)

print(df_analysis.shape)

print(df_analysis.columns.tolist())



# 1. Top 10 Customers by Number of Returns
top_customers = df_analysis[df_analysis['Quantity'] < 0].groupby('CustomerID')['Quantity'].count().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_customers.index.astype(str), y=top_customers.values, palette="viridis")
plt.title("Top 10 Customers by Number of Returns")
plt.xlabel("CustomerID")
plt.ylabel("Number of Returns")
plt.xticks(rotation=45)
plt.show()

# 2. Top 10 Products by Number of Returns
top_products = df_analysis[df_analysis['Quantity'] < 0].groupby('StockCode')['Quantity'].count().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_products.index, y=top_products.values, palette="magma")
plt.title("Top 10 Products by Number of Returns")
plt.xlabel("StockCode")
plt.ylabel("Number of Returns")
plt.xticks(rotation=45)
plt.show()

# 3. Returns Over Time
neg_df = df_analysis[df_analysis['Quantity'] < 0]
returns_over_time = neg_df.groupby(neg_df['InvoiceDate'].dt.date)['Quantity'].count()

plt.figure(figsize=(12,6))
returns_over_time.plot()
plt.title("Returns (Negative Quantities) Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Returns")
plt.show()


# ===============================
# 15. Storytelling and Business Insights
# ===============================

# Summary Insights (can be added as comments or markdown in VS Code)

# - A total of 8,872 transactions have negative quantities, indicating returns.
# - Returns are concentrated among a few customers:
#     - Top customer (ID 14911) alone accounts for 226 returns.
#     - A small group of customers is responsible for most negative transactions.
# - Certain products are returned more frequently:
#     - StockCode 22423 has the highest number of returns (180).
#     - Some stock codes like 'M', 'D', or 'POST' may represent discounts or adjustments.
# - Returns are episodic, not evenly distributed over time:
#     - Spikes likely correspond to promotions, holiday periods, or bulk returns.
# - Business Implications:
#     - Focus on high-return customers for targeted retention or feedback.
#     - Review frequently returned products for quality issues or packaging.
#     - Monitor periods with spikes to identify operational or promotional issues.


#solving the issue


# ===============================
# 16. Create NegativeFlag Column
# ===============================

# Flag transactions with negative quantity
df_analysis['NegativeFlag'] = (df_analysis['Quantity'] < 0).astype(int)

# Check the counts
neg_flag_counts = df_analysis['NegativeFlag'].value_counts()
print("Counts of NegativeFlag:")
print(neg_flag_counts)

# Sample flagged transactions
print("\nSample NegativeFlagged Transactions:")
print(df_analysis[df_analysis['NegativeFlag'] == 1].head())




# ===============================
# 17. Create Revenue and Adjust for Negatives
# ===============================

# Create Revenue column
df_analysis['Revenue'] = df_analysis['Quantity'] * df_analysis['UnitPrice']

# Create NetRevenue column (set negative quantities to 0)
df_analysis['NetRevenue'] = df_analysis['Revenue']
df_analysis.loc[df_analysis['NegativeFlag'] == 1, 'NetRevenue'] = 0

# Check sample
print(df_analysis[['InvoiceNo', 'Quantity', 'UnitPrice', 'Revenue', 'NetRevenue', 'NegativeFlag']].head(10))

# Optional: Total revenues
total_revenue = df_analysis['Revenue'].sum()
net_revenue = df_analysis['NetRevenue'].sum()
print(f"\nTotal Revenue (with negatives): {total_revenue}")
print(f"Net Revenue (negatives set to 0): {net_revenue}")



# ===============================
# 18. Adjust Inventory for Negative Quantities
# ===============================

# Create AdjustedQuantity column
df_analysis['AdjustedQuantity'] = df_analysis['Quantity']
df_analysis.loc[df_analysis['NegativeFlag'] == 1, 'AdjustedQuantity'] = 0

# Check sample
print(df_analysis[['InvoiceNo', 'Quantity', 'AdjustedQuantity', 'NegativeFlag']].head(10))

# Optional: Total quantity before and after adjustment
total_quantity = df_analysis['Quantity'].sum()
adjusted_quantity = df_analysis['AdjustedQuantity'].sum()
print(f"\nTotal Quantity (original): {total_quantity}")
print(f"Total Quantity (adjusted): {adjusted_quantity}")


#df modling

df_modeling = df_analysis.copy()

# ===============================
# 20. Prepare Data for Modeling
# ===============================

# Feature engineering: extract date features
df_analysis['InvoiceDate'] = pd.to_datetime(df_analysis['InvoiceDate'])
df_analysis['InvoiceMonth'] = df_analysis['InvoiceDate'].dt.month
df_analysis['InvoiceDay'] = df_analysis['InvoiceDate'].dt.day
df_analysis['InvoiceWeekday'] = df_analysis['InvoiceDate'].dt.weekday

# Select features and target
features = ['Quantity', 'UnitPrice', 'CustomerID', 'StockCode', 'Country', 'InvoiceMonth', 'InvoiceDay', 'InvoiceWeekday']
target = 'NegativeFlag'

X = df_analysis[features]
y = df_analysis[target]

# Encode categorical features (CustomerID, StockCode, Country)
X_encoded = pd.get_dummies(X, columns=['CustomerID', 'StockCode', 'Country'], drop_first=True)

print(f"Features shape after encoding: {X_encoded.shape}")
print(f"Target distribution:\n{y.value_counts()}")




# ===============================
# 22. Train/Test Split
# ===============================


# Copy df_modeling
df_modeling_reduced = df_modeling.copy()


# -------------------------------
# 2. Reduce high-cardinality categorical features
# -------------------------------
# Keep top 100 customers and group rest as 'Other'
top_customers = df_modeling_reduced['CustomerID'].value_counts().nlargest(100).index
df_modeling_reduced['CustomerID_reduced'] = df_modeling_reduced['CustomerID'].where(df_modeling_reduced['CustomerID'].isin(top_customers), 'Other')

# Keep top 100 products and group rest as 'Other'
top_products = df_modeling_reduced['StockCode'].value_counts().nlargest(100).index
df_modeling_reduced['StockCode_reduced'] = df_modeling_reduced['StockCode'].where(df_modeling_reduced['StockCode'].isin(top_products), 'Other')

# -------------------------------
# 3. Feature Engineering from InvoiceDate
# -------------------------------
df_modeling_reduced['InvoiceDate'] = pd.to_datetime(df_modeling_reduced['InvoiceDate'])
df_modeling_reduced['InvoiceMonth'] = df_modeling_reduced['InvoiceDate'].dt.month
df_modeling_reduced['InvoiceDay'] = df_modeling_reduced['InvoiceDate'].dt.day
df_modeling_reduced['InvoiceWeekday'] = df_modeling_reduced['InvoiceDate'].dt.weekday

# -------------------------------
# 4. Select Features and Target
# -------------------------------
features = ['Quantity', 'UnitPrice', 'CustomerID_reduced', 'StockCode_reduced', 'Country',
            'InvoiceMonth', 'InvoiceDay', 'InvoiceWeekday']
target = 'NegativeFlag'

X = df_modeling_reduced[features]
y = df_modeling_reduced[target]

# -------------------------------
# 5. One-hot encode categorical features
# -------------------------------
X_encoded = pd.get_dummies(X, columns=['CustomerID_reduced', 'StockCode_reduced', 'Country'], drop_first=True)

# Fill NaNs and ensure numeric dtype
X_encoded = X_encoded.fillna(0).astype(float)

print(f"Feature shape after reduction and encoding: {X_encoded.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# -------------------------------
# 6. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to sparse matrices for XGBoost
X_train_sparse = sparse.csr_matrix(X_train)
X_test_sparse = sparse.csr_matrix(X_test)

print(f"Training shape: {X_train_sparse.shape}, Test shape: {X_test_sparse.shape}")

# -------------------------------
# 7. Train XGBoost Classifier
# -------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),  # handle imbalance
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_sparse, y_train)

# -------------------------------
# 8. Predictions and Evaluation
# -------------------------------
y_pred = xgb_model.predict(X_test_sparse)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")


# Get feature importance
importance = xgb_model.feature_importances_
features = X_encoded.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)  # Top 20

# Plot
plt.figure(figsize=(12,6))
plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])
plt.xlabel('Importance')
plt.title('Top 20 Features Predicting Negative Quantity')
plt.show()