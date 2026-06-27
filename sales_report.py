import pandas as pd
import numpy as np

# 1. Create a raw transaction dataset
raw_data = {
    'TransactionID': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'Date': ['2026-06-01', '2026-06-01', '2026-06-02', '2026-06-02', '2026-06-03', 
             '2026-06-03', '2026-06-04', '2026-06-04', '2026-06-05', '2026-06-05'],
    'Product': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Monitor', 
                'Mouse', 'Monitor', 'Laptop', 'Keyboard', 'Monitor'],
    'Category': ['Electronics', 'Accessories', 'Electronics', 'Accessories', 'Electronics', 
                 'Accessories', 'Electronics', 'Electronics', 'Accessories', 'Electronics'],
    'UnitsSold': [2, 15, 1, 5, 3, 22, 2, 4, 12, 5],
    'UnitPrice': [1200, 25, 1200, 75, 300, 25, 300, 1200, 75, 300],
    'Region': ['North', 'East', 'West', 'North', 'South', 'East', 'West', 'South', 'North', 'East']
}

# Load data into a DataFrame
df = pd.DataFrame(raw_data)

# 2. Feature Engineering: Calculate Total Revenue per line item
df['TotalRevenue'] = df['UnitsSold'] * df['UnitPrice']

# ==========================================
# 3. GENERATE SALES REPORT METRICS
# ==========================================

print("=== EXECUTIVE SALES SUMMARY ===")
total_sales = df['TotalRevenue'].sum()
total_units = df['UnitsSold'].sum()
total_txns = df['TransactionID'].nunique()
avg_order_value = total_sales / total_txns

print(f"Total Revenue Generated : ${total_sales:,.2f}")
print(f"Total Units Sold        : {total_units}")
print(f"Total Transactions      : {total_txns}")
print(f"Average Order Value(AOV): ${avg_order_value:,.2f}")
print("-" * 40 + "\n")

print("=== PERFORMANCE BY PRODUCT CATEGORY ===")
# Group by category and sort by revenue
category_report = df.groupby('Category').agg(
    Units_Sold=('UnitsSold', 'sum'),
    Gross_Revenue=('TotalRevenue', 'sum'),
    Txn_Count=('TransactionID', 'count')
).sort_values(by='Gross_Revenue', ascending=False)

print(category_report.to_string())
print("-" * 40 + "\n")

print("=== TOP PERFORMING PRODUCTS ===")
# Group by product name
product_report = df.groupby('Product').agg(
    Units_Sold=('UnitsSold', 'sum'),
    Gross_Revenue=('TotalRevenue', 'sum')
).sort_values(by='Gross_Revenue', ascending=False)

print(product_report)
print("-" * 40 + "\n")

print("=== REGIONAL REVENUE BREAKDOWN ===")
# Group by geography
region_report = df.groupby('Region').agg(
    Gross_Revenue=('TotalRevenue', 'sum')
).sort_values(by='Gross_Revenue', ascending=False)

# Add a percentage contribution column
region_report['%_Contribution'] = ((region_report['Gross_Revenue'] / total_sales) * 100).round(2)
print(region_report)
print("-" * 40 + "\n")
