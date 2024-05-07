import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ideal_data = pd.read_csv("C:/Users/Rajavardhanreddy/OneDrive/Desktop/PRATICE/codercave1/Hackathon_ideal_Data.csv")
working_data = pd.read_csv("C:/Users/Rajavardhanreddy/OneDrive/Desktop/PRATICE/codercave1/Hackathon_Working_Data.csv")
validation_data = pd.read_csv("C:/Users/Rajavardhanreddy/OneDrive/Desktop/PRATICE/codercave1/Hackathon_Validation_Data.csv")


merged_data = pd.concat([ideal_data, working_data])


merged_data.dropna(inplace=True)


merged_data['TOTAL_AMOUNT'] = merged_data['QTY'] * merged_data['VALUE']


summary_stats = merged_data.describe()

sales_by_category = merged_data.groupby('GRP')['TOTAL_AMOUNT'].sum().sort_values(ascending=False)
sales_by_subcategory = merged_data.groupby('SGRP')['TOTAL_AMOUNT'].sum().sort_values(ascending=False)
sales_by_brand = merged_data.groupby('BRD')['TOTAL_AMOUNT'].sum().sort_values(ascending=False)


monthly_sales = merged_data.groupby('MONTH')['TOTAL_AMOUNT'].sum()
daily_sales = merged_data.groupby('DAY')['TOTAL_AMOUNT'].sum()


from sklearn.cluster import KMeans

X = merged_data[['QTY', 'VALUE', 'TOTAL_AMOUNT']]

kmeans = KMeans(n_clusters=3, random_state=42)
merged_data['Cluster'] = kmeans.fit_predict(X)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

merged_data_encoded = pd.get_dummies(merged_data[['GRP', 'SGRP', 'BRD']])

frequent_itemsets = apriori(merged_data_encoded, min_support=0.05, use_colnames=True)


association_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

plt.figure(figsize=(30, 10))  
sns.barplot(x=sales_by_category.index, y=sales_by_category.values, palette="viridis")
plt.title('Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right', fontsize=10)  
plt.tight_layout()  
plt.show()


plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='bar', color='skyblue')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x='QTY', y='VALUE', hue='Cluster', palette='viridis')
plt.title('Customer Segments by Quantity and Value')
plt.xlabel('Quantity')
plt.ylabel('Value')
plt.show()

association_rules["antecedents"] = association_rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
association_rules["consequents"] = association_rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")


plt.figure(figsize=(10, 6))
sns.scatterplot(x="antecedent support", y="confidence", size="lift", data=association_rules)
plt.title('Association Rules')
plt.xlabel('Antecedent Support')
plt.ylabel('Confidence')
plt.show()

