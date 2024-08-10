import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Function to convert age range to midpoint
def convert_age_range(age_range):
    if '-' in age_range:
        start, end = age_range.split('-')
        return (int(start) + int(end.split()[0])) / 2
    return int(age_range.split()[0])

# Load the customer data from CSV
customer_data = pd.read_csv('D:/Projects/Small_Data_Science_Projects/Customer Segmentation with K-Means/customer_data.csv')

# Check the first few rows to understand the structure
print(customer_data.head())

# Convert age ranges to midpoints
customer_data['Avg_age'] = customer_data['Avg_age'].apply(convert_age_range)

# Select the features for clustering
X = customer_data[['Avg_Salary', 'Avg_age']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_data['Avg_Salary'], y=customer_data['Avg_age'], hue=customer_data['Cluster'], palette='viridis', s=100)
plt.title('Customer Segmentation with K-Means')
plt.xlabel('Average Salary')
plt.ylabel('Average Age')
plt.show()
