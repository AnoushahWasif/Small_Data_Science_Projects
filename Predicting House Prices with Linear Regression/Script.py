import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
house_data = pd.read_csv('D:/Projects/Small_Data_Science_Projects/Predicting House Prices with Linear Regression/house-prices.csv')
print(house_data.head())

# Prepare the data for training
# Using 'SqFt', 'Bedrooms', 'Bathrooms', 'Offers' as features
X = house_data[['SqFt', 'Bedrooms', 'Bathrooms', 'Offers']]
y = house_data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()