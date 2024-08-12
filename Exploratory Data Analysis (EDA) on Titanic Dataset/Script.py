import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
titanic_data = pd.read_csv('D:/Projects/Small_Data_Science_Projects/Exploratory Data Analysis (EDA) on Titanic Dataset/titanic.csv')

# Display basic information
print(titanic_data.info())

# Display basic statistics
print(titanic_data.describe())

# Visualize missing data
sns.heatmap(titanic_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Visualize survival rate by gender
sns.barplot(x='Sex', y='Survived', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.show()

# Visualize survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=titanic_data)
plt.title('Survival Rate by Passenger Class')
plt.show()
