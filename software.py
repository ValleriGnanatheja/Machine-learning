#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # Load dataset

# In[2]:


df = pd.read_csv("medical_costs.csv")

# Display first few rows
print(df.head())


# # Data Preprocessing

# In[3]:


# Check for missing values
print(df.isnull().sum())


# In[4]:


# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Smoker', 'Region'], drop_first=True)


# In[5]:


# Scale numerical features
scaler = StandardScaler()
df[['Age', 'BMI', 'Children']] = scaler.fit_transform(df[['Age', 'BMI', 'Children']])


# In[6]:


# Define features (X) and target variable (y)
X = df.drop(columns=['Medical Cost'])
y = df['Medical Cost']


# In[7]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Exploratory Data Analysis 

# In[8]:


plt.figure(figsize=(8,5))
sns.histplot(df['Medical Cost'], bins=30, kde=True)
plt.title("Distribution of Medical Costs")
plt.show()


# In[9]:


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Age'], y=df['Medical Cost'])
plt.title("Age vs Medical Cost")
plt.show()


# In[10]:


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['BMI'], y=df['Medical Cost'])
plt.title("BMI vs Medical Cost")
plt.show()


# In[11]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df['Smoker_yes'], y=df['Medical Cost'])
plt.title("Smoker vs Medical Cost")
plt.show()


# In[12]:


plt.figure(figsize=(8,5))
sns.barplot(x=df['Children'], y=df['Medical Cost'])
plt.title("Number of Children vs Medical Cost")
plt.show()


# In[13]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# # Model 

# In[14]:


# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[15]:


# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression Model
print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))


# In[16]:


# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[17]:


# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest Model
print("Random Forest Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))


# In[18]:


# Compare model performance
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf)],
    'MSE': [mean_squared_error(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_rf)],
    'R2 Score': [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf)]
})

print(results)


# In[ ]:




