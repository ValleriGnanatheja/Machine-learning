#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# # Read Dataset

# In[2]:


data = pd.read_csv('US Superstore data.csv', encoding='ISO-8859-1')
data.head()


# # Data Preprocessing

# In[3]:


data.shape


# In[4]:


data.columns


# ## Check Data Type

# In[5]:


data.dtypes


# ## Check Null Values

# In[6]:


data.isnull().sum()


# ## Drop unwanted columns

# In[7]:


data=data.drop('Row ID',axis=1)
data=data.drop('Country',axis=1)
data.head()


# In[8]:


data['Category'].unique()


# In[9]:


data['Category'].value_counts()


# In[10]:


data['Sub-Category'].nunique()


# In[11]:


data['Sub-Category'].value_counts()


# # Data Visualization 

# ## Bar Plot

# In[12]:


plt.figure(figsize=(16,10))
plt.bar('Sub-Category','Category',data=data,color='green')
plt.show()


# In[13]:


plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Category')
plt.title('Count of Products by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ## Box Plot

# In[14]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Region', y='Profit')
plt.title('Profit by Region')
plt.xlabel('Region')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.show()


# ## Pie Chart

# In[15]:


plt.figure(figsize=(10,8))
data['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()


# In[16]:


plt.figure(figsize=(10,8))
data['Product Name'].value_counts().head(10).plot.pie(autopct="%1.1f%%")


# ## Histogram Plot

# In[17]:


plt.figure(figsize=(17,6))
sns.countplot(x="Sub-Category", hue="Region", data=data)
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Sales', bins=50, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# ## Scatter Plot

# In[19]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Sales', y='Profit', hue='Category')
plt.title('Sales vs. Profit')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.show()


# In[20]:


plt.figure(figsize=(8,6))
plt.scatter('Sales','Profit',data=data)
plt.legend("A")
plt.title("Scatter chart")
plt.show()


# ## Pairplot

# In[21]:


sns.pairplot(data=data, vars=['Sales', 'Quantity', 'Discount', 'Profit'])
plt.show()


# # Feature Selection

# In[22]:


selected_features = ["Sales"]
X = data[selected_features]
y = data["Profit"]


# # Split Dataset

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Linear Regression Model

# In[24]:


model = LinearRegression()
model.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)

# Evaluate Linear Regression Model
lr_r2_score = r2_score(y_test, y_pred_lr)
lr_rmse = mean_squared_error(y_test, y_pred_lr, squared=False)
print(f"Linear Regression R^2 Score: {lr_r2_score}")
print(f"Linear Regression RMSE: {lr_rmse}")


# In[25]:


y_pred = model.predict(X_test)


# In[26]:


linear_mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", linear_mse)


# # Decision Tree Model

# In[27]:


tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

tree_y_pred = tree_model.predict(X_test)

tree_mse = mean_squared_error(y_test, tree_y_pred)
print("Decision Tree Mean Squared Error:", tree_mse)


# # Random Forest Model

# In[28]:


forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

forest_y_pred = forest_model.predict(X_test)

forest_mse = mean_squared_error(y_test, forest_y_pred)
print("Random Forest Mean Squared Error:", forest_mse)


# # K-Nearest Neighbors (KNN) Model

# In[29]:


knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_y_pred = knn_model.predict(X_test)

knn_mse = mean_squared_error(y_test, knn_y_pred)
print("K-Nearest Neighbors (KNN) Mean Squared Error:", knn_mse)


# # Support Vector Machine (SVM) Model

# In[30]:


svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate SVM Model
svm_r2_score = r2_score(y_test, y_pred_svm)
svm_rmse = mean_squared_error(y_test, y_pred_svm, squared=False)
print(f"SVM R^2 Score: {svm_r2_score}")
print(f"SVM RMSE: {svm_rmse}")


# # Compare Models based on Mean Squared Error

# In[31]:


model_names = ["Linear Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]
mse_scores = [linear_mse, tree_mse, forest_mse, knn_mse, svm_rmse]

for name, mse in zip(model_names, mse_scores):
    print(f"{name} Mean Squared Error: {mse}")


# In[33]:


# Define Models

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVM': SVR(kernel='rbf')
}

# Store Results

mse_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores[name] = mse
    print(f'{name} Mean Squared Error: {mse}')

# Plot Model Comparison

plt.figure(figsize=(10, 6))
plt.bar(mse_scores.keys(), mse_scores.values(), color='skyblue')
plt.title('Model Comparison Based on Mean Squared Error (MSE)')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.show()


# In[ ]:




