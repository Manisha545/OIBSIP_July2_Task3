#!/usr/bin/env python
# coding: utf-8

# In[1]:


## import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# In[3]:


## read file

df = pd.read_csv(r'C:\Users\manis\Downloads\archive (4)\CarPrice_Assignment.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df = df.drop(columns = 'car_ID')


# In[9]:


# Identify the data types of the features

numerical_features = df.select_dtypes(include=['int', 'float']).columns
categorical_features = df.select_dtypes(include=['object']).columns


# In[10]:


# Group numerical features
numerical_data = df[numerical_features]
numerical_summary = numerical_data.describe()
print("Numerical Features Summary:")
print(numerical_summary)

# Group categorical features
categorical_data = df[categorical_features]
categorical_counts = categorical_data.nunique()
print("\nCategorical Features Counts:")
print(categorical_counts)


# In[11]:


# Create a figure and axis objects for subplots
fig, axs = plt.subplots(3, 5, figsize=(19, 10))

axs = axs.flatten()

# Iterate over the features and create scatter subplots
for i, feature in enumerate(numerical_data.columns):
    #print(i,feature)
    sns.scatterplot(data=df, x=feature, y='price', ax=axs[i])
    #axs[i].set_title(feature)


# In[12]:


# Create a figure and axis objects for subplots
fig, axs = plt.subplots(1, 3, figsize=(10, 2))

axs = axs.flatten()

# Iterate over the features and create scatter subplots
for i, feature in enumerate(['wheelbase', 'carlength', 'boreratio']):
    #print(i,feature)
    sns.boxplot(data=df, x=feature, ax=axs[i])
    #axs[i].set_title(feature)
    #axs[i].set_ylabel(feature)


# In[13]:


# Create the heatmap using seaborn
plt.figure(figsize=(10, 10))
sns.heatmap(numerical_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[14]:


# Calculate the correlation matrix
corr_matrix = numerical_data.corr()

# Extract the correlation values with the target variable
correlation_with_target = corr_matrix['price'].abs()

# Sort the features by their correlation values
sorted_features = correlation_with_target.sort_values(ascending=False)

# Print the sorted features
print("Sorted features by correlation with target:")
print(sorted_features)


# In[15]:


numerical_data.columns


# In[16]:


#continous_features = ['wheelbase', 'enginesize', 'boreratio', 'highwaympg']
continous_features = ['wheelbase', 'enginesize', 'boreratio',]


# In[17]:


## After checking multi-collinearity, decided to go with the below features

df2 = df[['wheelbase', 'enginesize', 'boreratio','price']]


# In[18]:


df2.head()


# In[19]:


categorical_data.head()


# In[20]:


categorical_data = categorical_data.drop(columns='CarName') ## drop 'CarName' column


# In[21]:


print('Unique values of each features:\n')
for feature in categorical_data.columns:
    print(feature)
    print(categorical_data[feature].unique())


# In[22]:


features_with_two_categories = ['fueltype','aspiration','doornumber','enginelocation']

# Apply one-hot encoding to multiple categorical features
encoded_df = pd.get_dummies(categorical_data, columns = features_with_two_categories)
encoded_df.head()


# In[23]:


features_with_more_categories = ['carbody','drivewheel','enginetype','fuelsystem']

# Extract the categorical features from the DataFrame
cat_data = categorical_data[['cylindernumber']]

# Create an instance of OrdinalEncoder
encoder = OrdinalEncoder()

# Fit the encoder on the categorical data
encoder.fit(cat_data)

# Perform ordinal encoding on the categorical data
encoded_data = encoder.transform(cat_data)

# Create a new DataFrame with the encoded data
encoded_df_cylindernumber = pd.DataFrame(encoded_data, columns=['cylindernumber'])

# Print the encoded DataFrame
encoded_df['cylindernumber'] = encoded_df_cylindernumber['cylindernumber']
encoded_df.head()


# In[24]:


# Create an instance of LabelEncoder
encoder = LabelEncoder()

# Perform label encoding on each categorical feature
for feature in features_with_more_categories:
    encoded_df[feature] = encoder.fit_transform(categorical_data[feature])

# Print the updated DataFrame
encoded_df.head()


# In[25]:


final_df = pd.concat([encoded_df,df2],axis = 1)
final_df.head()


# In[26]:


target = df2['price']


# In[27]:


# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df2[continous_features])


scaled_df = pd.DataFrame(scaled_features, columns=continous_features)
final_df = pd.concat([encoded_df, scaled_df,target], axis=1)


# In[28]:


final_df.head()


# In[29]:


## Split training & testing data
X = final_df.drop(columns = 'price')
Y = final_df['price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 23)

# Running linear Regression on train data
model = LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)


# Evaluate the model's performance using mean squared error (MSE)
mse = mean_squared_error(Y_test,y_pred)
print("Mean Squared Error:", mse)


# In[30]:


import statsmodels.api as sm

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:




