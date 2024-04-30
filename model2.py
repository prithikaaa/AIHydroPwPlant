#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pickle


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset (assuming your dataset is in a CSV file)
data = pd.read_csv("hydro_pw_plant_2(1).csv")
# Handling missing values in the output column by imputing with the most frequent category
most_frequent_category = data['Hydropower_Type'].mode()[0]
data['Hydropower_Type'].fillna(most_frequent_category, inplace=True)

# Split the dataset into features (X) and target variable (y)
X = data.iloc[:, :-1]  # Assuming the input features are in the first 8 columns
y = data.iloc[:, -1]   # Assuming the target variable is in the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
# Initialize the StandardScaler
scaler = StandardScaler()

# Fill missing values with 0 before scaling
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the logistic regression model
logistic_regression_model = LogisticRegression()

# Train the model
logistic_regression_model.fit(X_train, y_train)
# Predict the target variable on the testing set
y_pred = logistic_regression_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[8]:


with open('model2.pkl','wb') as files:
    pickle.dump(logistic_regression_model,files)


# In[3]:


# Assuming you already have the trained logistic regression model

# Input values for testing
input_values = [[7.0, 400.0, 2.0009, 600.777, 500.022, 190.0, 8.2, 6.02]]  # Replace value1 to value8 with your actual input values

# Standardize input values using the same scaler used for training
input_values_standardized = scaler.transform(input_values)

# Predict the output
output_prediction = logistic_regression_model.predict(input_values_standardized)

print("Predicted output:", output_prediction[0])

