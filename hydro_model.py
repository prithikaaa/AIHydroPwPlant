#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import FunctionTransformer


# In[80]:


import pickle


# In[81]:


import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import FunctionTransformer
original_data = pd.read_csv('hydro_pw_plant_data.csv')

# Separate input features and target columns
input_features = original_data.iloc[:, :8]  # Assuming first 8 columns are input features
target_columns = original_data.iloc[:, 8:]  # Assuming last 3 columns are target columns
# Apply logarithmic transformation to ensure non-negativity
transformer = FunctionTransformer(np.log1p, np.expm1)
transformed_target_columns = transformer.fit_transform(input_features,target_columns)
# Initialize CTGAN synthesizer
ctgan = CTGAN(epochs=10)

# Fit the CTGAN model to your data
ctgan.fit(original_data)

# Generate synthetic data
num_samples = 1000  # Number of synthetic samples to generate
synthetic_samples = ctgan.sample(num_samples)
# Reverse the logarithmic transformation to obtain original scale
inverse_transformed_samples = transformer.inverse_transform(synthetic_samples)

# Convert synthetic samples to DataFrame
synthetic_data = pd.DataFrame(inverse_transformed_samples, columns=target_columns.columns)

# Optionally, combine synthetic samples with original data
combined_data = pd.concat([original_data, synthetic_data], ignore_index=True)


# In[82]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
train = combined_data


# In[92]:


import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
# Separate features and targets
X = train.iloc[:, :-3]  # Features
y = train.iloc[:, -3:]  # Targets

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute the 85th percentile of the target variable
lower_bound = np.percentile(y_train, 5)
upper_bound = np.percentile(y_train, 37)
# Initialize the imputer
imputer = SimpleImputer(strategy='mean')  # You can change the strategy as needed

# Fit the imputer on the training data and transform both training and test data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
# Clip target variable to remove outliers
y_train_clipped = np.clip(y_train, a_min=lower_bound, a_max=upper_bound)
y_test_clipped = np.clip(y_test, a_min=lower_bound, a_max=upper_bound)

# Convert clipped arrays back to DataFrame
y_train_clipped_df = pd.DataFrame(y_train_clipped, columns=y.columns)
y_test_clipped_df = pd.DataFrame(y_test_clipped, columns=y.columns)

# Remove rows with NaN values
y_train_clipped_df = y_train_clipped_df.dropna()
y_test_clipped_df = y_test_clipped_df.dropna()

# Convert back to numpy arrays
y_train_clipped_cleaned = y_train_clipped_df.values
y_test_clipped_cleaned = y_test_clipped_df.values

# Initialize the base estimator (Random Forest Regressor)
base_estimator = RandomForestRegressor()

# Initialize the MultiOutput Regressor with the base estimator
multioutput_regressor = MultiOutputRegressor(base_estimator)

# Fit the MultiOutput Regressor to the training data
multioutput_regressor.fit(X_train_imputed, y_train_clipped_cleaned)

# Predict on the test set
y_pred = multioutput_regressor.predict(X_test_imputed)



# In[93]:


from sklearn.metrics import r2_score, mean_absolute_error

# Calculate R-squared
r2 = r2_score(y_test_clipped_cleaned, y_pred)
print("R-squared:", r2)

# Calculate mean absolute error
mae = mean_absolute_error(y_test_clipped_cleaned, y_pred)
print("Mean Absolute Error:", mae)


# In[86]:


with open('model.pkl','wb') as files:
    pickle.dump(multioutput_regressor,files)


# In[ ]:


input_row = np.array([7.0, 400.0, 2.0009, 200.777, 600.022, 190.0, 8.2, 6.02])
input_row_reshaped = input_row.reshape(1, -1)
output_values = multioutput_regressor.predict(input_row_reshaped)
print("Predicted values for the last 3 columns:")
print(output_values)

