# %% [markdown]
# Importing the dependencies

# %% [markdown]
# Importing the Dependencies

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# %% [markdown]
# Data Collection and Analysis
# 
# PIMA Diabetes Dataset

# %%
#loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# %%
#pd.read_csv?

# %%
# printing the first five rows of the dataset
diabetes_dataset.head()

# %%
# no. of rows and cols in the dataset
diabetes_dataset.shape

# %%
# getting the statistical measures of the data
diabetes_dataset.describe()

# %%
diabetes_dataset['Outcome'].value_counts()

# %% [markdown]
# 0 --> Non- Diabetic
# 
# 1 --> Diabetic

# %%
diabetes_dataset.groupby('Outcome').mean()

# %%
#separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# %%
print(X)

# %%
print(Y)

# %% [markdown]
# Data Standardization

# %%
scaler = StandardScaler()

# %%
scaler.fit(X)

# %%
standardized_data = scaler.transform(X)

# %%
print(standardized_data)

# %%
X = standardized_data
Y = diabetes_dataset['Outcome']

# %%
print(X)
print(Y)

# %% [markdown]
# Train Test Split

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=2)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %% [markdown]
# Training the Model

# %%
classifier= svm.SVC(kernel='linear')

# %%
#training the SUPPORT VECTOR MACHINE Classifier
classifier.fit(X_train, Y_train)

# %% [markdown]
# Model Evaluation

# %% [markdown]
# Accuracy Score

# %%
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# %%
print('Accuracy score of the training data : ', training_data_accuracy)

# %%
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# %%
print('Accuracy score of the test data : ', test_data_accuracy)

# %% [markdown]
# Making a Predictive System

# %%
input_data=(5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy_array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic.')
else:
   print('The person is diabetic.')

   import matplotlib.pyplot as plt

# Counting the occurrences of each outcome
outcome_counts = diabetes_dataset['Outcome'].value_counts()

# Plotting the bar graph
plt.bar(outcome_counts.index, outcome_counts.values)
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Count of Diabetic and Non-Diabetic Individuals')
plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'])
plt.show()


# %%