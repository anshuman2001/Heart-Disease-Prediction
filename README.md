# Heart Disease Prediction: Logistic Regression Evaluation and Imbalanced Data Handling.
# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import warnings 
warnings.filterwarnings('ignore')
# Step 2: Load dataset
df = pd.read_excel(r"C:\Users\anshu\OneDrive\Desktop\heart.xlsx")
df
# Step 3:  Exploratory Data Analysis (EDA)
## Display the top 5 rows of the dataframe¶
df_shape =df.shape 
print("Data shape:", df_shape)
df.rename(columns={'target': 'heart_diagnosis'},inplace = True)
df=df.head(5)
df

## Print the shape of the dataframe
df_shape =df.shape 
print("Data shape:", df_shape)
## Information about data

df_info = df.info()
df_info
## Drop the rows with missing values
df.dropna(inplace=True)    
## Check null values 
df_null_count = df.isnull().sum()
df_null_count
## Summary statistics

#For numerical columns
df_summary_numeric=df.describe(include='all')
df_summary_numeric
# Check correlations
* The correlation heatmap shows the correlation between all the numerical variables. This indicates which variables are related to the target and to each other.The magnitude and sign indicate the strength and direction of linear relation.

Hints:

1. You need to run corr on your dataframe
2. Pass the variable containing correlation information to heatmap
3. We are setting numeric_only=True to run correlation on only numeric variables
plt.figure(figsize=(10, 8))
corr = df.corr()
correlation_heatmap=sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot=True, cmap = 'viridis', vmin = -1, vmax = 1, center = 0,  fmt=".2f")
plt.show()
## Check distribution of the target variable
Hints:

You can use the value_counts method on the target column

heart_diagnosis_distribution = df['heart_diagnosis'].value_counts()
print("Target variable distribution:\n", heart_diagnosis_distribution)
## Histograms of all numerical columns
Hints:

1.For defining the number of columns and rows for subplots:

2.Calculate the number of columns in your DataFrame using df.shape[1].
3.Determine the number of rows needed for subplots by subtracting 1 from the number of columns and then using integer division (// 2 + 1).
4.For creating subplots:

5.Use plt.subplots() to create subplots. Specify the number of rows and columns using the values calculated in step 2.
6.Adjust the vertical spacing between subplots using fig.subplots_adjust(hspace=0.5) to avoid overlapping.
7.For flattening the axes array:

8.Flatten the 2D array of subplot axes into a 1D array using axes = axes.flatten() for easier iteration.
9.For plotting histograms for each column:

10.Use a for loop to iterate through each column in the DataFrame.
11.Skip plotting the target variable (e.g., 'heart_diagnosis') if it's in the DataFrame by using a conditional statement.
12.Access the data in the column using df[column].
13.Set the color of the histogram using the color parameter when calling ax.hist().
14.Specify the number of bins for the histogram using the bins parameter when calling ax.hist().
15.Set an appropriate title for the subplot using ax.set_title().
16.Label the x and y axes with descriptions of the data using ax.set_xlabel() and ax.set_ylabel().
17.Adjust the fontsize for better legibility by setting the fontsize parameter for titles, labels, etc.
18.Show each individual plot using plt.show() within the loop.
19.For removing any empty subplots:

20.After the loop, remove any empty subplots that might remain by using fig.delaxes().
21.for displaying the subplots:

22.Finally, display the subplots by calling plt.show() at the end of your code.
# Define the number of columns and rows for subplots
num_cols = df.shape[1]  # Number of columns in the DataFrame
num_rows = (num_cols - 1) // 2 + 1  # Calculate the number of rows needed for subplots (2 plots per row)

# Create subplots
fig, axes = plt.subplots(num_rows, 4, figsize=(16, 20))
fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between subplots

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Plot histograms for each column
for i, column in enumerate(df.columns):
    if column == 'heart_diagnosis':  # Skip the target variable
        continue
    ax = axes[i]
    ax.hist(df[column], bins=25, color='skyblue', edgecolor='black')
    ax.set_title(f'{column} Distribution')
    ax.set_xlabel(column)  # Set the label for the x-axis
    ax.set_ylabel('Frequency')

# Remove any empty subplots
for i in range(len(df.columns), len(axes)):
    fig.delaxes(axes[i])

# Show the subplots
plt.show()

##  Step 4: Target column preprocessing
* heart_diagnosis: Diagnosis of heart disease (angiographic disease status) (0 = No heart disease, >0 = heart disease)."

Hints:

1. Write a lambda function that will assign 0 when heart_diagnosis is 0 and and when >=0 as 1
df['heart_diagnosis'] = df['heart_diagnosis'].apply(lambda x: 'Disease' if x == 1 else 'No Disease')

heart_diagnosis_distribution_pp = df['heart_diagnosis'].value_counts()
print("Target variable distribution after preprocessing:\n", heart_diagnosis_distribution_pp)
## Step 5:Transform the categorical features which have more than two classes
* The binary categorical columns are already good to use as it is, since they already only have 0 and 1 classes.

Hints:

1. Define the subset of columns to one-hot encode

2. Perform one-hot encoding on the selected columns and concatenate with the original DataFrame

3. Drop the original columns that were one-hot encoded)
columns_to_encode = ['cp', 'restecg', 'ca', 'thal']
df[columns_to_encode] = df[columns_to_encode].astype(str)

# Perform one-hot encoding on the specified columns
df = pd.concat([df, pd.get_dummies(df[columns_to_encode])], axis=1)

# Drop the original categorical columns
df.drop(columns=columns_to_encode, inplace=True)

# Display the column names after encoding
df.columns
# Display the top 5 rows of the dataset
df.head(10)
# Step 6: Split data into train and test sets
Hints:

Create feature matrix X by dropping the target variable ("heart_diagnosis") from the DataFrame df.

Create a target vector y by selecting the target variable ("heart_diagnosis") from the DataFrame df.

Split the data into training and test sets using the train_test_split function. Specify X as the input features, y as the target variable, set test_size to 0.2 for an 80/20 split, and use random_state for reproducibility.
# Define the features (X) and the target variable (y)
X = df.drop('heart_diagnosis', axis=1)  # Features (drop the target column)
y = df['heart_diagnosis']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Step 7: Scaling numeric features
Normalization of Feature Scales:
Feature scaling ensures that all numeric features in a dataset have similar scales or ranges. It prevents certain features from dominating the learning process due to their larger scale, making the model more balanced.

Categorical features, on the other hand, are typically non-numeric and represent categories or labels rather than continuous values. Scaling categorical features is not meaningful because the scaling process would not preserve the categorical information
Hints:

Create a StandardScaler Object:

Create an instance of the StandardScaler from scikit-learn and assign it to the scaler variable.
This step initializes the scaler, allowing you to use it for feature scaling.
Fit the Scaler to the Training Data:

Use the fit_transform method on the scaler object to fit it to the training data (X_train) and simultaneously transform (fit_transform) the training data to scale its numeric features.
This step calculates the mean and standard deviation of each numeric feature in the training data and scales the features accordingly.
Transform the Test Data:

Use the transform method on the scaler object to transform the test data (X_test) using the same scaling parameters learned from the training data.
This step ensures that the test data is scaled consistently with the training data, which is crucial for accurate model evaluation.
# Displaying the columns in the training data
X_train.columns 
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale the specified numeric features in the training set
X_train[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']] = (
    scaler.fit_transform(X_train[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']])
)

# Scale the specified numeric features in the test set
X_test[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']] = (
    scaler.transform(X_test[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']])
)

#  Skewed data
# Step 8 : Build and evaluate the base line model

Hints:

Create a logistic regression model instance by initializing LogisticRegression().

Train the logistic regression model by fitting it to the training data. Use the fit method and provide X_train (training features) and y_train (training labels) as input.

Make predictions on the test data using the trained model. Use the predict method with X_test as input to generate y_pred.
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
# Step 9: Print the metrics of model developed using skewed data
Confusion matrix visualizes the number of true positives, true negatives, false positives, and false negatives in the model's predictions.

Classification report containing key metrics like precision, recall, and F1-score, for evaluating the model's performance.

Hints:

Compute the accuracy score of the model by comparing predictions (y_pred_skewed) to true labels (y_test).

Compute the roc auc score of the model using predictions (y_pred_skewed) and true labels (y_test).
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

accuracy_skewed = accuracy_score(y_test, y_pred)
auc_roc_skewed = roc_auc_score(y_test, y_pred_proba)  # Use predicted probabilities
conf_matrix_skewed = confusion_matrix(y_test, y_pred)
class_report_skewed = classification_report(y_test, y_pred)

# Print the evaluation metrics
print('Accuracy:', accuracy_skewed)
print("AUC-ROC Score:", auc_roc_skewed)
print('Confusion Matrix:\n', conf_matrix_skewed)
print('Classification Report:\n', class_report_skewed)
# Step 9: Feature importance of model developed using skewed data
Hints:

Obtain the coefficients of the logistic regression model using the coef_ attribute and assign them to model_coefs.

Retrieve the intercept of the logistic regression model using the intercept_ attribute and assign it to model_intercept.
# After fitting the model
model_coefs = model.coef_[0]  # For binary classification, it returns an array with one row
model_intercept = model.intercept_[0]  # Intercept is a single value for binary classification

print('Coefficients:', model_coefs)
print('Intercept:', model_intercept)

###
Negative coefficients in logistic regression indicate an inverse relationship between the feature and the probability of the event (positive class).
An increase in the feature's value tends to decrease the predicted probability of the event occurring, while a decrease in the feature's value tends to increase the predicted probability.
It's common to use the absolute values of coefficients, especially when we want to focus on the overall importance of each feature without being concerned about the direction of impact.
Hints:

1.Calculate the absolute values of the coefficients and assign them to the 'Importance' column. Use X.columns for the 'Feature' column to capture the feature names.

2.Sort the importances DataFrame in descending order based on the 'Importance' column using the sort_values method. Set the ascending parameter to False to get the most important features first. 
importances = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(model.coef_[0])})
importances = importances.sort_values('Importance', ascending=False)
print('Feature Importances:\n', importances)

# Undersampled data
# Step 10: Undersample the training data
Hints:

Create a RandomUnderSampler instance and assign it correcly
Use the instance created to under sample the training data 
from imblearn.under_sampling import RandomUnderSampler

# Initialize the RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Apply undersampling
X_undersample, y_undersample = rus.fit_resample(X_train, y_train)

# Print the shapes of the undersampled and original datasets
print("X_undersample shape:", X_undersample.shape)
print("X_test shape:", X_test.shape)
print("y_undersample shape:", y_undersample.shape)
print("y_test shape:", y_test.shape)

# Step 11 : Build and evaluate the model using undersampled data
Hints:

Create a logistic regression model instance by initializing LogisticRegression().

Train the logistic regression model by fitting it to the training data. Use the fit method and provide X_undersample (training features) and y_undersample (training labels) as input.

Make predictions on the test data using the trained model. Use the predict method with X_test as input to generate y_pred_undersampled.


# Step 1: Create a logistic regression model instance
model_undersampled = LogisticRegression()

# Step 2: Train the model by fitting it to the training data
model_undersampled.fit(X_undersample, y_undersample)

# Step 3: Make predictions on the test data
y_pred_undersampled = model_undersampled.predict(X_test)

# Step 12: Print the metrics of model developed using undersampled data
Hints:

Compute the accuracy score of the model by comparing predictions (y_pred_undersampled) to true labels (y_test).

Generate a confusion matrix to visualize the number of true positives, true negatives, false positives, and false negatives in the model's predictions.

Calculate the roc_auc_score and assign it to auc_roc_undersampled
# Calculate the accuracy score
accuracy_undersampled = accuracy_score(y_test, y_pred_undersampled)

# Calculate the ROC AUC score
auc_roc_undersampled = roc_auc_score(y_test, model_undersampled.predict_proba(X_test)[:, 1])

# Generate the confusion matrix
conf_matrix_undersampled = confusion_matrix(y_test, y_pred_undersampled)

# Generate the classification report
class_report_undersampled = classification_report(y_test, y_pred_undersampled)

# Print the evaluation metrics
print('Accuracy:', accuracy_undersampled)
print('Confusion Matrix:\n', conf_matrix_undersampled)
print('Classification Report:\n', class_report_undersampled)

# Step 13: Feature importance of model developed using undersampled data
Hints:

1.Obtain the coefficients of the logistic regression model using the coef_ attribute and assign them to model_coefs.

2.Retrieve the intercept of the logistic regression model using the intercept_ attribute and assign it to model_intercept.
# Obtain the coefficients of the logistic regression model
model_undersampled_coefs = model_undersampled.coef_[0]  # Get the first row of coefficients
model_undersampled_intercept = model_undersampled.intercept_[0]  # Get the intercept

# Print the coefficients and intercept
print('Coefficients:', model_undersampled_coefs)
print('Intercept:', model_undersampled_intercept)

### 
1. Negative coefficients in logistic regression indicate an inverse relationship between the feature and the probability of the event (positive class).
2. An increase in the feature's value tends to decrease the predicted probability of the event occurring, while a decrease in the feature's value tends to increase the predicted probability.
3. It's common to use the absolute values of coefficients, especially when we want to focus on the overall importance of each feature without being concerned about the direction of impact.

Hints:

1. Calculate the absolute values of the coefficients and assign them to the 'Importance' column. Use X.columns for the    'Feature' column to capture the feature names.

2. Sort the importances DataFrame in descending order based on the 'Importance' column using the sort_values method. Set the ascending parameter to False to get the most important features first.
# Calculate feature importances
importances_undersampled = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(model_undersampled.coef_[0])})

# Sort the DataFrame by Importance in descending order
importances_undersampled = importances_undersampled.sort_values('Importance', ascending=False)

# Print the feature importances
print('Feature Importances:\n', importances_undersampled)

# Oversample data
# Step 14: Oversample the training data
Hints:

Create a SMOTE instance and assign it correcly
Use the instance created to over sample the training data
# Create a SMOTE instance
smote = SMOTE(random_state=42)

# Use the SMOTE instance to oversample the training data
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)

# Print the shapes of the oversampled data and the test data
print("X_oversample shape:", X_oversample.shape)
print("X_test shape:", X_test.shape)
print("y_oversample shape:", y_oversample.shape)
print("y_test shape:", y_test.shape)

# Step 15 : Build and evaluate the model develped using oversampled data
Hints:

Create a logistic regression model instance by initializing LogisticRegression().

Train the logistic regression model by fitting it to the training data. Use the fit method and provide X_oversample (training features) and y_oversample (training labels) as input.

Make predictions on the test data using the trained model. Use the predict method with X_test as input to generate y_pred_oversampled.
# Create a logistic regression model instance
model_oversampled = LogisticRegression()

# Train the logistic regression model with the oversampled data
model_oversampled.fit(X_oversample, y_oversample)

# Make predictions on the test data
y_pred_oversampled = model_oversampled.predict(X_test)

# Step 16: Print the metrics of the model developed using oversampled data
Hints:

Compute the accuracy score of the model by comparing predictions (y_pred_undersampled) to true labels (y_test).

Generate a confusion matrix to visualize the number of true positives, true negatives, false positives, and false negatives in the model's predictions.

Calculate the roc_auc_score and assign it to auc_roc_undersampled

Generate a classification report containing key metrics like precision, recall, and F1-score, for evaluating the model's performance.
import numpy as np

# Convert y_test to numeric values
y_test_numeric = np.where(y_test == 'No Disease', 0, 1)  # Assuming 'No Disease' is class 0 and 'Disease' is class 1
y_pred_oversampled_numeric = np.where(y_pred_oversampled == 'No Disease', 0, 1)

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Compute the accuracy score
accuracy_oversampled = accuracy_score(y_test_numeric, y_pred_oversampled_numeric)

# Calculate the AUC-ROC score
auc_roc_oversampled = roc_auc_score(y_test_numeric, y_pred_oversampled_numeric)

# Generate the confusion matrix
conf_matrix_oversampled = confusion_matrix(y_test_numeric, y_pred_oversampled_numeric)

# Generate the classification report
class_report_oversampled = classification_report(y_test_numeric, y_pred_oversampled_numeric)

# Print the metrics
print('Accuracy:', accuracy_oversampled)
print("AUC-ROC Score:", auc_roc_oversampled)
print('Confusion Matrix:\n', conf_matrix_oversampled)
print('Classification Report:\n', class_report_oversampled)


# Step 17: Feature importance of the model developed using oversampled data
Hints:

Obtain the coefficients of the logistic regression model using the coef_ attribute and assign them to model_coefs.

Retrieve the intercept of the logistic regression model using the intercept_ attribute and assign it to model_intercept.
# Get the coefficients and intercept from the logistic regression model
model_oversampled_coefs = model_oversampled.coef_[0]  # coef_ returns an array of shape (1, n_features)
model_oversampled_intercept = model_oversampled.intercept_[0]  # intercept_ returns an array, take the first element

# Print the coefficients and intercept
print('Coefficients:', model_oversampled_coefs)
print('Intercept:', model_oversampled_intercept)

###
1. Negative coefficients in logistic regression indicate an inverse relationship between the feature and the probability of the event (positive class).
2. An increase in the feature's value tends to decrease the predicted probability of the event occurring, while a decrease in the feature's value tends to increase the predicted probability.
3. It's common to use the absolute values of coefficients, especially when we want to focus on the overall importance of each feature without being concerned about the direction of impact.
Hints:
1. Calculate the absolute values of the coefficients and assign them to the 'Importance' column. Use X.columns for the 'Feature' column to capture the feature names.

2. Sort the importances DataFrame in descending order based on the 'Importance' column using the sort_values method. Set the ascending parameter to False to get the most important features first.
import pandas as pd
import numpy as np

# Create a DataFrame for importances
importances_oversampled = pd.DataFrame({
    'Feature': X.columns,                    # Use feature names from the input DataFrame
    'Importance': np.abs(model_oversampled.coef_[0])  # Get absolute values of the coefficients
})

# Sort the DataFrame in descending order based on the Importance column
importances_oversampled = importances_oversampled.sort_values('Importance', ascending=False)

# Print the feature importances
print('Feature Importances:\n', importances_oversampled)

