import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
!pip install streamlit
import streamlit as st
import pickle
# Load training and testing datasets
train_data = pd.read_csv('/content/Titanic_train.csv')
test_data = pd.read_csv('/content/Titanic_test.csv')
# Display basic information
print("Training Data Info")
print(train_data.info())
print("\nTesting Data Info")
print(test_data.info())
# Display summary statistics
print("\nTraining Data Summary")
print(train_data.describe())
# Visualizations - Histograms
train_data.hist(bins=20, figsize=(10, 10))
plt.suptitle('Training Data - Histograms')
plt.show()
# Boxplot to detect outliers
plt.figure(figsize=(10,8))
sns.boxplot(data=train_data)
plt.title('Training Data - Boxplot')
plt.show()
# Correlation heatmap
plt.figure(figsize=(10,8))
# Select only numerical features for correlation calculation
numerical_data = train_data.select_dtypes(include=['number'])
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Training Data - Correlation Heatmap')
plt.show()
# Handle missing values (fill with mean or drop based on analysis)
# Select only numerical features for mean calculation
numerical_data = train_data.select_dtypes(include=['number'])

# Calculate the mean of numerical features
numerical_means = numerical_data.mean()

# Fill missing values in numerical features with their respective means
train_data[numerical_data.columns] = train_data[numerical_data.columns].fillna(numerical_means)

# Repeat for test_data
numerical_data_test = test_data.select_dtypes(include=['number'])
numerical_means_test = numerical_data_test.mean()
test_data[numerical_data_test.columns] = test_data[numerical_data_test.columns].fillna(numerical_means_test)
# Encode categorical variables (One-hot encoding)
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)
# Align columns in test and train datasets (for consistent model input)
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)
# Split training data into X (features) and y (target)
target_column_name = 'Survived'
X_train = train_data.drop(target_column_name, axis=1)
y_train = train_data[target_column_name]
# Prepare test data without the target variable
X_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']
# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Predict on the test dataset
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]
# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
import numpy as np

# Check the unique classes and their counts in y_test
unique_classes, class_counts = np.unique(y_test, return_counts=True)
print("Unique Classes:", unique_classes)
print("Class Counts:", class_counts)
# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
# Interpret model coefficients
coefficients = pd.DataFrame(model.coef_, columns=X_train.columns).T
coefficients.columns = ['Coefficient']
print(coefficients.sort_values(by='Coefficient', ascending=False))
import pickle
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
filename = 'titanic_model.pkl'
pickle.dump(model, open(filename, 'wb'))
!pip install streamlit
import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load the trained model
loaded_model = pickle.load(open('titanic_model.pkl', 'rb'))

# Create a Streamlit app
st.title("Titanic Survival Prediction")

# User input features
st.header("Enter Passenger Information:")
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked_s = st.selectbox("Embarked (Southampton)", [0, 1])

# Create a feature vector from user input
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex_male': [1 if sex == 'male' else 0],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_Q': [0],  # Assuming default is not Q
    'Embarked_S': [embarked_s]
})

# Make a prediction
if st.button("Predict"):
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:
        st.write("The passenger is predicted to survive.")
    else:
        st.write("The passenger is predicted to not survive.")

