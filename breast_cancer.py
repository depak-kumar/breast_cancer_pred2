import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math
# Set title and description
st.title("Breast Cancer Prediction App")
st.write("Enter the values below to predict breast cancer.")
st.image('https://th.bing.com/th/id/OIP.-WiJwKni2VaRJhuPiV7XywHaFj?w=235&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7',width=400,caption='Breast Cancer Prediction ')
# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
# st.dataframe(data)
st.write('Training Data Frame : ',X.head())
st.write('Prediction Based On Feautures :',X.columns)
# st.write('')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create input widgets for features
st.sidebar.title("Input Features")
feature_names = data.feature_names
input_features = {}
for feature in feature_names:
    input_features[feature] = st.sidebar.slider(f"Select {feature}", float(X[feature].min()), float(X[feature].max()))

# Convert user input to a DataFrame
input_df = pd.DataFrame([input_features])

# Display user input
st.write("Input Features:")
st.write(input_df)
prediction = model.predict(input_df)
prediction_label = " You might be have Cancer" if prediction[0] == 0 else "You Have not a Cancerous Symptom"

st.set_option('deprecation.showPyplotGlobalUse', False)

# Visualize feature importance
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
plt.barh(data.feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
st.pyplot()


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate ROC curve and AUC
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
st.pyplot()


# Import the necessary libraries
from sklearn.metrics import confusion_matrix, classification_report

# ...

# Predict
y_pred = model.predict(X_test)

# Generate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
st.write("Confusion Matrix:")
st.write(confusion)

# Display classification report
report = classification_report(y_test, y_pred)
st.write("Classification Report:")
st.write(report)




# Predict
if st.button('Recommend '):
  st.success('Action Sucessful')
  st.write("Prediction:")
  st.success(prediction_label)


# Display prediction
# st.write("Prediction:")
# st.write(prediction_label)

# Calculate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy=round(accuracy*100)
# Display model accuracy
st.write(f"Model Accuracy: {accuracy}%  accurate \nThis project is intended for educational and informational purposes only.")

# Add a footer
st.sidebar.text("Made with ❤️ ❤️  by Your Deepak Kumar")
st.subheader("Made By Deepak KUmar")