import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
#Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Accuracy
from sklearn.metrics import accuracy_score

#Reading
testing_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1sBu7XLI7mtBJSjxSiJTuUAkOeRzsEViF")
training_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1sAqxpGKmo7koA5L0Ohn4Mko6RUbDGSnU")

correlation_matrix = training_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

X_col = ['exercise angina', 'ST slope', 'oldpeak', 'fasting blood sugar', 'chest pain type', 'resting ecg', 'cholesterol', 'sex']

X = training_data[X_col] #Features we testing
Y = training_data["target"] #Class heart disease or not

X_test = testing_data[X_col]
Y_test = testing_data["target"]

#Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X, Y)

RF_Y_pred = rf_classifier.predict(X_test)

RF_accuracy = accuracy_score(Y_test, RF_Y_pred)
print("Random Forest Accuracy:", RF_accuracy)

#Gradient Boosting
gb_classifier = GradientBoostingClassifier(learning_rate = 0.2, random_state=42)
gb_classifier.fit(X, Y)

GB_Y_pred = gb_classifier.predict(X_test)

GB_accuracy = accuracy_score(Y_test, GB_Y_pred)
print("Gradient Boosting Accuracy:", GB_accuracy)

#Ada Boost
adaboost_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
adaboost_classifier.fit(X, Y)
AB_Y_pred = adaboost_classifier.predict(X_test)
AB_accuracy = accuracy_score(Y_test, AB_Y_pred)
print("AdaBoost Accuracy:", AB_accuracy)

# Logistic Regression
logistic_classifier = LogisticRegression(max_iter=1000, random_state=42)
logistic_classifier.fit(X, Y)
LR_Y_pred = logistic_classifier.predict(X_test)
LR_accuracy = accuracy_score(Y_test, LR_Y_pred)
print("Logistic Regression Accuracy:", LR_accuracy)

# Support Vector Machines
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X, Y)
SVM_Y_pred = svm_classifier.predict(X_test)
SVM_accuracy = accuracy_score(Y_test, SVM_Y_pred)
print("SVM Accuracy:", SVM_accuracy)

# XGBoost Classifier
xgb_classifier = XGBClassifier(n_estimators=100, random_state=42)
xgb_classifier.fit(X, Y)
XGB_Y_pred = xgb_classifier.predict(X_test)
XGB_accuracy = accuracy_score(Y_test, XGB_Y_pred)
print("XGB Accuracy:", XGB_accuracy)

data_models = ['Random Forest', 'Gradient Boosting', 'AdaBoost', 'Logistic Regression', 'Support Vector Machines', 'XGB Classifier']
accuracies = [RF_accuracy, GB_accuracy, AB_accuracy, LR_accuracy, SVM_accuracy, XGB_accuracy]
df_model_accuracy = pd.DataFrame({'Models': data_models, 'accuracy': accuracies})

fig = px.bar(df_model_accuracy, x='Models', y='accuracy', color='Models', text='accuracy', title='Comparing Models')
fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
fig.update_layout(yaxis=dict(tickformat=".2%"))
fig.show()
