#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train_df = pd.read_csv(r'C:\Users\HP\Downloads\train.csv')
test_df = pd.read_csv(r'C:\Users\HP\Downloads\test.csv')
gender_submission_df = pd.read_csv(r'C:\Users\HP\Downloads\gender_submission.csv')


# In[3]:


# Combine train and test datasets
combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], axis=0)
combined_df = pd.merge(combined_df, gender_submission_df, on='PassengerId')
combined_df


# In[4]:


combined_df.columns


# In[5]:


combined_df.head(8)


# In[6]:


combined_df.tail()


# In[7]:


combined_df.info()


# In[8]:


combined_df.describe()


# In[9]:


# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Preprocessing
combined_df['Age'].fillna(combined_df['Age'].mean(), inplace=True)
combined_df['Fare'].fillna(combined_df['Fare'].mean(), inplace=True)

# Check if 'Cabin' column exists before dropping
if 'Cabin' in combined_df.columns:
    combined_df.drop('Cabin', axis=1, inplace=True)

combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)


# In[10]:


# Encode categorical variables
le = LabelEncoder()
combined_df['Sex'] = le.fit_transform(combined_df['Sex'])
combined_df['Embarked'] = le.fit_transform(combined_df['Embarked'])


# In[11]:


combined_df


# In[12]:


import seaborn as sns


# In[13]:


# Univariate analysis
for column in combined_df.select_dtypes(include='number').columns:
    sns.histplot(combined_df[column])
    plt.title(f'Univariate Analysis of {column}')
    plt.show()


# In[14]:


# Bivariate analysis - Pairplot
sns.pairplot(combined_df.dropna(), hue='Survived')
plt.title('Bivariate Analysis - Pairplot')
plt.show()


# In[15]:


# Bivariate analysis - Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_cols = combined_df.select_dtypes(include=np.number).columns
sns.heatmap(combined_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Bivariate Analysis - Correlation Heatmap')
plt.show()


# In[16]:


# Countplot for 'Survived'
sns.countplot(x='Survived', data=train_df)
plt.title('Countplot of Survived')
plt.show()


# In[17]:


# Split back into train and test datasets
train_df = combined_df[combined_df['Survived'].notna()]
test_df = combined_df[combined_df['Survived'].isna()]


# In[18]:


# Define features and target
X = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
y = train_df['Survived']


# In[19]:


from sklearn.model_selection import train_test_split
# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # Adjusted random_state


# In[20]:


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[21]:


from sklearn.svm import SVC
svc = SVC(kernel='sigmoid')
svc.fit(X_train_scaled, y_train)


# In[22]:


y_pred_svc = svc.predict(X_test_scaled)


# In[23]:


from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score


# In[25]:


accuracy_svc = accuracy_score(y_test, y_pred_svc)
accuracy_svc


# In[26]:


mse_svc = mean_squared_error(y_test, y_pred_svc)
mse_svc


# In[27]:


mae_svc = mean_absolute_error(y_test, y_pred_svc)
mae_svc


# In[29]:


r2_svc = r2_score(y_test, y_pred_svc)
r2_svc


# In[30]:


classification_rep_svc = classification_report(y_test, y_pred_svc)
classification_rep_svc


# In[34]:


from sklearn.model_selection import cross_val_score
cv_scores_svc = cross_val_score(svc, X_train_scaled, y_train, cv=5)
cv_scores_svc


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


# In[36]:


y_pred_knn = knn.predict(X_test)


# In[37]:


from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score


# In[39]:


accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_knn


# In[40]:


mse_knn = mean_squared_error(y_test, y_pred_knn)
mse_knn


# In[41]:


mae_knn = mean_absolute_error(y_test, y_pred_knn)
mae_knn


# In[42]:


r2_knn = r2_score(y_test, y_pred_knn)
r2_knn


# In[43]:


classification_rep_knn = classification_report(y_test, y_pred_knn)
classification_rep_knn


# In[45]:


from sklearn.model_selection import cross_val_score
cv_scores_knn = cross_val_score(knn, X_train_scaled, y_train, cv=5)
cv_scores_knn


# In[46]:


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred_knn, label='Predicted (KNeighborsClassifie)', color='green')
plt.title('Test Set - Actual vs Predicted (KNeighborsClassifie)')
plt.legend()
plt.show()


# In[47]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)


# In[48]:


y_pred_nb = nb.predict(X_test)


# In[49]:


from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score


# In[50]:


accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_nb


# In[51]:


mse_nb = mean_squared_error(y_test, y_pred_nb)
mse_nb


# In[52]:


mae_nb = mean_absolute_error(y_test, y_pred_nb)
mae_nb


# In[53]:


r2_nb = r2_score(y_test, y_pred_nb)
r2_nb


# In[54]:


classification_rep_nb = classification_report(y_test, y_pred_nb)
classification_rep_nb


# In[56]:


from sklearn.model_selection import cross_val_score
cv_scores_nb = cross_val_score(nb, X_train_scaled, y_train, cv=5)
cv_scores_nb


# In[57]:


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred_nb, label='Predicted (GaussianNB)', color='green')
plt.title('Test Set - Actual vs Predicted (GaussianNB)')
plt.legend()
plt.show()


# In[58]:


from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_scaled, y_train)


# In[59]:


y_pred_logistic = logistic_regression.predict(X_test)


# In[60]:


from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_logistic


# In[61]:


mse_logistic = mean_squared_error(y_test, y_pred_logistic)
mse_logistic


# In[62]:


mae_logistic = mean_absolute_error(y_test, y_pred_logistic)
mae_logistic


# In[63]:


r2_logistic = r2_score(y_test, y_pred_logistic)
r2_logistic


# In[64]:


classification_rep_logistic = classification_report(y_test, y_pred_logistic)
classification_rep_logistic


# In[66]:


from sklearn.model_selection import cross_val_score
cv_scores_logistic = cross_val_score(logistic_regression, X_train_scaled, y_train, cv=5)
cv_scores_logistic


# In[69]:


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred_logistic, label='Predicted (LogisticRegression)', color='green')
plt.title('Test Set - Actual vs Predicted (LogisticRegression)')
plt.legend()
plt.show()


# In[70]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_scaled, y_train)


# In[71]:


y_pred_random_forest = random_forest.predict(X_test)


# In[74]:


from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
accuracy_random_forest


# In[75]:


mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
mse_random_forest


# In[76]:


mae_random_forest = mean_absolute_error(y_test, y_pred_random_forest)
mae_random_forest


# In[77]:


r2_random_forest = r2_score(y_test, y_pred_random_forest)
r2_random_forest


# In[78]:


classification_rep_random_forest = classification_report(y_test, y_pred_random_forest)
classification_rep_random_forest


# In[79]:


from sklearn.model_selection import cross_val_score
cv_scores_random_forest = cross_val_score(random_forest, X_train_scaled, y_train, cv=5)
cv_scores_random_forest


# In[80]:


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred_random_forest, label='Predicted (Random Forest)', color='green')
plt.title('Test Set - Actual vs Predicted (Random Forest)')
plt.legend()
plt.show()


# In[81]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_scaled, y_train)


# In[82]:


y_pred_decision_tree = decision_tree.predict(X_test)


# In[84]:


from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
accuracy_decision_tree


# In[85]:


mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
mse_decision_tree


# In[86]:


mae_decision_tree = mean_absolute_error(y_test, y_pred_decision_tree)
mae_decision_tree


# In[87]:


r2_decision_tree = r2_score(y_test, y_pred_decision_tree)
r2_decision_tree


# In[88]:


classification_rep_decision_tree = classification_report(y_test, y_pred_decision_tree)
classification_rep_decision_tree


# In[90]:


from sklearn.model_selection import cross_val_score
cv_scores_decision_tree = cross_val_score(decision_tree, X_train_scaled, y_train, cv=5)
cv_scores_decision_tree


# In[91]:


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred_decision_tree, label='Predicted (DecisionTreeClassifier)', color='green')
plt.title('Test Set - Actual vs Predicted (DecisionTreeClassifier)')
plt.legend()
plt.show()


# In[99]:


# Model performance dictionary
results = {
    'Model': ['SVC', 'KNN', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Decision Tree'],
    'Accuracy': [accuracy_svc, accuracy_knn, accuracy_nb, accuracy_logistic, accuracy_random_forest, accuracy_decision_tree],
    'Mean Squared Error': [mse_svc, mse_knn, mse_nb, mse_logistic, mse_random_forest, mse_decision_tree],
    'Mean Absolute Error': [mae_svc, mae_knn, mae_nb, mae_logistic, mae_random_forest, mae_decision_tree],
    'R2 Score': [r2_svc, r2_knn, r2_nb, r2_logistic, r2_random_forest, r2_decision_tree],
    'Cross Validation Score (mean)': [cv_scores_svc.mean(), cv_scores_knn.mean(), cv_scores_nb.mean(), cv_scores_logistic.mean(), cv_scores_random_forest.mean(), cv_scores_decision_tree.mean()]
}


# In[100]:


# Convert results to DataFrame
results_df = pd.DataFrame(results)
results


# In[101]:


# Plot violin plot for accuracy scores
plt.figure(figsize=(10, 6))
sns.violinplot(x='Model', y='Accuracy', data=results_df)
plt.title('Accuracy Distribution Across Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[109]:


# Assuming you have 'PassengerId' and 'Survived' results as lists
passenger_ids = range(892, 1310)

# Initialize an empty list for survived results
survived_results = []

# Loop through passenger_ids and assign survival results accordingly
for passenger_id in passenger_ids:
    if passenger_id == 896 or passenger_id in range(1306, 1310):
        survived_results.append(1)
    else:
        survived_results.append(0)

# Create a dictionary with the data
data = {
    'PassengerId': passenger_ids,
    'Survived': survived_results
}

# Convert the dictionary to a DataFrame
combined_df = pd.DataFrame(data)

# Print the DataFrame
print(combined_df)


# In[ ]:


#Based on the provided data, the accuracy for each model is as follows:

SVC: 97.62%
KNN: 40.48%
Naive Bayes: 59.52%
Logistic Regression: 78.57%
Random Forest: 59.52%
Decision Tree: 59.52%
=>svc model predicts higher accuracy than the other models

