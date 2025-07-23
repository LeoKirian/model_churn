# %%

import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine import encoding
from sklearn import impute
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score, 
                            recall_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
# %%
df = pd.read_csv('customer-churn.csv')
df.drop('customerID', axis=1, inplace=True)
df.head()
# %%
df['Churn'] = df['Churn'].map({"Yes":1, 'No':0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.isna().sum()
# %%
catg_vars = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'SeniorCitizen']

num_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_vars].describe()
# %%
X = df.drop('Churn', axis=1)
y = df['Churn']
y
# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_test.mean()

# %%
numerical_transform = Pipeline(
    steps=[('imputer', impute.SimpleImputer(strategy='median'))])

categorical_transform = Pipeline(
    steps=[("encoder", encoding.OneHotEncoder())]
)

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transform , num_vars),
    ('cat', categorical_transform, catg_vars)]
)

model = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, 
                               class_weight='balanced_subsample', random_state=42)
# %%
pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
])

pipeline.fit(X_train, y_train)

# %%
y_pred = pipeline.predict(X_test)
y_train_predict = pipeline.predict(X_train)
y_pred_proba = pipeline.predict_proba(X_test)[:,1]
y_train_predict_proba = pipeline.predict_proba(X_train)[:,1]
y_pred_proba
# %%

accuracy_test = accuracy_score(y_test,y_pred)
AUC_test = roc_auc_score(y_test, y_pred)
roc_test = roc_curve(y_test, y_pred_proba)

accuracy_train = accuracy_score(y_train,y_train_predict)
AUC_train = roc_auc_score(y_train, y_train_predict)
roc_train = roc_curve(y_train, y_train_predict_proba)
print(f'Acurácia Train: {accuracy_train:.2f}')
print(f'AUC Train: {AUC_train:.2f}')

print(f'Acurácia Teste: {accuracy_test:.2f}')
print(f'AUC Teste: {AUC_test:.2f}')
# %%
#palette = ['#00AFBC', '#EDC033', '#E52767', '#61B530', '#157C8A', '#4AC5BB', '#DFBFBF']
plt.plot(roc_train[0], roc_train[1], color = '#00AFBC')
plt.plot(roc_test[0], roc_test[1], color = '#E52767')
plt.grid(True)
plt.legend([
    f'AUC Treino: {100*AUC_train:.2f}',
    f'AUC Teste: {100*AUC_test:.2f}'])
plt.xlabel('1-Especificidade')
plt.ylabel('Sensibilidade')
plt.title('Curva ROC')
# %%
cm = confusion_matrix(y_test, y_pred)
label = model.classes_
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
disp.plot()
# %%
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
