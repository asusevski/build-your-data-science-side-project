import pandas as pd
import lightgbm as lgb
import re
from sklearn.metrics import confusion_matrix
import joblib


df_train = pd.read_csv('./data/processed_data/train.csv')
df_test = pd.read_csv('./data/processed_data/test.csv')

X_train = df_train.drop(columns=['target'])
X_test = df_test.drop(columns=['target'])

# Necesssary for LGBM
X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

y_train = df_train[['target']]
y_test = df_test[['target']]

model = lgb.LGBMClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

# Save model:
joblib.dump(model, './models/lgb.pkl')
