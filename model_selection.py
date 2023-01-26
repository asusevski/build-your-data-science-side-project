import pandas as pd
from lazypredict.Supervised import LazyClassifier
import lightgbm as lgb
import xgboost as xgb
import re

df_train = pd.read_csv('./data/processed_data/train.csv')
df_test = pd.read_csv('./data/processed_data/test.csv')

X_train = df_train.drop(columns=['target'])
X_test = df_test.drop(columns=['target'])

y_train = df_train[['target']]
y_test = df_test[['target']]

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None)

models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)


# Amex Metric
def amex_metric(y_true, y_pred) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()
    
    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

clf1 = lgb.LGBMClassifier()
clf2 = xgb.XGBClassifier()

# Necessary because LGB does not support special characters in feature names
X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred1 = pd.DataFrame(clf1.predict(X_test)).rename(columns={0: 'prediction'})
y_pred2 = pd.DataFrame(clf2.predict(X_test)).rename(columns={0: 'prediction'})

print(f"LGB metric: {amex_metric(y_test, y_pred1)}")
print(f"XGB metric: {amex_metric(y_test, y_pred2)}")
