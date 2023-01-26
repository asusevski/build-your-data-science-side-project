import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


df = pd.read_parquet('./data/train.parquet')
df = df.sample(n=100000, random_state=42)


def process_and_feature_engineer(df):
    df = df.fillna(-127)
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cat_variables = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    num_features = [col for col in all_cols if col not in cat_variables]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_variables].agg(['count'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    retval = pd.concat([test_num_agg, test_cat_agg], axis=1)
    
    return retval

df = process_and_feature_engineer(df)

df_labels = pd.read_csv('./data/train_labels.csv')

df_labels = pd.read_csv('./data/train_labels.csv')

# Merge
df_labels = df_labels.set_index('customer_ID')
df = df.merge(df_labels, left_index=True, right_index=True, how='left')
df.target = df.target.astype('int8')

# Get rid of customer id
df = df.reset_index(drop=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
for train_index, test_index in split.split(df, df["target"]):
    train_split = df.loc[train_index]
    test_split = df.loc[test_index]


train_split.to_csv('./data/processed_data/train.csv')
test_split.to_csv('./data/processed_data/test.csv')
