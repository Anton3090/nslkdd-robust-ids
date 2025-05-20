# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_nsl_kdd_dataset(path):
    columns = [ ... ]  # same 43 columns as your code
    train_df = pd.read_csv(f"{path}/KDDTrain+.txt", header=None, names=columns)
    test_df = pd.read_csv(f"{path}/KDDTest+.txt", header=None, names=columns)

    # Binary classification
    train_df['label'] = train_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    test_df['label'] = test_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # Encode categorical
    cat_cols = ['protocol_type', 'service', 'flag']
    encoder = LabelEncoder()
    for col in cat_cols:
        train_df[col] = encoder.fit_transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

    # Separate features and labels
    X_train = train_df.drop(['label'], axis=1)
    y_train = LabelEncoder().fit_transform(train_df['label'])

    X_test = test_df.drop(['label'], axis=1)
    y_test = LabelEncoder().fit_transform(test_df['label'])

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
