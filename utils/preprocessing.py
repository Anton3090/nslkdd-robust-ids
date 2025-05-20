# preprocessing.py
import pandas as pd
import torch

def load_data():
    # Example loading NSL-KDD dataset CSV files (adjust path accordingly)
    train_df = pd.read_csv("data/KDDTrain.csv")
    test_df = pd.read_csv("data/KDDTest.csv")

    # Basic preprocessing: convert categorical to numeric, normalize, etc.
    # (Implement your own preprocessing pipeline here)

    X_train = torch.tensor(train_df.iloc[:, :-1].values).float()
    y_train = torch.tensor(train_df.iloc[:, -1].values).long()
    X_test = torch.tensor(test_df.iloc[:, :-1].values).float()
    y_test = torch.tensor(test_df.iloc[:, -1].values).long()
    return X_train, y_train, X_test, y_test

def preprocess_packet(packet):
    # Convert packet to features vector for the model
    # Dummy placeholder: replace with actual packet parsing
    features = torch.zeros(41)  # 41 features assumed
    return features
