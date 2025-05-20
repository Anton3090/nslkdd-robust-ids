# src/evaluate.py

import torch

def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        preds = torch.argmax(model(X_test_tensor), dim=1)
        acc = (preds == y_test_tensor).float().mean()
    print("Test Accuracy:", acc.item())
