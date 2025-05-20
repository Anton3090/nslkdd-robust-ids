# evaluate.py
import torch
from sklearn.metrics import accuracy_score
from model import IDSModel
from utils.preprocessing import load_data

def evaluate():
    X_train, y_train, X_test, y_test = load_data()
    model = IDSModel(input_dim=X_train.shape[1], num_classes=len(set(y_train)))
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test.float())
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
