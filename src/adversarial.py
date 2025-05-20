# adversarial.py
import torch
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from model import IDSModel
from utils.preprocessing import load_data

def generate_adversarial_examples():
    X_train, y_train, X_test, y_test = load_data()
    model = IDSModel(input_dim=X_train.shape[1], num_classes=len(set(y_train)))
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    classifier = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer,
                                   input_shape=(X_train.shape[1],), nb_classes=len(set(y_train)))

    fgsm = FastGradientMethod(estimator=classifier, eps=0.1)
    X_test_adv = fgsm.generate(X_test.numpy())
    print("Generated adversarial examples")

if __name__ == "__main__":
    generate_adversarial_examples()
