# src/adversarial.py

import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

def adversarial_accuracy(model, X_test, y_test, loss_fn, optimizer):
    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(X_test.shape[1],),
        nb_classes=2,
    )

    X_test = X_test.astype(np.float32)
    fgsm = FastGradientMethod(estimator=classifier, eps=0.1)
    X_test_adv = fgsm.generate(X_test)
    preds = np.argmax(classifier.predict(X_test_adv), axis=1)

    acc = np.mean(preds == y_test)
    print("Robust Accuracy under FGSM attack:", acc)
