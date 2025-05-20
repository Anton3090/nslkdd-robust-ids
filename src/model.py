# model.py
import torch.nn as nn

class IDSModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IDSModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
