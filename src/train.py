# train.py
import torch
from model import IDSModel
from utils.preprocessing import load_data
from torch.utils.data import DataLoader

def train():
    X_train, y_train, X_test, y_test = load_data()
    model = IDSModel(input_dim=X_train.shape[1], num_classes=len(set(y_train)))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.float())
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()
  
