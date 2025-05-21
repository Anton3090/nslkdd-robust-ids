# 🔐 NSL-KDD Intrusion Detection System with Adversarial Robustness

A deep learning-based Intrusion Detection System (IDS) trained on the NSL-KDD dataset. This project integrates PyTorch and the Adversarial Robustness Toolbox (ART) to evaluate model robustness against adversarial attacks (FGSM) and supports real-time packet detection with Scapy.

---

## 📌 Features

- Preprocessing of NSL-KDD dataset using `pandas`, `sklearn`
- Deep neural network classifier using PyTorch
- Adversarial attack evaluation using `art.attacks.evasion.FastGradientMethod`
- Real-time packet sniffing and prediction using Scapy
- Clean modular Python code and Jupyter Notebook demonstration

---

## 📁 Project Structure

```
nslkdd-robust-ids/
├── data/               # Dataset download instructions (KaggleHub)
├── notebooks/          # kaggel notebooks for experimentation
├── src/                # Core Python modules
├── realtime/           # Real-time detection using Scapy
├── utils/              # Helper utilities
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## 📥 Dataset

This project uses the [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html).

You can download it using KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("hassan06/nslkdd")
```

---

## 🛠️ Installation

```bash
git clone https://github.com/Anton3090/nslkdd-robust-ids.git
cd nslkdd-robust-ids
pip install -r requirements.txt
```

### Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
adversarial-robustness-toolbox
kagglehub
scapy
```

---

## 🧠 Model Architecture

```python
nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
```

---

## 🧪 Usage

### 1. Preprocess and Train

```bash
python src/train.py
```

### 2. Evaluate on Clean Data

```bash
python src/evaluate.py
```

### 3. Evaluate Robustness

```bash
python src/adversarial.py
```

---

## 🔴 Real-Time Packet Detection (Scapy)

Detect potential intrusions in real-time from live network packets.

```bash
sudo python realtime/detect.py
```

Inside `realtime/detect.py`, we:

- Use Scapy to sniff live packets
- Extract basic features (protocol, ports, packet length, flags)
- Normalize and reshape into model input format
- Predict using the trained PyTorch model

Sample logic:

```python
from scapy.all import sniff, IP, TCP, UDP
import torch
import torch.nn as nn
import numpy as np
import joblib
from datetime import datetime

# Define the model
class IDSModel(nn.Module):
    def __init__(self, input_dim):
        super(IDSModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.layers(x)

# Load model and scaler
model = IDSModel(input_dim=42)
model.load_state_dict(torch.load("ids_model.pth"))
model.eval()

scaler = joblib.load("scaler.save")

# Feature extraction function
def extract_features(pkt):
    try:
        length = len(pkt)
        ttl = pkt[IP].ttl if IP in pkt else 0
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        features = [length, ttl, dport]
        features += [0] * (42 - len(features))  # Pad to 42 features
        return np.array(features).reshape(1, -1)
    except:
        return np.zeros((1, 42))  # Return dummy on failure

# Classify and log packet
def classify_packet(pkt):
    features = extract_features(pkt)
    scaled = scaler.transform(features)
    tensor = torch.tensor(scaled, dtype=torch.float32)
    output = model(tensor)
    pred = torch.argmax(output).item()
    label = "attack" if pred == 1 else "normal"

    print(f"[{datetime.now()}] Packet classified as: {label}")
    with open("log.txt", "a") as f:
        f.write(f"{datetime.now()} | {pkt.summary()} | Result: {label}\n")

# Start sniffing for 10 seconds
print("Sniffing packets for 10 seconds...")
sniff(prn=classify_packet, timeout=10, store=0)
print("Sniffing finished.")

```

> Note: Ensure features extracted match the trained model’s input structure.

---

## 📊 Results

| Dataset       | Accuracy |
|---------------|----------|
| Clean Test    | 86.7%    |
| FGSM Attack   | 81.9%    |

---

## 🔒 Adversarial Robustness

This project uses **Fast Gradient Sign Method (FGSM)** from [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox):

```python
from art.attacks.evasion import FastGradientMethod
fgsm = FastGradientMethod(estimator=classifier, eps=0.1)
X_test_adv = fgsm.generate(X_test.astype(np.float32))
```

---

## 📌 Credits

- NSL-KDD dataset by Canadian Institute for Cybersecurity
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [Scapy](https://scapy.net/) for real-time packet sniffing
- KaggleHub for dataset access

---

## 📃 License

This project is licensed under the MIT License.

---

## 🤝 Contributions

Feel free to fork the repo and submit pull requests. Issues and improvements welcome!
