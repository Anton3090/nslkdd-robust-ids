# 🔐 NSL-KDD Intrusion Detection System with Adversarial Robustness

A deep learning-based Intrusion Detection System (IDS) trained on the NSL-KDD dataset. This project integrates PyTorch and the Adversarial Robustness Toolbox (ART) to evaluate model robustness against adversarial attacks (e.g., FGSM) and supports real-time packet detection with Scapy.

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
├── notebooks/          # Jupyter notebooks for experimentation
├── src/                # Core Python modules (training, model, attack, etc.)
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
from scapy.all import sniff
from src.model import load_model
from utils.preprocessing import preprocess_packet

model = load_model("Model/ids_model.pth")

def predict_packet(packet):
    features = preprocess_packet(packet)
    pred = model(torch.tensor(features).float().unsqueeze(0))
    print("ALERT 🚨" if pred.argmax().item() == 1 else "Normal ✅")

sniff(prn=predict_packet, store=0)
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
