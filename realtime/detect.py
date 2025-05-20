# detect.py
from scapy.all import sniff
import torch
from model import IDSModel
from utils.preprocessing import preprocess_packet
import logging

logging.basicConfig(filename="network_log.txt", level=logging.INFO)

model = IDSModel(input_dim=41, num_classes=2)  # Adjust input_dim & num_classes
model.load_state_dict(torch.load("../model.pth"))
model.eval()

def packet_callback(packet):
    features = preprocess_packet(packet)
    with torch.no_grad():
        output = model(features.float())
        pred = torch.argmax(output).item()
    label = "Attack" if pred == 1 else "Normal"
    logging.info(f"Packet: {packet.summary()}, Prediction: {label}")
    print(f"Packet detected as: {label}")

sniff(prn=packet_callback, store=False)
