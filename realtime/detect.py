# realtime/detect.py

import datetime

def log_packet(pkt, result):
    with open("log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} | {pkt.summary()} | Result: {result}\n")
