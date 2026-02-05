import socket
import pickle
import struct
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
import msvcrt   # Windows-only

HOST = "127.0.0.1"
PORT = 5000
ROUNDS = 3
if len(sys.argv) < 2:
    print("Usage: python server.py <noise_std>")
    sys.exit(1)

NOISE_STD = float(sys.argv[1])

# ---------- Message framing ----------
def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)

def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    return pickle.loads(recvall(sock, msg_len))

# ---------- Model ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------- Setup ----------
model = Net()
test_data = torch.load("../data/test.pt", weights_only=False)
test_loader = DataLoader(test_data, batch_size=128)

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100 * correct / total

def fed_avg(weights):
    avg = {}
    for k in weights[0]:
        avg[k] = sum(w[k] for w in weights) / len(weights)
    return avg

def print_update(old_w, new_w):
    print("Averaged update stats:")
    for k in old_w:
        delta = new_w[k] - old_w[k]
        print(f"{k} mean={delta.mean():.6f} std={delta.std():.6f}")

# ---------- Server ----------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()
server.setblocking(False)

clients = []

print("Server started.")
print("Waiting for clients (press ENTER to start federated learning)...")

# Accept clients until ENTER
while True:
    try:
        conn, addr = server.accept()
        clients.append(conn)
        print(f"Client connected from {addr}")
    except BlockingIOError:
        pass

    if msvcrt.kbhit():
        if msvcrt.getch() == b'\r':
            print(f"\nStarting training with {len(clients)} clients")
            for c in clients:
                c.setblocking(True)
            break

    time.sleep(0.1)

if not clients:
    print("No clients connected. Exiting.")
    sys.exit(0)

# Send config
config = {"noise": NOISE_STD, "rounds": ROUNDS}
for c in clients:
    send_msg(c, config)

# ---------- Federated Loop ----------
for r in range(ROUNDS):
    print(f"\n--- Round {r+1} ---")

    global_weights = model.state_dict()
    for c in clients:
        send_msg(c, global_weights)

    updates = [recv_msg(c) for c in clients]

    old_weights = {k: v.clone() for k, v in global_weights.items()}
    new_weights = fed_avg(updates)

    print_update(old_weights, new_weights)

    model.load_state_dict(new_weights)
    print(f"Accuracy: {evaluate(model):.2f}%")

for c in clients:
    c.close()
server.close()
