import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# ---------------- Message framing helpers ----------------
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

# ---------------- Client identity ----------------
CLIENT_ID = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

HOST = "127.0.0.1"
PORT = 5000

# ---------------- Model ----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------------- Setup ----------------
DATA_DIR = './'
dataset = torch.load(
    os.path.join(DATA_DIR, f"{CLIENT_ID}.pt"),
    weights_only=False
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

# Receive configuration
config = recv_msg(client)
NOISE_STD = config["noise"]
ROUNDS = config["rounds"]

print(f"{CLIENT_ID} connected | Noise std = {NOISE_STD}")

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---------------- Federated Learning Loop ----------------
for _ in range(ROUNDS):
    global_weights = recv_msg(client)
    model.load_state_dict(global_weights)

    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # Add noise locally
    noisy_weights = {}
    for k, v in model.state_dict().items():
        noise = torch.normal(0, NOISE_STD, size=v.shape)
        noisy_weights[k] = v + noise

    send_msg(client, noisy_weights)

client.close()
