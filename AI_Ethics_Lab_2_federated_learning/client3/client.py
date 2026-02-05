import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# Arguments
CLIENT_ID = sys.argv[1]          # client1 / client2 / client3
NOISE_STD = float(sys.argv[2])   # e.g. 0.0, 0.1, 0.5, 1.0

DATA_PATH = f"../data/{CLIENT_ID}.pt"
MODEL_PATH = "../server/global_model.pt"
OUT_PATH = f"{CLIENT_ID}_weights.pt"

LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

# Simple MLP model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load dataset
dataset = torch.load(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load global model
model = Net()
model.load_state_dict(torch.load(MODEL_PATH))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# Local training
model.train()
for _ in range(LOCAL_EPOCHS):
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

# Add Gaussian noise to weights (local DP)
noisy_weights = {}
for k, v in model.state_dict().items():
    if NOISE_STD > 0:
        noise = torch.normal(0, NOISE_STD, size=v.shape)
        noisy_weights[k] = v + noise
    else:
        noisy_weights[k] = v.clone()

torch.save(noisy_weights, OUT_PATH)
print(f"{CLIENT_ID}: sent weights with noise std = {NOISE_STD}")
