import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import subprocess

ROUNDS = 3

# Client noise configuration
CLIENTS = {
    "client1": "0.0",
    "client2": "0.1",
    "client3": "0.5"
}

# Model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize global model
model = Net()
torch.save(model.state_dict(), "global_model.pt")

# Load test dataset
test_dataset = torch.load("../data/test.pt")
test_loader = DataLoader(test_dataset, batch_size=128)

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def fed_avg(weights):
    avg = {}
    for k in weights[0]:
        avg[k] = sum(w[k] for w in weights) / len(weights)
    return avg

def print_averaged_update(old_weights, new_weights):
    print("Averaged update (effective gradient) statistics:")
    for k in old_weights:
        delta = new_weights[k] - old_weights[k]
        print(
            f"{k}: mean={delta.mean().item():.6f}, "
            f"std={delta.std().item():.6f}"
        )

# Federated learning loop
for r in range(ROUNDS):
    print(f"\n--- Federated Round {r + 1} ---")

    # Run each client
    for client, noise in CLIENTS.items():
        subprocess.run(
            ["python", "client.py", client, noise],
            cwd=f"../{client}"
        )

    # Collect client weights
    client_weights = []
    for client in CLIENTS:
        weights = torch.load(f"../{client}/{client}_weights.pt")
        client_weights.append(weights)

    # Save old weights
    old_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # Aggregate
    new_weights = fed_avg(client_weights)

    # Print averaged update
    print_averaged_update(old_weights, new_weights)

    # Update global model
    model.load_state_dict(new_weights)
    torch.save(model.state_dict(), "global_model.pt")

    acc = evaluate(model)
    print(f"Global Model Accuracy: {acc:.2f}%")
