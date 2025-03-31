import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import os
from torch.utils.data import Dataset, DataLoader, random_split

# Custom Dataset Class
class CodeDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # Modify based on your model input format

# Load CSV or JSON Data

def load_data(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".csv":
        df = pd.read_csv(file_path)
        data = df.to_dict(orient="records")  # Convert DataFrame to list of dicts
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    return CodeDataset(data)

# Split dataset into Train, Val, and Test
def split_data(dataset, train_ratio=0.8, val_ratio=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

# Example Mixture-of-Experts Model (Modify as Needed)
class MoEModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4):
        super(MoEModel, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gating = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        gate_scores = torch.softmax(self.gating(x), dim=1)  # Compute gating weights
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Compute expert outputs
        output = torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=1)  # Weighted sum
        return output

# Training Loop
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()  # Modify loss based on task
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)  # Modify to match model input format
            loss = criterion(output, batch)  # Modify target
            loss.backward()
            optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return model

# Save and Load Model
def save_model(model, path="moe_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="moe_model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Example Usage
if __name__ == "__main__":
    dataset = load_data("data.json")  # Change file path as needed
    train_set, val_set, test_set = split_data(dataset)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)
    
    model = MoEModel(input_dim=10, output_dim=1)  # Modify input/output dims
    model = train_model(model, train_loader, val_loader)
    save_model(model)
    
    loaded_model = MoEModel(input_dim=10, output_dim=1)
    loaded_model = load_model(loaded_model)
