"""
Trains a PyTorch Neural Classifier for Cardiovascular Disease Risk
using clinically grounded physiological markers:
1. Age (years) [30 - 80]
2. Systolic Blood Pressure (mm Hg) [90 - 200]
3. Serum Cholesterol (mg/dL) [140 - 350]
4. Maximum Heart Rate (bpm) [80 - 210]
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CardiovascularRiskMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=6, output_dim=2):
        super(CardiovascularRiskMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def generate_clinical_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Age: 30 to 80
    age = np.random.normal(55, 12, n_samples).clip(30, 80)
    # 2. Systolic BP: 90 to 200
    bp = np.random.normal(130, 20, n_samples).clip(90, 200)
    # 3. Cholesterol: 140 to 350
    chol = np.random.normal(210, 35, n_samples).clip(140, 350)
    # 4. Max Heart Rate: 80 to 210
    max_hr = (220 - age) + np.random.normal(0, 15, n_samples)
    max_hr = max_hr.clip(80, 210)

    X_raw = np.column_stack([age, bp, chol, max_hr])

    # Clinically standard Framingham-style risk score
    # High BP, high chol, older age, lower max HR -> higher risk
    risk_score = (
        0.04 * (age - 50) +
        0.05 * (bp - 125) +
        0.03 * (chol - 200) -
        0.03 * (max_hr - 140) +
        np.random.normal(0, 0.8, n_samples)
    )
    y = (risk_score > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_raw, X_scaled, y, scaler

def train():
    os.makedirs("models", exist_ok=True)
    
    X_raw, X, y, scaler = generate_clinical_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    model = CardiovascularRiskMLP(input_dim=4, hidden_dim=6, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)

    print("--- Training Clinical Cardiovascular Model ---")
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                preds = model(X_test_t).argmax(dim=1)
                acc = (preds == y_test_t).float().mean().item()
                print(f"Epoch {epoch+1:3d}/150 | Loss: {loss.item():.4f} | Test Accuracy: {acc*100:.1f}%")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        final_acc = (preds == y_test_t).float().mean().item()
    print(f"\nFinal Test Accuracy: {final_acc * 100:.2f}%")

    W1 = model.fc1.weight.detach().numpy()
    b1 = model.fc1.bias.detach().numpy()
    W2 = model.fc2.weight.detach().numpy()
    b2 = model.fc2.bias.detach().numpy()

    weights_data = {
        "architecture": [4, 6, 2],
        "feature_names": ["Age (years)", "Systolic BP (mmHg)", "Cholesterol (mg/dL)", "Max HR (bpm)"],
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open("models/cardio_weights.json", "w") as f:
        json.dump(weights_data, f, indent=2)

    torch.save(model.state_dict(), "models/cardio_model.pt")
    print("Saved trained weights to 'models/cardio_weights.json'")

if __name__ == "__main__":
    train()
