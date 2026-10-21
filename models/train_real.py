"""
Trains the Cardiovascular Risk AI Model directly on the uploaded real clinical data:
`heart_4feature.csv` (Cleveland Heart Disease Dataset).
Evaluates with 5-Fold Cross Validation, trains the final production model,
and exports weights & scaling parameters for the zkML circuit.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def train_and_evaluate():
    print("=" * 70)
    print("      [+] TRAINING zkML MODEL ON REAL CLEVELAND CLINICAL DATA")
    print("=" * 70)

    # 1. Load real dataset
    df = pd.read_csv("heart_4feature.csv")
    feature_cols = ['age', 'systolic_bp', 'cholesterol', 'max_heart_rate']
    X_raw = df[feature_cols].values
    y = df['high_risk'].values

    # 2. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1s = []
    fold_aucs = []

    print("\n--- Running 5-Fold Stratified Cross Validation ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_raw[train_idx])
        X_val = scaler_fold.transform(X_raw[val_idx])
        y_tr, y_val = y[train_idx], y[val_idx]

        model_fold = CardiovascularRiskMLP(input_dim=4, hidden_dim=6, output_dim=2)
        optimizer = optim.Adam(model_fold.parameters(), lr=0.015, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y_tr)
        X_val_t = torch.FloatTensor(X_val)

        for epoch in range(120):
            optimizer.zero_grad()
            out = model_fold(X_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()

        model_fold.eval()
        with torch.no_grad():
            logits = model_fold(X_val_t)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = logits.argmax(dim=1).numpy()

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        print(f"Fold {fold}: Accuracy = {acc*100:.1f}%, F1 = {f1*100:.1f}%, ROC-AUC = {auc:.3f}")

    print(f"\nMean CV Accuracy: {np.mean(fold_accuracies)*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f}%)")
    print(f"Mean CV ROC-AUC:  {np.mean(fold_aucs):.3f}")

    # 3. Train Production Model on Full Dataset
    print("\n--- Training Production zkML Model on Full Dataset ---")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_raw)

    torch.manual_seed(42)
    np.random.seed(42)
    final_model = CardiovascularRiskMLP(input_dim=4, hidden_dim=6, output_dim=2)
    optimizer = optim.Adam(final_model.parameters(), lr=0.012, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_scaled)
    y_t = torch.LongTensor(y)

    for epoch in range(150):
        optimizer.zero_grad()
        out = final_model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    final_model.eval()
    with torch.no_grad():
        final_preds = final_model(X_t).argmax(dim=1).numpy()
        full_acc = accuracy_score(y, final_preds)
        full_f1 = f1_score(y, final_preds)
    print(f"Final Model Fit: Accuracy = {full_acc*100:.2f}%, F1-Score = {full_f1*100:.2f}%")

    # 4. Extract parameters and weights
    W1 = final_model.fc1.weight.detach().numpy()
    b1 = final_model.fc1.bias.detach().numpy()
    W2 = final_model.fc2.weight.detach().numpy()
    b2 = final_model.fc2.bias.detach().numpy()

    weights_data = {
        "architecture": [4, 6, 2],
        "dataset_source": "Cleveland Heart Disease Dataset (heart_4feature.csv)",
        "feature_names": feature_cols,
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "scaler_mean": final_scaler.mean_.tolist(),
        "scaler_scale": final_scaler.scale_.tolist(),
        "cv_accuracy_mean": round(float(np.mean(fold_accuracies)), 4),
        "cv_roc_auc_mean": round(float(np.mean(fold_aucs)), 4),
    }

    with open("models/cardio_weights.json", "w") as f:
        json.dump(weights_data, f, indent=2)

    torch.save(final_model.state_dict(), "models/cardio_model.pt")
    print("Saved production weights and scaler to 'models/cardio_weights.json'")

    # 5. Evaluate on demo_patients.json
    print("\n--- Evaluating on 'demo_patients.json' ---")
    with open("demo_patients.json") as f:
        demo = json.load(f)

    for label, patient in demo.items():
        vitals = [patient['age'], patient['systolic_bp'], patient['cholesterol'], patient['max_heart_rate']]
        vitals_scaled = final_scaler.transform([vitals])
        with torch.no_grad():
            out_logits = final_model(torch.FloatTensor(vitals_scaled))[0].numpy()
            pred_class = int(np.argmax(out_logits))
            prob = torch.softmax(torch.FloatTensor(out_logits), dim=0)[1].item()
        
        status = "HIGH_RISK" if pred_class == 1 else "LOW_RISK"
        print(f"  * {label.upper():12s}: Vitals {vitals} -> Pred: {status} (High-Risk Prob: {prob*100:.1f}%) | Logits: {out_logits.round(3).tolist()}")

if __name__ == "__main__":
    train_and_evaluate()
