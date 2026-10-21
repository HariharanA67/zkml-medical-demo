"""
Improved zkML Cardiovascular Diagnostic Model Training.
Trained on the canonical unified dataset: `data/heart_master_4feature.csv` (744 patients).
Features:
- Architecture: [4 -> 8 -> 2] (higher capacity, calibrated for circuit efficiency)
- 10-Fold Stratified Cross-Validation
- Robust Standard Scaling
- Model checkpointing & weights export for BN254 arithmetic circuit
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

class ImprovedCardioMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super(ImprovedCardioMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_and_benchmark():
    print("=" * 75)
    print("     [+] TRAINING IMPROVED MODEL ON UNIFIED DATASET (744 PATIENTS)")
    print("=" * 75)

    # 1. Load canonical unified dataset
    df = pd.read_csv("data/heart_master_4feature.csv")
    feature_cols = ['age', 'systolic_bp', 'cholesterol', 'max_heart_rate']
    X_raw = df[feature_cols].values
    y = df['high_risk'].values

    print(f"Dataset Size: {len(df)} samples | Features: {feature_cols}")
    print(f"Class Balance: Healthy={np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%), At-Risk={np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")

    # 2. 10-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_aucs = []

    print("\n--- Running 10-Fold Stratified Cross-Validation ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_raw[train_idx])
        X_val = scaler_fold.transform(X_raw[val_idx])
        y_tr, y_val = y[train_idx], y[val_idx]

        model_fold = ImprovedCardioMLP(input_dim=4, hidden_dim=8, output_dim=2)
        optimizer = optim.AdamW(model_fold.parameters(), lr=0.012, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y_tr)
        X_val_t = torch.FloatTensor(X_val)

        for epoch in range(140):
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
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_aucs.append(auc)

    print(f"CV Accuracy:  {np.mean(fold_accuracies)*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f}%)")
    print(f"CV Precision: {np.mean(fold_precisions)*100:.2f}%")
    print(f"CV Recall:    {np.mean(fold_recalls)*100:.2f}%")
    print(f"CV F1-Score:  {np.mean(fold_f1s)*100:.2f}%")
    print(f"CV ROC-AUC:   {np.mean(fold_aucs):.4f}")

    # 3. Train Production Model on Full Master Dataset
    print("\n--- Training Production zkML Model on Full Cohort ---")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_raw)

    torch.manual_seed(42)
    np.random.seed(42)
    production_model = ImprovedCardioMLP(input_dim=4, hidden_dim=8, output_dim=2)
    optimizer = optim.AdamW(production_model.parameters(), lr=0.012, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_scaled)
    y_t = torch.LongTensor(y)

    for epoch in range(160):
        optimizer.zero_grad()
        out = production_model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    production_model.eval()
    with torch.no_grad():
        final_preds = production_model(X_t).argmax(dim=1).numpy()
        full_acc = accuracy_score(y, final_preds)
        full_f1 = f1_score(y, final_preds)
        full_prec = precision_score(y, final_preds)
        full_rec = recall_score(y, final_preds)
        full_auc = roc_auc_score(y, torch.softmax(production_model(X_t), dim=1)[:, 1].numpy())

    print(f"Production Model Fit (744 Patients):")
    print(f" - Accuracy:  {full_acc*100:.2f}%")
    print(f" - ROC-AUC:   {full_auc:.4f}")
    print(f" - Precision: {full_prec*100:.2f}%")
    print(f" - Recall:    {full_rec*100:.2f}%")
    print(f" - F1-Score:  {full_f1*100:.2f}%")

    # 4. Export weights & scaler for the zkML circuit
    W1 = production_model.fc1.weight.detach().numpy()
    b1 = production_model.fc1.bias.detach().numpy()
    W2 = production_model.fc2.weight.detach().numpy()
    b2 = production_model.fc2.bias.detach().numpy()

    weights_payload = {
        "model_version": "2.0-improved-multicenter",
        "training_dataset": "data/heart_master_4feature.csv",
        "cohort_size": int(len(df)),
        "architecture": [4, 8, 2],
        "feature_names": feature_cols,
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "scaler_mean": final_scaler.mean_.tolist(),
        "scaler_scale": final_scaler.scale_.tolist(),
        "metrics": {
            "cv_accuracy_mean": round(float(np.mean(fold_accuracies)), 4),
            "cv_roc_auc_mean": round(float(np.mean(fold_aucs)), 4),
            "cv_f1_mean": round(float(np.mean(fold_f1s)), 4),
            "full_fit_accuracy": round(float(full_acc), 4),
            "full_fit_auc": round(float(full_auc), 4),
        }
    }

    with open("models/cardio_weights.json", "w") as f:
        json.dump(weights_payload, f, indent=2)

    torch.save(production_model.state_dict(), "models/cardio_model.pt")
    print("\nSaved production weights and circuit parameters to 'models/cardio_weights.json'")

    # 5. Evaluate on all profiles in `data/demo_patients_master.json`
    print("\n--- Evaluating on Master Demo Patients ('data/demo_patients_master.json') ---")
    with open("data/demo_patients_master.json") as f:
        demo = json.load(f)

    for key, p in demo.items():
        vitals = [p['age'], p['systolic_bp'], p['cholesterol'], p['max_heart_rate']]
        vitals_scaled = final_scaler.transform([vitals])
        with torch.no_grad():
            out_logits = production_model(torch.FloatTensor(vitals_scaled))[0].numpy()
            pred_class = int(np.argmax(out_logits))
            prob = torch.softmax(torch.FloatTensor(out_logits), dim=0)[1].item()
        
        status = "HIGH_RISK_ANOMALY" if pred_class == 1 else "LOW_RISK_HEALTHY"
        print(f"  * {p['name']:48s}: Pred = {status:18s} (Risk Prob = {prob*100:.1f}%) | Expected: {p['expected']}")

if __name__ == "__main__":
    train_and_benchmark()
