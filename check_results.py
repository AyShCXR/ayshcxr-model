import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

print("Loading files...")

# Load both files
preds  = pd.read_csv('val_predictions_fixed.csv')
labels = pd.read_csv('nih_demo_labels.csv')

print("=== PREDICTIONS FILE ===")
print("Shape    :", preds.shape)
print("Columns  :", preds.columns.tolist())
print(preds.head(3))

print()
print("=== LABELS FILE ===")
print("Shape    :", labels.shape)
print("Columns  :", labels.columns.tolist())
print(labels.head(3))

# ─── Merge predictions with true labels ───────────
print()
print("Merging predictions with true labels...")

labels_subset = labels.head(len(preds)).reset_index(drop=True)
preds         = preds.reset_index(drop=True)

diseases = [
    'Pneumonia',
    'Effusion',
    'Pneumothorax',
    'Infiltration'
]

print()
print("=" * 55)
print("      YOUR MODEL PERFORMANCE REPORT")
print("=" * 55)

scores = []

for disease in diseases:
    score_col = f'score_{disease}'
    label_col = disease

    if score_col in preds.columns and label_col in labels_subset.columns:

        y_true = labels_subset[label_col].values
        y_pred = preds[score_col].values

        # Remove any NaN values
        mask   = ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        try:
            auc = roc_auc_score(y_true, y_pred)
            scores.append(auc)

            # Grade the result
            if auc >= 0.75:
                grade = "✅ GOOD"
            elif auc >= 0.60:
                grade = "⚠️  FAIR"
            else:
                grade = "❌ POOR"

            bar = '█' * int(auc * 20)
            print(f"{grade}  {disease:<16} AUC = {auc:.4f}  {bar}")

        except Exception as e:
            print(f"SKIP  {disease:<16} Error: {e}")

    else:
        print(f"SKIP  {disease:<16} column not found")

# ─── Overall result ───────────────────────────────
if scores:
    mean_auc = np.mean(scores)
    print("-" * 55)
    print(f"      Mean AUC = {mean_auc:.4f}")
    print()

    if mean_auc >= 0.75:
        print("Overall → ✅ Model is performing well!")
    elif mean_auc >= 0.60:
        print("Overall → ⚠️  Model is learning but needs improvement")
    else:
        print("Overall → ❌ Model needs more training")

    print()
    print("AUC SCORE GUIDE:")
    print("  0.50 = random guessing (worst)")
    print("  0.60 = weak but learning")
    print("  0.70 = decent performance")
    print("  0.75 = good performance")
    print("  0.80 = strong performance")
    print("  0.90 = excellent (research level)")