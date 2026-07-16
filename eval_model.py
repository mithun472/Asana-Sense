import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

CSV_PATH = "train_angles.csv"
MODEL_PATH = "extra_trees_pose_model.pkl"
OUTPUT_DIR = "eval_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load data + model ----
df = pd.read_csv(CSV_PATH)
df = df.dropna(axis=1, how="all")

bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
feature_cols = bundle["feature_cols"]

df = df.dropna(subset=feature_cols)
X = df[feature_cols]
y = df["label"]

# Same split logic as train_model.py so val set matches what model saw
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# ==========================================================
# 1. CORRELATION HEATMAP - how joint angles relate to each other
# ==========================================================
corr = X.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)
ax.set_yticklabels(feature_cols, fontsize=7)
ax.set_title("Joint Angle Correlation Matrix")
fig.colorbar(im, ax=ax, label="Pearson correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"Saved -> {OUTPUT_DIR}/correlation_heatmap.png")

# ==========================================================
# 2. FEATURE IMPORTANCE - which angles drive the model most
# ==========================================================
importances = clf.feature_importances_
order = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(feature_cols)), importances[order], color="teal")
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels([feature_cols[i] for i in order], rotation=90, fontsize=8)
ax.set_ylabel("Importance")
ax.set_title("ExtraTrees Feature Importance (per joint angle)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print(f"Saved -> {OUTPUT_DIR}/feature_importance.png")

# ==========================================================
# 3. CONFUSION MATRIX - class-wise prediction accuracy
# ==========================================================
val_preds = clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_val, val_preds, cmap="Blues", ax=ax, xticks_rotation=45
)
ax.set_title("Confusion Matrix - Pose Classification")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"Saved -> {OUTPUT_DIR}/confusion_matrix.png")

# ==========================================================
# 4. ANGLE DISTRIBUTION PER POSE - boxplot of top 4 important angles
# ==========================================================
top4 = [feature_cols[i] for i in order[:4]]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, col in zip(axes.flatten(), top4):
    df.boxplot(column=col, by="label", ax=ax)
    ax.set_title(f"{col} distribution per pose")
    ax.set_xlabel("")
plt.suptitle("Top Angle Distributions Across Poses")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "angle_distribution_boxplots.png"), dpi=150)
plt.close()
print(f"Saved -> {OUTPUT_DIR}/angle_distribution_boxplots.png")