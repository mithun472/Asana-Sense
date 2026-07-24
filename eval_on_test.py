import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TEST_CSV = "test_angles.csv"        # from extract_angles.py run on dataset/test
MODEL_PATH = "extra_trees_pose_model.pkl"

# Run this ONCE per finalized model version. Do not tune anything based
# on these numbers - if results are bad, go fix train data/features and
# retrain, then come back and run this again as a fresh, single check.

bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
feature_cols = bundle["feature_cols"]

df = pd.read_csv(TEST_CSV)
before = len(df)
df = df.dropna(subset=feature_cols)
print(f"Test set: {len(df)} usable rows ({before - len(df)} dropped, no detection).")

X_test = df[feature_cols]
y_test = df["label"]

preds = clf.predict(X_test)

print("\n=== HELD-OUT TEST ACCURACY (real generalization number) ===")
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Oranges", ax=ax, xticks_rotation=45)
ax.set_title("Confusion Matrix - HELD-OUT TEST SET")
plt.tight_layout()
plt.savefig("test_confusion_matrix.png", dpi=150)
print("\nSaved -> test_confusion_matrix.png")
