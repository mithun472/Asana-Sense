import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

CSV_PATH = "train_angles.csv"
MODEL_OUT = "extra_trees_pose_model.pkl"

df = pd.read_csv(CSV_PATH)

# Drop angle columns that are ALWAYS None (face joints with no valid triplet)
df = df.dropna(axis=1, how="all")

feature_cols = [c for c in df.columns if c.startswith("angle_")]

# Drop any row still missing a value (pose not fully detected in that image)
before = len(df)
df = df.dropna(subset=feature_cols)
print(f"Dropped {before - len(df)} rows with incomplete landmarks. {len(df)} rows remain.")

X = df[feature_cols]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

clf = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

val_preds = clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

# Feature importance - which joint angles matter most
importances = sorted(
    zip(feature_cols, clf.feature_importances_), key=lambda x: x[1], reverse=True
)
print("\nTop 10 important joint angles:")
for name, score in importances[:10]:
    print(f"  {name}: {score:.4f}")

joblib.dump({"model": clf, "feature_cols": feature_cols}, MODEL_OUT)
print(f"\nModel saved -> {MODEL_OUT}")
