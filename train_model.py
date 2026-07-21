import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

CSV_PATH = "train_angles.csv"
MODEL_OUT = "extra_trees_pose_model.pkl"

# Floor for per-joint std so a joint that happens to be perfectly rigid
# in the training images (std == 0) doesn't become an impossible-to-pass
# tolerance at inference time.
MIN_JOINT_STD_DEG = 5.0

# Percentile of the TRUE class's own predicted probability (on the
# validation split) used as that class's live confidence threshold.
# Using a per-class value instead of one flat number is what fixes
# poses (e.g. plank) whose angle profile is naturally less distinctive
# and therefore produces lower confidence even when correctly predicted.
CONF_PERCENTILE = 10
DEFAULT_CONF_THRESH = 55.0  # fallback if a class has too few val samples

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

# ---------------------------------------------------------------------
# Per-class reference angles: mean + std of EVERY joint angle, learned
# from the training images. class_means feeds the "how far off is this
# joint" diff in live_combined.py; class_stds lets that diff be judged
# per-joint instead of with one blanket degree threshold, since some
# joints (e.g. a supporting leg) are naturally much more rigid than
# others (e.g. a raised arm) even within the same correct pose.
# ---------------------------------------------------------------------
class_means = {}
class_stds = {}
for label, group in df.groupby("label"):
    means = group[feature_cols].mean().tolist()
    stds = group[feature_cols].std(ddof=0).fillna(0.0).tolist()
    stds = [max(s, MIN_JOINT_STD_DEG) for s in stds]
    class_means[label] = means
    class_stds[label] = stds

# ---------------------------------------------------------------------
# Per-class confidence threshold. Poses with less distinctive joint
# angles (e.g. plank, where most joints are simply straight, close to
# a neutral/resting profile) will legitimately get lower max-probability
# scores from the classifier than a highly distinctive pose like tree,
# even when the prediction is correct. A single global CONF_THRESH
# therefore under-fires for some classes and over-fires for others.
# We instead look, per class, at how confident the model was on
# validation rows that truly belong to that class and pick a low
# percentile of that as the live threshold.
# ---------------------------------------------------------------------
val_proba = clf.predict_proba(X_val)
class_index = {c: i for i, c in enumerate(clf.classes_)}

class_thresholds = {}
y_val_arr = y_val.reset_index(drop=True)
for label in clf.classes_:
    mask = (y_val_arr == label).values
    n = mask.sum()
    if n < 5:
        # not enough validation rows for this class to trust a percentile
        class_thresholds[label] = DEFAULT_CONF_THRESH
        continue
    true_class_proba = val_proba[mask, class_index[label]] * 100
    thresh = float(np.percentile(true_class_proba, CONF_PERCENTILE))
    # never go below a sane floor, so a genuinely ambiguous class
    # doesn't end up accepting near-random guesses
    class_thresholds[label] = max(thresh, 30.0)

print("\nPer-class confidence thresholds (live-detection cutoff):")
for label, t in class_thresholds.items():
    print(f"  {label}: {t:.1f}%")

# ---------------------------------------------------------------------
# Per-image angle diff vs class mean. For every training image, compute
# abs(angle - class_mean) per joint. Same math live_combined.py uses at
# inference, run here over the fed dataset so diff distribution can be
# eyeballed / used for eval (e.g. sanity check MIN_JOINT_STD_DEG, spot
# mislabeled images with abnormally huge diffs).
# ---------------------------------------------------------------------
diff_rows = []
for _, row in df.iterrows():
    label = row["label"]
    means = class_means[label]
    diffs = {"label": label}
    for col, mean_val in zip(feature_cols, means):
        diffs[col] = abs(row[col] - mean_val)
    diff_rows.append(diffs)

diff_df = pd.DataFrame(diff_rows)
diff_df.to_csv("train_angle_diffs.csv", index=False)
print(f"\nPer-image angle diffs -> train_angle_diffs.csv ({len(diff_df)} rows)")

print("\nMean abs diff per joint, per class:")
print(diff_df.groupby("label")[feature_cols].mean().round(1))

joblib.dump(
    {
        "model": clf,
        "feature_cols": feature_cols,
        "class_means": class_means,
        "class_stds": class_stds,
        "class_thresholds": class_thresholds,
    },
    MODEL_OUT,
)
print(f"\nModel saved -> {MODEL_OUT}")