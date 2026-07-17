import time
import cv2
import mediapipe as mp
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from angle_utils import calc_all_angles, JOINT_TRIPLETS

MODEL_PATH = "pose_landmarker_full.task"
CLASSIFIER_PATH = "extra_trees_pose_model.pkl"

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

# Joints worth labeling on screen (main limb joints, skip face clutter)
DISPLAY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

JOINT_NAMES = {
    11: "left shoulder", 12: "right shoulder",
    13: "left elbow", 14: "right elbow",
    15: "left wrist", 16: "right wrist",
    23: "left hip", 24: "right hip",
    25: "left knee", 26: "right knee",
    27: "left ankle", 28: "right ankle",
}

# Load trained classifier + exact feature column order it was trained on
bundle = joblib.load(CLASSIFIER_PATH)
clf = bundle["model"]
feature_cols = bundle["feature_cols"]
class_means = bundle["class_means"]        # {label: [mean_angle_per_feature_col]}
class_stds = bundle["class_stds"]          # {label: [std_angle_per_feature_col]}
class_thresholds = bundle["class_thresholds"]  # {label: confidence_pct_cutoff}
joint_ids = [int(c.split("_")[1]) for c in feature_cols]

DEFAULT_CONF_THRESH = 55.0   # fallback for a class with no learned threshold
STD_MULTIPLIER = 1.5         # how many stds off before a joint is "wrong"
MAX_FLAGGED_JOINTS = 3       # how many worst joints to name in the label

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.8,
    min_pose_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not activate camera device index 0.")

start_time_ms = int(time.time() * 1000)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    print("Live angle + classifier engine active. Press 'q' to stop.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_time_ms = int(time.time() * 1000)
        timestamp_ms = current_time_ms - start_time_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        label_text = "No pose detected"

        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:

                # skeleton lines
                for a, b in POSE_CONNECTIONS:
                    if a < len(pose_landmarks) and b < len(pose_landmarks):
                        pt1 = pose_landmarks[a]
                        pt2 = pose_landmarks[b]
                        x1, y1 = int(pt1.x * w), int(pt1.y * h)
                        x2, y2 = int(pt2.x * w), int(pt2.y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # full 33-joint angle calc (used for both display + prediction)
                angles = calc_all_angles(pose_landmarks)

                # build feature vector in exact training order, predict pose
                feature_vec = [angles[j] for j in joint_ids]

                # per-joint diff (populated below once we know the predicted
                # class) drives both the joint dot color and the on-screen
                # "fix: ..." callouts.
                joint_status = {}  # landmark_id -> "ok" | "off" | None (n/a)

                if all(v is not None for v in feature_vec):
                    pred = clf.predict([feature_vec])[0]
                    proba = clf.predict_proba([feature_vec])[0]
                    confidence = max(proba) * 100

                    conf_thresh = class_thresholds.get(pred, DEFAULT_CONF_THRESH)

                    if confidence < conf_thresh:
                        label_text = f"neutral ({confidence:.1f}%, needs {conf_thresh:.0f}%)"
                    else:
                        ref_vec = class_means[pred]
                        std_vec = class_stds.get(pred, [15.0] * len(ref_vec))

                        # per-joint diff, normalized by that joint's natural
                        # spread for this pose, so a joint that's always
                        # rigid in this pose gets a tight tolerance and a
                        # joint that naturally varies gets more slack.
                        per_joint_diff = []
                        for jid, feat_val, ref_val, std_val in zip(joint_ids, feature_vec, ref_vec, std_vec):
                            diff = abs(feat_val - ref_val)
                            is_off = diff > (STD_MULTIPLIER * std_val)
                            joint_status[jid] = "off" if is_off else "ok"
                            per_joint_diff.append((jid, diff, is_off))

                        overall_diff = sum(d for _, d, _ in per_joint_diff) / len(per_joint_diff)
                        flagged = sorted(
                            (item for item in per_joint_diff if item[2]),
                            key=lambda item: item[1], reverse=True
                        )[:MAX_FLAGGED_JOINTS]

                        if not flagged:
                            label_text = f"{pred} ({confidence:.1f}%) - correct"
                        else:
                            names = ", ".join(
                                f"{JOINT_NAMES.get(jid, jid)} ({d:.0f} deg off)"
                                for jid, d, _ in flagged
                            )
                            label_text = f"{pred} ({confidence:.1f}%) - fix: {names}"
                else:
                    label_text = "Incomplete landmarks"

                # joint dots, color-coded by correction status when available
                for i, lm in enumerate(pose_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    status = joint_status.get(i)
                    if status == "off":
                        color = (0, 0, 255)     # red = needs correction
                    elif status == "ok":
                        color = (0, 220, 0)      # green = within tolerance
                    else:
                        color = (0, 0, 255)      # default (untracked joint)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

                # draw angle numbers on key limb joints
                for j in DISPLAY_JOINTS:
                    if j not in JOINT_TRIPLETS:
                        continue
                    ang = angles[j]
                    if ang is None:
                        continue
                    lm = pose_landmarks[j]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(
                        frame, f"{int(ang)}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                    )

        cv2.putText(
            frame, label_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )

        cv2.imshow("Live Pose Angles + Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()