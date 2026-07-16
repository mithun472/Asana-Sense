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

# Load trained classifier + exact feature column order it was trained on
bundle = joblib.load(CLASSIFIER_PATH)
clf = bundle["model"]
feature_cols = bundle["feature_cols"]
class_means = bundle["class_means"]  # {label: [mean_angle_per_feature_col]}
joint_ids = [int(c.split("_")[1]) for c in feature_cols]

CONF_THRESH = 85.0        # below this -> neutral/idle
ANGLE_DIFF_THRESH = 15.0  # mean abs angle diff vs ref -> correct pose cutoff

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

                # joint dots
                for lm in pose_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # full 33-joint angle calc (used for both display + prediction)
                angles = calc_all_angles(pose_landmarks)

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

                # build feature vector in exact training order, predict pose
                feature_vec = [angles[j] for j in joint_ids]
                if all(v is not None for v in feature_vec):
                    pred = clf.predict([feature_vec])[0]
                    proba = clf.predict_proba([feature_vec])[0]
                    confidence = max(proba) * 100

                    if confidence < CONF_THRESH:
                        label_text = f"neutral ({confidence:.1f}%)"
                    else:
                        ref_vec = class_means[pred]
                        diff = sum(abs(a - b) for a, b in zip(feature_vec, ref_vec)) / len(ref_vec)
                        status = "correct" if diff <= ANGLE_DIFF_THRESH else "adjust pose"
                        label_text = f"{pred} ({confidence:.1f}%) - {status}, diff {diff:.1f}deg"
                else:
                    label_text = "Incomplete landmarks"

        cv2.putText(
            frame, label_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2
        )

        cv2.imshow("Live Pose Angles + Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
