import time
import cv2
import mediapipe as mp
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from angle_utils import calc_all_angles, JOINT_TRIPLETS, JOINT_LABELS, build_feedback_sentences

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
class_means = bundle["class_means"]        # {label: [mean_angle_per_feature_col]}
class_stds = bundle["class_stds"]          # {label: [std_angle_per_feature_col]}
class_thresholds = bundle["class_thresholds"]  # {label: confidence_pct_cutoff}
joint_ids = [int(c.split("_")[1]) for c in feature_cols]

DEFAULT_CONF_THRESH = 55.0   # fallback for a class with no learned threshold
STD_MULTIPLIER = 1.5         # how many stds off before a joint is "wrong"
MAX_FLAGGED_JOINTS = 3       # how many worst joints to name in the label

# ---- Smoothing / hysteresis (kills frame-to-frame label flicker) ----
EMA_ALPHA = 0.3          # weight on new frame's proba vector. Lower = smoother, laggier.
SWITCH_MARGIN = 8.0      # candidate class must beat locked class by this many EMA % pts
SWITCH_STREAK_NEEDED = 5 # ...for this many consecutive frames before switch commits

ema_proba = None         # running EMA over clf.classes_ order
locked_label = None      # currently displayed pose (None = nothing locked yet)
challenger_label = None  # candidate trying to unseat locked_label
challenger_streak = 0    # consecutive frames challenger has been ahead

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

WINDOW_NAME = "Live Pose Angles + Classifier"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
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
        feedback_lines = []

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
                # class) drives the joint dot color + on-screen +/- degree
                # readout. feedback_lines (English sentences) is what
                # actually gets shown to the user as advice.
                joint_status = {}  # landmark_id -> "ok" | "off" | None (n/a)
                joint_diff = {}    # landmark_id -> SIGNED angle diff from class mean (deg)

                if all(v is not None for v in feature_vec):
                    raw_proba = clf.predict_proba([feature_vec])[0]

                    # ---- EMA smoothing over the whole proba vector ----
                    if ema_proba is None:
                        ema_proba = raw_proba.copy()
                    else:
                        ema_proba = EMA_ALPHA * raw_proba + (1 - EMA_ALPHA) * ema_proba

                    # rank classes by smoothed confidence
                    order = ema_proba.argsort()[::-1]
                    top_idx = order[0]
                    top_label = clf.classes_[top_idx]
                    top_conf = ema_proba[top_idx] * 100

                    # ---- hysteresis: only let a NEW label take over the
                    # display if it beats the currently locked one by a
                    # margin, and keeps beating it for several frames in a
                    # row. A single noisy frame can't flip the display.
                    if locked_label is None:
                        candidate = top_label
                        candidate_conf = top_conf
                    else:
                        locked_idx = list(clf.classes_).index(locked_label)
                        locked_conf = ema_proba[locked_idx] * 100

                        if top_label == locked_label:
                            challenger_label = None
                            challenger_streak = 0
                            candidate = locked_label
                            candidate_conf = locked_conf
                        else:
                            beats_by_margin = (top_conf - locked_conf) >= SWITCH_MARGIN
                            if beats_by_margin and top_label == challenger_label:
                                challenger_streak += 1
                            elif beats_by_margin:
                                challenger_label = top_label
                                challenger_streak = 1
                            else:
                                challenger_label = None
                                challenger_streak = 0

                            if challenger_streak >= SWITCH_STREAK_NEEDED:
                                candidate = top_label
                                candidate_conf = top_conf
                                challenger_label = None
                                challenger_streak = 0
                            else:
                                candidate = locked_label
                                candidate_conf = locked_conf

                    pred = candidate
                    confidence = candidate_conf

                    conf_thresh = class_thresholds.get(pred, DEFAULT_CONF_THRESH)

                    if confidence < conf_thresh:
                        label_text = f"neutral ({confidence:.1f}%, needs {conf_thresh:.0f}%)"
                        locked_label = None
                        challenger_label = None
                        challenger_streak = 0
                    else:
                        locked_label = pred
                        ref_vec = class_means[pred]
                        std_vec = class_stds.get(pred, [15.0] * len(ref_vec))

                        # per-joint diff, normalized by that joint's natural
                        # spread for this pose, so a joint that's always
                        # rigid in this pose gets a tight tolerance and a
                        # joint that naturally varies gets more slack.
                        # SIGNED diff kept here (not abs) so direction info
                        # survives for the English feedback sentences below
                        # and for the on-screen +/- degree readout.
                        per_joint_diff = []
                        for jid, feat_val, ref_val, std_val in zip(joint_ids, feature_vec, ref_vec, std_vec):
                            diff = feat_val - ref_val
                            is_off = abs(diff) > (STD_MULTIPLIER * std_val)
                            joint_status[jid] = "off" if is_off else "ok"
                            joint_diff[jid] = diff
                            per_joint_diff.append((jid, diff, is_off))

                        feedback_lines = build_feedback_sentences(
                            feature_vec, joint_ids, ref_vec, std_vec,
                            std_multiplier=STD_MULTIPLIER, max_msgs=MAX_FLAGGED_JOINTS
                        )

                        if not feedback_lines:
                            label_text = f"{pred} ({confidence:.1f}%) - correct"
                        else:
                            label_text = f"{pred} ({confidence:.1f}%)"
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

                # draw per-joint numbers: diff-from-reference (deg) when we
                # have a confident classification, else raw joint angle
                for j in DISPLAY_JOINTS:
                    if j not in JOINT_TRIPLETS:
                        continue
                    lm = pose_landmarks[j]
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if j in joint_diff:
                        d = joint_diff[j]
                        txt = f"{d:+.0f}"
                        color = (0, 0, 255) if joint_status.get(j) == "off" else (0, 220, 0)
                    else:
                        ang = angles[j]
                        if ang is None:
                            continue
                        txt = f"{int(ang)}"
                        color = (0, 0, 255)

                    cv2.putText(
                        frame, txt, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

        cv2.putText(
            frame, label_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

        # English coaching sentences, stacked below the main label
        for i, line in enumerate(feedback_lines):
            cv2.putText(
                frame, line, (20, 75 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2
            )

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nCtrl-C caught. Shutting down clean.")

finally:
    cap.release()
    cv2.destroyAllWindows()