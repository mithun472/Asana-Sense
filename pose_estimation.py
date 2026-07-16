import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 33 BlazePose landmark layout topology mappings
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

# Ensure you download 'pose_landmarker_full.task' before running this script
MODEL_PATH = "pose_landmarker_full.task"

# Configure configuration objects
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.8,   # Minimum score to detect a person
    min_pose_presence_confidence=0.8,    # Minimum score to confirm landmarks present
    min_tracking_confidence=0.8          # Minimum score for temporal tracking continuity
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not activate camera device index 0.")

# Track loop baseline timestamp reference
start_time_ms = int(time.time() * 1000)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    print("Inference engine active. Press 'q' to stop.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # 1. Natural Mirror Matrix Conversion
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 2. Fix Color Space Mismatch (BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Create MediaPipe Image Instance passing the converted RGB data
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 4. Generate Strict Monotonic Millisecond Clock Timestamps
        current_time_ms = int(time.time() * 1000)
        timestamp_ms = current_time_ms - start_time_ms

        # 5. Execute Context Inference
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 6. Render Data on Original BGR Preview frame
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                
                # Draw lines connecting skeletal structures
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        pt1 = pose_landmarks[start_idx]
                        pt2 = pose_landmarks[end_idx]
                        
                        # Denormalize mapping to canvas grid
                        x1, y1 = int(pt1.x * w), int(pt1.y * h)
                        x2, y2 = int(pt2.x * w), int(pt2.y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw joint coordinates 
                for landmark in pose_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        cv2.imshow("Optimized MediaPipe Task Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
