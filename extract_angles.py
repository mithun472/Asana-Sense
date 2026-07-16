import os
import csv
import argparse
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from angle_utils import calc_all_angles, NUM_LANDMARKS

MODEL_PATH = "pose_landmarker_full.task"


def build_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,   # single-shot mode, not VIDEO
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
    )
    return vision.PoseLandmarker.create_from_options(options)


def process_dataset(dataset_dir, output_csv):
    """
    dataset_dir structure expected:
        dataset_dir/
            downdog/img1.jpg ...
            goddess/img1.jpg ...
            plank/ ...
            tree/ ...
            warrior2/ ...
    """
    landmarker = build_landmarker()
    fieldnames = ["image_path", "label"] + [f"angle_{i}" for i in range(NUM_LANDMARKS)]
    dataset_root = Path(dataset_dir).resolve()

    rows_written = 0
    skipped = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        classes = sorted(
            d.name for d in dataset_root.iterdir()
            if d.is_dir()
        )

        for label in classes:
            class_dir = dataset_root / label
            images = sorted(class_dir.iterdir())
            print(f"[{label}] {len(images)} images")

            for img_path in images:
                if not img_path.is_file():
                    continue

                frame = cv2.imread(str(img_path))
                if frame is None:
                    skipped += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                result = landmarker.detect(mp_image)  # IMAGE mode call, no timestamp needed

                if not result.pose_landmarks:
                    skipped += 1
                    continue

                landmarks = result.pose_landmarks[0]  # first detected person
                angles = calc_all_angles(landmarks)

                row = {"image_path": str(img_path), "label": label}
                for i in range(NUM_LANDMARKS):
                    row[f"angle_{i}"] = angles[i]

                writer.writerow(row)
                rows_written += 1

    landmarker.close()
    print(f"Done. Rows written: {rows_written}, skipped (no detection/bad read): {skipped}")
    print(f"CSV saved -> {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Root folder with class subfolders")
    parser.add_argument("--output_csv", default="angles_dataset.csv")
    args = parser.parse_args()

    process_dataset(args.dataset_dir, args.output_csv)
