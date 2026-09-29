import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Settings
DATASET_DIR = 'dataset/UCF-101/'
SEQUENCE_LENGTH = 30
NUM_ACTIONS = 10          # How many action classes to use
MAX_VIDEOS_PER_ACTION = 50  # Cap per class to keep collection time reasonable

from feature_extraction import extract_keypoints


def process_video(args):
    """
    Worker function: processes a single video file and returns a list of
    (sequence, label) pairs.  Each call creates its own MediaPipe landmarker
    so it is safe to run in a subprocess.
    """
    video_path, label, model_asset_path = args

    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    frames_buffer = []
    prev_keypoints = None
    frame_timestamp_ms = 0       # per-video timestamp — avoids the global-state bug
    sequences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        frame_timestamp_ms += int(1000 / fps)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        keypoints = extract_keypoints(results.pose_landmarks, prev_keypoints)
        prev_keypoints = keypoints
        frames_buffer.append(keypoints)

        if len(frames_buffer) == SEQUENCE_LENGTH:
            sequences.append((np.array(frames_buffer, dtype=np.float32), label))
            frames_buffer = []   # non-overlapping windows

    cap.release()
    landmarker.close()
    return sequences


def main():
    # Resolve absolute path so subprocesses can find the model file
    model_asset_path = os.path.abspath('pose_landmarker_full.task')

    # Discover actions (sorted for reproducibility; pick first NUM_ACTIONS)
    all_folders = sorted([
        f for f in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, f))
    ])
    actions = np.array(all_folders[:NUM_ACTIONS])
    print(f"Using {len(actions)} action classes: {list(actions)}")

    label_map = {label: num for num, label in enumerate(actions)}

    # Build work list: (video_path, label_int, model_path)
    tasks = []
    for action in actions:
        action_path = os.path.join(DATASET_DIR, action)
        videos = [f for f in os.listdir(action_path) if f.endswith(('.avi', '.mp4'))]
        videos = videos[:MAX_VIDEOS_PER_ACTION]
        for vf in videos:
            tasks.append((os.path.join(action_path, vf), label_map[action], model_asset_path))

    print(f"Total videos to process: {len(tasks)}")

    all_sequences, all_labels = [], []

    # Use up to 4 workers (MediaPipe is CPU-bound; more workers = faster collection)
    num_workers = min(4, os.cpu_count() or 1)
    print(f"Processing with {num_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(process_video, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Extracting keypoints", unit="video") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    seqs = future.result()
                    for seq, lbl in seqs:
                        all_sequences.append(seq)
                        all_labels.append(lbl)
                except Exception as exc:
                    print(f"\n[WARNING] {task[0]} failed: {exc}")
                finally:
                    pbar.update(1)

    if not all_sequences:
        print("No sequences extracted. Check your dataset path and video files.")
        return

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\nData collection complete!")
    print(f"  X shape : {X.shape}  (sequences, frames, features)")
    print(f"  y shape : {y.shape}")
    print(f"  Classes : {dict(zip(actions, [int((y==i).sum()) for i in range(len(actions))]))} ")

    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('actions.npy', actions)
    print("Saved  ->  X.npy, y.npy, actions.npy")


if __name__ == '__main__':
    main()
