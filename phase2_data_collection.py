import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Settings
DATASET_DIR = 'dataset'
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 33 * 4 # x, y, z, visibility for 33 landmarks

def extract_keypoints(pose_landmarks):
    """Flattens 33 landmarks into a 132-element numpy array, normalized relative to the hips."""
    if not pose_landmarks:
        return np.zeros(FEATURES_PER_FRAME)
    
    # We take the first person detected
    landmarks = pose_landmarks[0]
    
    # We use the midpoint of the hips (landmarks 23 and 24) as the center origin
    center_x = (landmarks[23].x + landmarks[24].x) / 2
    center_y = (landmarks[23].y + landmarks[24].y) / 2
    center_z = (landmarks[23].z + landmarks[24].z) / 2
    
    keypoints = []
    for lm in landmarks:
        # Subtract center to make coordinates relative to the body
        keypoints.extend([lm.x - center_x, lm.y - center_y, lm.z - center_z, lm.visibility])
    return np.array(keypoints)

def main():
    # Setup Pose Landmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    landmarker = vision.PoseLandmarker.create_from_options(options)

    actions = np.array([folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))])
    print(f"Found actions: {actions}")

    # Label map
    label_map = {label:num for num, label in enumerate(actions)}

    sequences, labels = [], []
    frame_timestamp_ms = 0

    for action in actions:
        action_path = os.path.join(DATASET_DIR, action)
        videos = [f for f in os.listdir(action_path) if f.endswith('.avi')]
        
        for video_file in videos:
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            # We will extract non-overlapping 30-frame windows
            frames_buffer = []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30
                
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process the frame
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                # Timestamp needs to be strictly increasing
                frame_timestamp_ms += int(1000 / fps)
                
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                # Extract keypoints
                keypoints = extract_keypoints(results.pose_landmarks)
                frames_buffer.append(keypoints)
                
                # If we collected 30 frames, save as a sequence and clear buffer
                if len(frames_buffer) == SEQUENCE_LENGTH:
                    sequences.append(frames_buffer)
                    labels.append(label_map[action])
                    
                    # Overlap handling: to get more data, we could keep the last 15 frames
                    # frames_buffer = frames_buffer[15:]
                    
                    # Or clear it completely for non-overlapping sequences:
                    frames_buffer = [] 
                    
            cap.release()
            print(f"Processed {video_file} for action '{action}'.")

    landmarker.close()

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)

    print(f"\nData collection complete! X shape: {X.shape}, y shape: {y.shape}")

    # Save data
    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('actions.npy', actions) # Save the action names to map back the labels later
    print("Saved features to X.npy, labels to y.npy, and class names to actions.npy")

if __name__ == '__main__':
    main()
