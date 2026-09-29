import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import ActionLSTM
from feature_extraction import extract_keypoints

def main():
    # Load labels mapping
    try:
        actions = np.load('actions.npy', allow_pickle=True)
    except FileNotFoundError:
        print("actions.npy not found. Please run Phase 2 first.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")
    
    # Initialize and load the trained model
    # Must match training hyperparameters: HIDDEN_SIZE=128, BiLSTM
    model = ActionLSTM(input_size=231, hidden_size=128, num_layers=2, num_classes=len(actions)).to(device)
    
    try:
        model.load_state_dict(torch.load('action_model.pth', map_location=device, weights_only=True))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Setup MediaPipe Pose Landmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.7)

    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Rolling window buffer to store the last 30 frames
    sequence = deque(maxlen=30)
    
    # Smoothing algorithm buffer (Extension feature from PRD)
    predictions_buffer = deque(maxlen=10)
    confidence_buffer = deque(maxlen=10)

    print("Webcam live! Press 'q' to quit.")
    
    frame_timestamp_ms = 0
    current_action = "Gathering frames..."
    prev_keypoints = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Prepare image for MediaPipe (DO NOT flip before MediaPipe, it reverses handedness)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        frame_timestamp_ms += int(1000 / 30)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # Flip frame horizontally for selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Draw basic pose landmarks on the frame
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in pose_landmarks:
                    if lm.visibility > 0.5:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cx = w - cx # Invert x coordinate because the frame is flipped
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Extract features and append to rolling window
        keypoints = extract_keypoints(results.pose_landmarks, prev_keypoints)
        prev_keypoints = keypoints
        sequence.append(keypoints)
        
        # Run inference ONLY if we have collected a full 30-frame sequence
        if len(sequence) == 30:
            # Convert sequence to PyTorch tensor
            input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                
                # Convert raw scores to probabilities using Softmax
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predictions_buffer.append(predicted_idx.item())
                confidence_buffer.append(confidence.item())
                
                # Apply smoothing (majority vote over the last 10 frames)
                most_common_idx = max(set(predictions_buffer), key=predictions_buffer.count)
                avg_confidence = sum(confidence_buffer) / len(confidence_buffer)
                
                # Only display if confidence is high enough
                if avg_confidence > 0.5:
                    current_action = f"{actions[most_common_idx]}: {avg_confidence*100:.1f}%"
                else:
                    current_action = "Unknown"
                    
        # Overlay the predicted action on the screen
        cv2.putText(frame, current_action, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, current_action, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the result
        cv2.imshow('Real-Time Human Activity Recognition', frame)

        # Quit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == '__main__':
    main()
