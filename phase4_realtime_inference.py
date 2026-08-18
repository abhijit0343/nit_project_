import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Model Architecture (Must match Phase 3 exactly) ---
class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ActionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def extract_keypoints(pose_landmarks):
    """Flattens 33 landmarks into a 132-length array, normalized to hips."""
    if not pose_landmarks:
        return np.zeros(33 * 4)
    landmarks = pose_landmarks[0]
    
    # We use the midpoint of the hips (landmarks 23 and 24) as the center origin
    center_x = (landmarks[23].x + landmarks[24].x) / 2
    center_y = (landmarks[23].y + landmarks[24].y) / 2
    center_z = (landmarks[23].z + landmarks[24].z) / 2
    
    keypoints = []
    for lm in landmarks:
        keypoints.extend([lm.x - center_x, lm.y - center_y, lm.z - center_z, lm.visibility])
    return np.array(keypoints)

def main():
    # Load labels mapping
    try:
        actions = np.load('actions.npy')
    except FileNotFoundError:
        print("actions.npy not found. Please run Phase 2 first.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")
    
    # Initialize and load the trained model
    model = ActionLSTM(input_size=132, hidden_size=32, num_layers=2, num_classes=len(actions)).to(device)
    
    try:
        model.load_state_dict(torch.load('action_model.pth', map_location=device))
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

    print("Webcam live! Press 'q' to quit.")
    
    frame_timestamp_ms = 0
    current_action = "Gathering frames..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        
        # Prepare image for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        frame_timestamp_ms += int(1000 / 30)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # Draw basic pose landmarks on the frame
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in pose_landmarks:
                    if lm.visibility > 0.5:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Extract features and append to rolling window
        keypoints = extract_keypoints(results.pose_landmarks)
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
                
                # Apply smoothing (majority vote over the last 10 frames)
                most_common_idx = max(set(predictions_buffer), key=predictions_buffer.count)
                
                # Only display if confidence is high enough
                if confidence.item() > 0.5:
                    current_action = f"{actions[most_common_idx]}: {confidence.item()*100:.1f}%"
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
