import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Define pose connections manually since older solutions module is removed
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def draw_landmarks(image, landmarks, connections):
    h, w, c = image.shape
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(image, start_pt, end_pt, (255, 255, 255), 2)
                
    # Draw points
    for idx, lm in enumerate(landmarks):
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

def main():
    # Setup Pose Landmarker
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
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    frame_timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the image and find pose
        frame_timestamp_ms += int(1000 / 30) # approx 30 fps
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw the pose annotations on the original image
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                draw_landmarks(frame, pose_landmarks, POSE_CONNECTIONS)

        # Display the resulting frame
        cv2.imshow('MediaPipe Pose Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
