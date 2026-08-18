# Project Requirements Document (PRD): Human Activity Recognition

## 1. Project Overview
**Project Name:** Real-Time Human Activity Recognition (HAR)
**Type:** Computer Vision / Deep Learning Personal Project
**Objective:** Build a lightweight, real-time computer vision system capable of detecting and classifying 10-12 distinct human activities (e.g., standing, walking, sitting, jumping, waving) using a standard webcam.

## 2. Problem & Motivation
Standard object detection models can only identify *who* or *what* is in a frame, but not *what they are doing*. Analyzing human actions requires understanding motion over time. This project serves as a hands-on implementation of spatial-temporal deep learning using modern, efficient tools (MediaPipe + LSTM) rather than heavy 3D CNNs.

## 3. Tech Stack & Libraries
*   **Language:** Python 3.8+
*   **Video Processing:** `opencv-python` (cv2)
*   **Pose Estimation:** `mediapipe` (Google's framework for skeletal keypoints)
*   **Deep Learning:** `tensorflow`/`keras` or `pytorch` (for the LSTM model)
*   **Data Handling:** `numpy`, `pandas`, `scikit-learn`

## 4. Scope & Features
### Must-Have (Core Project)
*   **Real-time skeletal tracking:** Draw MediaPipe pose landmarks on the live webcam feed.
*   **Data Collection Script:** A script to record and save landmark data (X, Y, Z coordinates) for each of the 10-12 classes into CSV/Numpy files.
*   **Sequential Neural Network:** An LSTM model trained on the sequential landmark data.
*   **Live Inference:** On-screen text displaying the predicted activity and confidence percentage (e.g., "Walking: 92%").

### Nice-to-Have (Extensions)
*   **FPS Counter:** Display processing speed on the video feed.
*   **Action Logging:** Save a text log of detected actions with timestamps.
*   **Smoothing Algorithm:** A moving average or buffer to prevent the on-screen label from flickering between classes.

## 5. Development Phases
1.  **Phase 1: Environment Setup & Pose Detection**
    *   Install libraries.
    *   Write a script to open the webcam and overlay MediaPipe pose skeletons.
2.  **Phase 2: Data Collection**
    *   Define the 10-12 target actions.
    *   Record 30-frame sequences (approx. 1 second of movement) for each action.
    *   Export 33 keypoints (132 coordinates per frame) to arrays.
3.  **Phase 3: Model Training**
    *   Pre-process data (train/test split, categorical encoding).
    *   Build a sequential LSTM model.
    *   Train the model and save the weights (`.h5` or `.pt`).
4.  **Phase 4: Real-Time Integration**
    *   Combine Phase 1 video capture with the Phase 3 trained model.
    *   Implement a rolling window buffer (keep the last 30 frames in memory).
    *   Output predictions to the screen.

## 6. Success Metrics
*   **Accuracy:** Achieve >85% validation accuracy on the test dataset.
*   **Performance:** Run at a minimum of 15-20 FPS on a standard CPU/integrated GPU.
*   **Robustness:** The model should accurately distinguish between similar motions (e.g., sitting down vs. standing up).