# Human Activity Recognition (HAR): Stages & Best Practices

## Project Development Stages

1. **Data Collection & Curation**
   - Gather or record video data for the 10-12 target activities.
   - Organize and cleanly label the dataset.

2. **Pose Extraction (Feature Engineering)**
   - Utilize OpenCV to process video frames.
   - Run MediaPipe to extract 2D/3D skeletal keypoints (joints).
   - Export coordinates (X, Y, Z, visibility) into CSV or NumPy arrays.

3. **Temporal Windowing & Preprocessing**
   - Group the sequential keypoint data into fixed-length windows (e.g., 30 frames).
   - Encode categorical text labels into a machine-readable format (e.g., one-hot encoding).

4. **LSTM Model Training & Evaluation**
   - Build a sequential neural network (LSTM) to process the temporal data.
   - Train the model using training/validation splits to prevent overfitting.
   - Save the finalized model weights.

5. **Real-Time Inference Pipeline**
   - Open a live OpenCV webcam stream.
   - Maintain a rolling buffer (using a `deque`) of the most recent frames.
   - Feed the buffer to the model and overlay the predicted action on the live video feed.

## Critical Best Practices

* **Normalize Your Coordinates:** Convert absolute pixel coordinates into relative coordinates (e.g., relative to the hip joint) to make the model scale-invariant.
* **Temporal Smoothing:** Apply a moving average to the output predictions to prevent the displayed classification from flickering.
* **Confidence Thresholding:** Filter out frames where the MediaPipe confidence score falls below a certain threshold to prevent injecting noise into the model buffer.
* **Balance Your Dataset:** Ensure an equal distribution of samples across all 10-12 classes to prevent prediction bias.
* **Optimize Inference:** Use a rolling window approach rather than processing entirely new batches of frames to maintain a high framerate during live detection.