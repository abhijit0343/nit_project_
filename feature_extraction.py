import numpy as np

def extract_keypoints(pose_landmarks, prev_keypoints=None):
    """
    Extracts 231 features from 33 pose landmarks.
    Features: 
    - (x, y, z) relative to hips and scaled by shoulder width.
    - visibility
    - (dx, dy, dz) velocity relative to previous frame
    
    If visibility < 0.5, x, y, z and dx, dy, dz are set to 0.
    """
    if not pose_landmarks:
        return np.zeros(33 * 7)
        
    landmarks = pose_landmarks[0]
    
    # Calculate Center (midpoint of hips: 23, 24)
    center_x = (landmarks[23].x + landmarks[24].x) / 2
    center_y = (landmarks[23].y + landmarks[24].y) / 2
    center_z = (landmarks[23].z + landmarks[24].z) / 2
    
    # Calculate Scale (distance between shoulders: 11, 12)
    # Using Euclidean distance in 3D
    shoulder_dist = np.sqrt(
        (landmarks[11].x - landmarks[12].x)**2 + 
        (landmarks[11].y - landmarks[12].y)**2 + 
        (landmarks[11].z - landmarks[12].z)**2
    )
    
    # Avoid division by zero
    if shoulder_dist < 1e-5:
        shoulder_dist = 1.0
        
    current_keypoints = []
    
    # We will build a temporary list of (x,y,z) just to compute velocity easier
    current_xyz = []
    
    for lm in landmarks:
        if lm.visibility >= 0.5:
            norm_x = (lm.x - center_x) / shoulder_dist
            norm_y = (lm.y - center_y) / shoulder_dist
            norm_z = (lm.z - center_z) / shoulder_dist
            current_keypoints.extend([norm_x, norm_y, norm_z, lm.visibility])
            current_xyz.append((norm_x, norm_y, norm_z))
        else:
            current_keypoints.extend([0.0, 0.0, 0.0, lm.visibility])
            current_xyz.append((0.0, 0.0, 0.0))
            
    # Calculate velocity
    for i in range(33):
        curr_x, curr_y, curr_z = current_xyz[i]
        
        if prev_keypoints is not None and landmarks[i].visibility >= 0.5:
            # prev_keypoints structure: [x0, y0, z0, v0, dx0, dy0, dz0, x1, y1, z1, v1...]
            # The indices for xyz of landmark i in prev_keypoints are: i*7, i*7+1, i*7+2
            prev_x = prev_keypoints[i*7]
            prev_y = prev_keypoints[i*7 + 1]
            prev_z = prev_keypoints[i*7 + 2]
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            dz = curr_z - prev_z
        else:
            dx, dy, dz = 0.0, 0.0, 0.0
            
        current_keypoints.extend([dx, dy, dz])
        
    # Re-arrange the list so it is grouped by landmark [x, y, z, vis, dx, dy, dz] * 33
    final_features = []
    for i in range(33):
        # Base features (4) + Velocity features (3)
        base_idx = i * 4
        vel_idx = 33 * 4 + i * 3
        
        final_features.extend(current_keypoints[base_idx : base_idx+4])
        final_features.extend(current_keypoints[vel_idx : vel_idx+3])
        
    return np.array(final_features)
