import numpy as np
import os

def augment_data(X, y, num_augmentations=20):
    """
    Augments the skeletal coordinate data by applying:
    1. Spatial Jitter (random noise)
    2. Body Scaling (making the skeleton slightly taller/shorter/wider)
    3. Random Time Shifting (shifting the sequence slightly)
    """
    augmented_X = []
    augmented_y = []
    
    print(f"Original dataset size: {X.shape[0]} sequences")
    print(f"Generating {num_augmentations} augmented copies per sequence...")
    
    for i in range(X.shape[0]):
        original_sequence = X[i]
        label = y[i]
        
        # Keep the original
        augmented_X.append(original_sequence)
        augmented_y.append(label)
        
        for _ in range(num_augmentations):
            aug_seq = np.copy(original_sequence)
            
            # 1. Spatial Jitter (Add random noise to coordinates)
            # Standard deviation of 0.02 (2% of screen space)
            noise = np.random.normal(0, 0.02, aug_seq.shape)
            # Do not apply noise to visibility scores (every 4th element)
            noise[:, 3::4] = 0 
            aug_seq += noise
            
            # 2. Body Scaling
            # Randomly scale X and Y independently to simulate different body proportions
            scale_x = np.random.uniform(0.9, 1.1)
            scale_y = np.random.uniform(0.9, 1.1)
            scale_z = np.random.uniform(0.9, 1.1)
            
            aug_seq[:, 0::4] *= scale_x # Scale all X coordinates
            aug_seq[:, 1::4] *= scale_y # Scale all Y coordinates
            aug_seq[:, 2::4] *= scale_z # Scale all Z coordinates
            
            # 3. Temporal Shifting (Simulate starting the action slightly later/earlier)
            # Shift the sequence by up to 2 frames forward or backward
            shift = np.random.randint(-2, 3)
            if shift > 0:
                # Shift forward (pad beginning with first frame)
                aug_seq = np.vstack([np.tile(aug_seq[0], (shift, 1)), aug_seq[:-shift]])
            elif shift < 0:
                # Shift backward (pad end with last frame)
                aug_seq = np.vstack([aug_seq[-shift:], np.tile(aug_seq[-1], (-shift, 1))])
            
            augmented_X.append(aug_seq)
            augmented_y.append(label)
            
    return np.array(augmented_X), np.array(augmented_y)

def main():
    try:
        X = np.load('X.npy')
        y = np.load('y.npy')
    except FileNotFoundError:
        print("X.npy or y.npy not found!")
        return

    # Augment the dataset (Creates 20 new variations for every 1 original sequence)
    # 159 * 21 = 3339 sequences
    X_aug, y_aug = augment_data(X, y, num_augmentations=20)
    
    print(f"\nAugmentation Complete!")
    print(f"New dataset size: {X_aug.shape[0]} sequences (was {X.shape[0]})")
    
    # Save the augmented data
    np.save('X_aug.npy', X_aug)
    np.save('y_aug.npy', y_aug)
    print("Saved to X_aug.npy and y_aug.npy")

if __name__ == '__main__':
    main()
