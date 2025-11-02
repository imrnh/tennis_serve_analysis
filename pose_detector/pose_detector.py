import cv2
import mediapipe as mp
import numpy as np
import os



# Core Pose Detector | Detect pose from a region.
def get_pose_landmarks(image_crop, pose_model):
    """
    Runs MediaPipe Pose detection on a single image crop.

    Args:
        image_crop (np.array): The image (as a NumPy array) to process.
        pose_model: The initialized MediaPipe Pose model instance.

    Returns:
        The detected pose_landmarks object, or None if no pose is found.
    """
    # MediaPipe works with RGB, OpenCV uses BGR
    image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    
    # Process the image crop
    results = pose_model.process(image_crop_rgb)
    
    # Return the landmarks
    return results.pose_landmarks

# Bounding Box Loop | Detect all the person and their pose for an image.
def detect_poses_in_boxes(mp_pose, image, bounding_boxes):
    """
    Detects poses within specified bounding boxes in an image.

    Args:
        image (np.array): The full original image.
        bounding_boxes (list): A list of tuples, where each tuple is
                               (x1, y1, x2, y2) defining a bounding box.

    Returns:
        list: A list of tuples, (landmarks, box), where 'landmarks'
              is the detected pose_landmarks object and 'box' is the
              original bounding box it was found in.
    """
    all_pose_data = []
    
    # Initialize the Pose model using a 'with' block for proper management
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            
            # Create the image crop based on the bounding box
            # Add checks for valid box dimensions
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0] or x1 >= x2 or y1 >= y2:
                print(f"Skipping invalid bounding box: {box}")
                continue
                
            image_crop = image[y1:y2, x1:x2]
            
            # Check if crop is empty
            if image_crop.size == 0:
                print(f"Skipping empty crop for box: {box}")
                continue

            # Call the core detector
            landmarks = get_pose_landmarks(image_crop, pose)
            
            if landmarks:
                # If a pose is found, store both the landmarks and the box
                all_pose_data.append((landmarks, box))
                
    return all_pose_data
