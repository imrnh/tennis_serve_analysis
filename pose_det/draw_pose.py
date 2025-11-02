import numpy as np
import cv2

# Drawing pose to a black image.
def draw_pose_on_image(mp_drawing, mp_pose, LANDMARK_DRAWING_SPEC, CONNECTION_DRAWING_SPEC, original_image_shape, all_pose_data, target_image=None):
    """
    Draws pose skeletons onto a target image or a new black image.

    Args:
        original_image_shape (tuple): The (height, width, channels) of the
                                      original image.
        all_pose_data (list): The list of (landmarks, box) tuples from
                              detect_poses_in_boxes.
        target_image (np.array, optional): The image to draw on. If None,
                                           a new black image is created.

    Returns:
        np.array: The image with poses drawn on it.
    """
    
    h, w, _ = original_image_shape
    
    # 1. Create or prepare the canvas
    if target_image is None:
        # Create a new black image
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Use the provided target image
        # Ensure its size matches the original image shape
        if target_image.shape != original_image_shape:
            print(f"Warning: Target image shape {target_image.shape} does not match original {original_image_shape}.")
            print("Resizing target image to match.")
            canvas = cv2.resize(target_image, (w, h))
        else:
            canvas = target_image.copy() # Use a copy to avoid modifying the original

    # 2. Draw each pose
    for (landmarks, box) in all_pose_data:
        x1, y1, x2, y2 = box
        
        # Create a "view" (a sub-array) of the canvas corresponding to the
        # bounding box. Drawing on this 'canvas_crop' will directly
        # modify the main 'canvas'.
        canvas_crop = canvas[y1:y2, x1:x2]
        
        # Get the dimensions of the crop for drawing
        crop_h, crop_w, _ = canvas_crop.shape
        if crop_h == 0 or crop_w == 0:
            continue # Skip if the crop is invalid

        # Draw the landmarks onto the canvas sub-region
        mp_drawing.draw_landmarks(
            image=canvas_crop,
            landmark_list=landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_DRAWING_SPEC,
            connection_drawing_spec=CONNECTION_DRAWING_SPEC
        )
        
    return canvas