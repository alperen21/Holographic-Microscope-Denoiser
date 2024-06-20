import cv2
import numpy as np


def is_background_segmentation(mask):
    # Calculate the area of the segmentation
    segmentation_area = np.sum(mask == 255)
    
    # Calculate the total area of the image
    total_area = mask.size
    
    # Check if the segmentation occupies more than half of the image
    if segmentation_area > total_area / 2:
        return True
    else:
        return False

def contour_to_mask(normalized_contour_data, width : int, height : int, normalised : bool = False):
    # Example normalized points: x1 y1 x2 y2 x3 y3...

    # Split the string into floats
        
    points = list(map(float, normalized_contour_data))
    # Desired dimensions for the mask

    # Convert normalized coordinates to pixel coordinates
    points = np.array(points).reshape(-1, 2) 
    
    if normalised:
        points *= np.array([height, width])

    # Reshape into integer values as OpenCV doesn't handle floats for points
    points = np.array(points, dtype=np.int32)

    # Create a blank mask
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw the contour on the mask
    cv2.fillPoly(mask, [points], color=255)
    
    return mask


def display_mask(mask):
    # Optionally, display the mask
    cv2.imshow('Combined Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_combined_mask(mask_path, image_shape):
    with open(mask_path, "r") as file:
        contours = file.readlines()
        
    masks = []
    height, width, _ = image_shape
    for contour in contours:
        contour = contour.strip().split(" ")[1:]
        masks.append(contour_to_mask(contour, width, height))
    
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for mask in masks:
        if not is_background_segmentation(mask):
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask

def overlay_mask_on_image(image, mask):
    # Create a copy of the image to avoid modifying the original
    image_with_mask = image.copy()
    
    # Set the pixels to zero wherever the mask is true
    image_with_mask[mask == 255] = 0
    
    return image_with_mask
