import numpy as np
import cv2
import os

def print_png_info(path):
    # Check that file exists
    if not os.path.isfile(path):
        print(f"File not found at: {path}")

    # Load image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Unable to read image")

    # Gather or calculate data
    filesize = os.path.getsize(path)
    shape = img.shape
    img_type = type(img)
    max = np.max(img)
    min = np.min(img)
    avg = np.average(img)

    color_mode = ""
    if len(img.shape) == 2:
        color_mode = "Greyscale"
    elif len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 1:
            color_mode = "Greyscale"
        elif channels == 3:
            color_mode = "BGR"
        elif channels == 4:
            color_mode = "BGRA"
    else:
        color_mode = "Unknown"
    
    dtype = img.dtype

    print(f"--- PNG at {path} ---")
    print(f"Filesize: {filesize}")
    print(f"Shape: {shape}")
    print(f"dtype: {dtype}")
    print(f"Type: {img_type}")
    print(f"Max: {max}")
    print(f"Min: {min}")
    print(f"Avg: {avg}")



def pad_to_square(image):
    height, width = image.shape[:2]
    side_size = max(height, width)
    # Create new image of zeros
    if len(image.shape) == 3:
        new_image = np.zeros((side_size, side_size, image.shape[2]), dtype=image.dtype)
    else:
        new_image = np.zeros((side_size, side_size), dtype=image.dtype)

    # Compute padding amounts
    y_offset = (side_size - height) // 2
    x_offset = (side_size - width) // 2

    # Insert original image into the center of new_image
    new_image[y_offset:y_offset+height, x_offset:x_offset+width] = image

    return new_image, x_offset, y_offset


def calcualte_transformation(depth_map, depth_scale, bbox):
    """
    
    """
    # Camera intrinsics
    fx = 1.636
    fy = 1.0225
    cx = 0
    cy = 0

    # Unpack bbox and calculate center
    x1, y1, x2, y2 = bbox
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    # Retrieve depth value at point
    depth_value = depth_map[v, u] * depth_scale

    if depth_value <= 0:
        print(f"Invalid depth value at ({u}, {v})")
        return None, None
    
    # Compute 3D coord
    x = (u - cx) * depth_value / fx
    y = (v - cy) * depth_value / fy
    z = depth_value

    # Construct transformation matrix
    transformation = np.eye(4)
    transformation[0:3, 0:3] = np.eye(3)
    transformation[0:3, 3] = np.array([x, y, z])

    return transformation, (x, y, z)




if __name__ == "__main__":
    print_png_info("C:\\Users\sondr\\Documents\\NTNU\\9_semester\\prosjekt\\synthetic_data\\training\\images\\train\\0709.png")
    print_png_info("C:\\Users\sondr\\Documents\\NTNU\\9_semester\\prosjekt\\scripts\\Images\\realsense_rawx100.png")