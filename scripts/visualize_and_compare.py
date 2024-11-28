import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns


def read_depth_map(filepath):
    depth_map = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if depth_map is None:
        raise FileNotFoundError(f"File not found at {filepath}")
    
    return depth_map

def visualize_depth_map(depth_map, title= "Depth Map"):
    plt.figure()
    plt.imshow(depth_map, cmap= 'jet')
    plt.colorbar(label= 'Depth value (mm)')
    plt.title(title)
    plt.show()

def print_image_properties(image, image_name= "Image"):
    height, width = image.shape[:2]
    dtype = image.dtype
    min_val = np.min(image)
    max_val = np.max(image)
    mean_val = np.mean(image[image > 0])

    print(f"---------- {image_name} properties ----------")
    print(f"    - Dimensions: {width} x {height}")
    print(f"    - Data type: {dtype}")
    print(f"    - Min value: {min_val}")
    print(f"    - Max value: {max_val}")
    print(f"    - Mean value: {mean_val}")

def analyze_depth_map(depth_map_path, title="Depth Map Analysis", save_path=None):
    """
    Analyze depth map with title on left for vertical stacking
    """
    depth_map = read_depth_map(depth_map_path)
    fig = plt.figure(figsize=(18, 4))
    
    # Add title to the left of the plots
    plt.subplots_adjust(left=0.15)
    fig.text(0.02, 0.5, title, rotation=0, verticalalignment='center', fontsize=10)

    # Convert and normalize
    depth_uint8 = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_normalized = depth_uint8.astype(np.float32) / 255.0
    
    # Plot depth map
    plt.subplot(121)
    im = plt.imshow(depth_normalized, cmap='viridis')
    plt.colorbar(im)
    plt.title('Normalized Depth Map')
    
    # Plot log histogram
    plt.subplot(122)
    plt.hist(depth_normalized.flatten(), bins=50, log=True)
    plt.title('Depth Distribution\n(log scale)')
    plt.xlabel('Normalized Depth (uint8/255)')
    plt.ylabel('Count (log)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()  
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()
    
    return depth_normalized, depth_uint8

def compare_depth_maps(map1, map2, title1= 'Realsense', title2= 'Blender'):
    # Show depth maps next to eachother
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(map1, cmap= 'jet')
    plt.colorbar(label= 'Depth value (mm)')
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(map2, cmap= 'jet')
    plt.colorbar(label= 'Depth value (mm)')
    plt.title(title2)

    plt.show()

    # Calculate and plot histograms
    hist1, bins1 = np.histogram(map1.flatten(), bins= 50, range=(0, np.max(map1)))
    hist2, bins2 = np.histogram(map2.flatten(), bins= 50, range=(0, np.max(map1)))

    plt.figure()
    plt.plot(bins1[:-1], hist1, label=title1)
    plt.plot(bins2[:-1], hist2, label=title2)
    plt.xlabel('Depth value (mm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Depth Map Histograms')
    plt.show()

def draw_bounding_boxes(folder_path, filename, class_names= None):
    """
    
    """
    # Setup paths
    image_path = os.path.join(folder_path, 'depth_maps', f"{filename}.png")
    label_path = os.path.join(folder_path, 'labels', f"{filename}.txt")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at: {image_path}")
        return
    height, width, _ = image.shape

    # Read annotation file
    if not os.path.exists(label_path):
        print(f"Label file not found at: {label_path}\nDisplaying image withound bounding boxes")
    else:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Process each line in text file as seperate bounding boxes
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) != 5:
                print(f"Invalid annotation format: {line}")
                continue

            class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
            class_id = int(class_id)

            # Convert to pixel values
            x_center_abs = x_center * width
            y_center_abs = y_center * height
            bbox_width_abs = bbox_width * width
            bbox_height_abs = bbox_height * height

            # Calcalutate corners
            x1 = int(x_center_abs - bbox_width_abs/2)
            y1 = int(y_center_abs - bbox_height_abs/2)
            x2 = int(x_center_abs + bbox_width_abs/2)
            y2 = int(y_center_abs + bbox_height_abs/2)

            # Define and draw bounding box
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # TODO: Add something for display class labels
    
    # Display image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize= (10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    

def main():
    # COMPARISON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    realsense_dir = os.path.join(script_dir, 'Images', 'realsense_test.png')
    bledner_dir = os.path.join(script_dir, 'Images', '0050.png')
    realsense = read_depth_map(realsense_dir)
    blender = read_depth_map(bledner_dir)
    #compare_depth_maps(realsense, blender)

    # BOUNDING BOX
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    synthetic_data_path = os.path.join(parent_dir, 'synthetic_data', 'test')

    filename = '0009'
    #draw_bounding_boxes(synthetic_data_path, filename)

    # ANALYZE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    filename = 'final'
    depth_map_path = os.path.join(parent_dir, 'synthetic_data', 'report', filename + '.png')
    save_path = os.path.join(parent_dir, 'synthetic_data', 'report', filename + '_plot')

    analyze_depth_map(depth_map_path, 'Final', save_path)

if __name__=="__main__":
    main()