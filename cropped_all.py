import cv2
import numpy as np
import os
def crop_centered_on_objects(image_path, label_path, crop_sizes, output_dir, dataset_type, image_name):
    """
    Generate cropped images centered on each object in the label map.

    Args:
        image_path (str): Path to the input image.
        label_path (str): Path to the label (segmentation mask).
        crop_sizes (list of tuples): List of crop sizes (height, width).
        output_dir (str): Root directory to save cropped outputs.
        dataset_type (str): Dataset type (e.g., 'train' or 'val').
        image_name (str): Name of the input image.
    """
    # Load the image and label
    image = cv2.imread(image_path)
    labels = cv2.imread(label_path, 0)  # Load label as grayscale

    # Find connected components (objects) in the label map
    num_objects, object_mask = cv2.connectedComponents((labels > 0).astype(np.uint8))

    # Create output directories specific to dataset type
    images_dir = os.path.join(output_dir, dataset_type, "JPEGImages")
    labels_dir = os.path.join(output_dir, dataset_type, "SegmentationClass")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Generate crops for each connected object
    for crop_size in crop_sizes:
        for obj_id in range(1, num_objects):  # Skip background (0)
            # Get mask for the current object
            object_mask_current = (object_mask == obj_id).astype(np.uint8)

            # Find the coordinates of the object's pixels
            coords = np.column_stack(np.where(object_mask_current > 0))
            if coords.size == 0:
                continue

            # Select the center of the object
            center_y, center_x = coords.mean(axis=0).astype(int)

            # Define crop boundaries
            half_h, half_w = crop_size[0] // 2, crop_size[1] // 2
            y_min, y_max = max(0, center_y - half_h), min(image.shape[0], center_y + half_h)
            x_min, x_max = max(0, center_x - half_w), min(image.shape[1], center_x + half_w)

            # Extract the crop
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_label = labels[y_min:y_max, x_min:x_max]

            # Save the cropped image and label
            crop_image_filename = f"{image_name}_object_{obj_id}_cropsize_{crop_size[0]}x{crop_size[1]}.jpg"
            crop_label_filename = f"{image_name}_object_{obj_id}_cropsize_{crop_size[0]}x{crop_size[1]}.png"
            cv2.imwrite(os.path.join(images_dir, crop_image_filename), cropped_image)
            cv2.imwrite(os.path.join(labels_dir, crop_label_filename), cropped_label)

# Define parameters
image_dir = "SCRAP_V6_original/SCRAP_V6/Iron_material/JPEGImages"
label_dir = "SCRAP_V6_original/SCRAP_V6/Iron_material/SegmentationClass"
train_val_sets = {
    "train": "SCRAP_V6_original/SCRAP_V6/Iron_material/ImageSets/Segmentation/train.txt",
    "val": "SCRAP_V6_original/SCRAP_V6/Iron_material/ImageSets/Segmentation/val.txt"
}
crop_sizes = [(400, 400), (512, 512), (700, 700)]  # Crop sizes
output_dir = "cropped_output_new"  # New output directory

# Process all images in training and validation sets
for dataset_type, txt_file in train_val_sets.items():
    with open(txt_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]

    for image_name in image_names:
        image_path = os.path.join(image_dir, f"{image_name}.jpg")
        label_path = os.path.join(label_dir, f"{image_name}.png")

        if os.path.exists(image_path) and os.path.exists(label_path):
            crop_centered_on_objects(image_path, label_path, crop_sizes, output_dir, dataset_type, image_name)
        else:
            print(f"Warning: Missing file for {image_name}")
