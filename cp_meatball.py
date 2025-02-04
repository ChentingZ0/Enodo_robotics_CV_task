import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
background_images_dir = 'background/Background_images/'  # Background images folder
background_masks_dir = 'background/Background_masks/'  # Background masks folder
meatball_images_dir = 'meatballs/JPEGImages/'  # Meatball images folder
meatball_masks_dir = 'meatballs/Masks/'  # Meatball masks folder

# Output directories
output_images_dir = 'background/appended_background/'
output_masks_dir = 'background/append_background_masks/'

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Get the list of background and meatball files
background_images = sorted(os.listdir(background_images_dir))
background_masks = sorted(os.listdir(background_masks_dir))
meatball_images = sorted(os.listdir(meatball_images_dir))
meatball_masks = sorted(os.listdir(meatball_masks_dir))

# Counter for naming
mix_counter = 1

# Loop through all background images
for bg_idx, (background_image_file, background_mask_file) in enumerate(zip(background_images, background_masks)):
    # Load the background image and mask
    background = cv2.imread(os.path.join(background_images_dir, background_image_file))
    background_mask = cv2.imread(os.path.join(background_masks_dir, background_mask_file), cv2.IMREAD_GRAYSCALE)

    # Randomly shuffle meatballs and select up to 3
    indices = np.random.choice(len(meatball_images), size=min(3, len(meatball_images)), replace=False)

    # Loop through up to 3 selected meatballs
    for mb_idx in indices:
        meatball_image_file = meatball_images[mb_idx]
        meatball_mask_file = meatball_masks[mb_idx]

        # Load meatball image and mask
        meatball_image = cv2.imread(os.path.join(meatball_images_dir, meatball_image_file))
        meatball_mask = cv2.imread(os.path.join(meatball_masks_dir, meatball_mask_file), cv2.IMREAD_GRAYSCALE)

        # Extract the meatball using the mask
        meatball = cv2.bitwise_and(meatball_image, meatball_image, mask=meatball_mask)

        # Find the bounding box for the meatball
        x, y, w, h = cv2.boundingRect(meatball_mask)

        # Crop the meatball and mask
        cropped_meatball = meatball[y:y + h, x:x + w]
        cropped_meatball_mask = meatball_mask[y:y + h, x:x + w]

        # Choose a random position for pasting
        paste_x = np.random.randint(0, background.shape[1] - cropped_meatball.shape[1])
        paste_y = np.random.randint(0, background.shape[0] - cropped_meatball.shape[0])

        # Overlay the meatball on the background
        region_of_interest = background[paste_y:paste_y + cropped_meatball.shape[0],
                             paste_x:paste_x + cropped_meatball.shape[1]]
        blended = np.where(cropped_meatball_mask[..., None] > 0, cropped_meatball, region_of_interest)
        background[paste_y:paste_y + cropped_meatball.shape[0], paste_x:paste_x + cropped_meatball.shape[1]] = blended

        # Update the background mask with value `1`
        roi_mask = background_mask[paste_y:paste_y + cropped_meatball_mask.shape[0],
                   paste_x:paste_x + cropped_meatball_mask.shape[1]]
        updated_region = np.where(cropped_meatball_mask > 0, 1, roi_mask)
        background_mask[paste_y:paste_y + cropped_meatball_mask.shape[0],
        paste_x:paste_x + cropped_meatball_mask.shape[1]] = updated_region

    # Save the updated background and mask with sequential names
    output_image_path = os.path.join(output_images_dir, f'mix{mix_counter}.jpg')
    output_mask_path = os.path.join(output_masks_dir, f'mix{mix_counter}.png')
    cv2.imwrite(output_image_path, background)
    cv2.imwrite(output_mask_path, background_mask)

    print(f"Processed mix{mix_counter} (background {bg_idx + 1}/{len(background_images)})")
    mix_counter += 1
