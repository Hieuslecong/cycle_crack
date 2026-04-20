import os
import shutil
import random
import cv2
import numpy as np

# Paths
img_dir = "/mnt/hieudeptrai/Hieu/omnicrack30k_data/images/training/"
label_dir = "/mnt/hieudeptrai/Hieu/omnicrack30k_data/annotations/training/"
crack_out = "/home/hieulc/avitech_13/cycle_crack/data/crack"
noncrack_out = "/home/hieulc/avitech_13/cycle_crack/data/noncrack"

# Ensure output dirs exist
os.makedirs(crack_out, exist_ok=True)
os.makedirs(noncrack_out, exist_ok=True)

# Get all image files
all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(all_files)} total images.")

# Randomly select 5000 files
if len(all_files) > 5000:
    selected_files = random.sample(all_files, 5000)
else:
    selected_files = all_files
    print(f"Note: Less than 5000 images found, using all {len(selected_files)}.")

count_crack = 0
count_noncrack = 0
not_found = 0

print("Starting filtering...")
for i, filename in enumerate(selected_files):
    label_path = os.path.join(label_dir, filename)
    img_path = os.path.join(img_dir, filename)
    
    if not os.path.exists(label_path):
        # Try finding png if original was jpg or vice versa
        name_no_ext = os.path.splitext(filename)[0]
        # Common pattern: labels might be .png always
        possible_label = os.path.join(label_dir, name_no_ext + ".png")
        if os.path.exists(possible_label):
            label_path = possible_label
        else:
            not_found += 1
            continue

    # Read label
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        not_found += 1
        continue

    # Crack is black (0), background is white (255)
    # If there is any pixel < 128, consider it as crack
    if np.any(label_img < 128):
        shutil.copy2(img_path, os.path.join(crack_out, filename))
        count_crack += 1
    else:
        shutil.copy2(img_path, os.path.join(noncrack_out, filename))
        count_noncrack += 1

    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/5000...")

print("\nDone!")
print(f"Crack images copied: {count_crack}")
print(f"Non-crack images copied: {count_noncrack}")
print(f"Labels not found/skipped: {not_found}")
print(f"Total processed: {count_crack + count_noncrack}")
