import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

def prepare_cycle_crack_data(src_images_dir, src_annotations_dir, dest_cracked, dest_crack_free, total_target=3000):
    os.makedirs(dest_cracked, exist_ok=True)
    os.makedirs(dest_crack_free, exist_ok=True)
    
    # Collect all image/annotation pairs from training, validation, test
    pairs = []
    for split in ['training', 'validation', 'test']:
        img_split_dir = os.path.join(src_images_dir, split)
        ann_split_dir = os.path.join(src_annotations_dir, split)
        
        if not os.path.exists(img_split_dir) or not os.path.exists(ann_split_dir):
            continue
            
        for fname in os.listdir(img_split_dir):
            if not fname.endswith(('.jpg', '.png', '.bmp')):
                continue
            
            base = os.path.splitext(fname)[0]
            # Assumes annotation has same base and .png or .jpg
            ann_path_png = os.path.join(ann_split_dir, base + '.png')
            ann_path_jpg = os.path.join(ann_split_dir, base + '.jpg')
            
            if os.path.exists(ann_path_png):
                pairs.append((os.path.join(img_split_dir, fname), ann_path_png))
            elif os.path.exists(ann_path_jpg):
                pairs.append((os.path.join(img_split_dir, fname), ann_path_jpg))
                
    random.seed(42)
    random.shuffle(pairs)
    
    print(f"Total pairs found: {len(pairs)}")
    
    cracked_images = []
    crack_free_images = []
    
    print("Analyzing annotations to split into cracked and crack-free...")
    # We want to sample roughly total_target/2 for each if possible, or just sample until total_target
    target_cracked = total_target // 2
    target_crack_free = total_target - target_cracked
    
    for img_path, ann_path in tqdm(pairs):
        if len(cracked_images) >= target_cracked and len(crack_free_images) >= target_crack_free:
            break
            
        try:
            ann_img = Image.open(ann_path).convert('L') # grayscale
            ann_np = np.array(ann_img)
            
            is_cracked = np.sum(ann_np > 0) > 100  # [paper §IV-A]: >100 crack pixels; 1-100 discarded
            
            if is_cracked and len(cracked_images) < target_cracked:
                cracked_images.append(img_path)
            elif not is_cracked and len(crack_free_images) < target_crack_free:
                crack_free_images.append(img_path)
                
        except Exception as e:
            print(f"Error reading {ann_path}: {e}")
            
    print(f"Found {len(cracked_images)} cracked and {len(crack_free_images)} crack-free images.")
    
    print("Copying to destination directories...")
    for img_path in tqdm(cracked_images, desc="Copying cracked"):
        shutil.copy(img_path, os.path.join(dest_cracked, os.path.basename(img_path)))
        
    for img_path in tqdm(crack_free_images, desc="Copying crack-free"):
        shutil.copy(img_path, os.path.join(dest_crack_free, os.path.basename(img_path)))
        
    print("Done!")

if __name__ == "__main__":
    src_images = "/home/hieulc/avitech_13/omnicrack30k_data/images"
    src_annotations = "/home/hieulc/avitech_13/omnicrack30k_data/annotations"
    dest_cracked = "/home/hieulc/avitech_13/cycle_crack/data/cracked"
    dest_crack_free = "/home/hieulc/avitech_13/cycle_crack/data/crack_free"
    
    prepare_cycle_crack_data(src_images, src_annotations, dest_cracked, dest_crack_free, total_target=3000)
