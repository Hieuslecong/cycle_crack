import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

def prepare_cycle_crack_patches(src_images_dir, src_annotations_dir, dest_cracked, dest_crack_free, total_target=3000, patch_size=256):
    os.makedirs(dest_cracked, exist_ok=True)
    os.makedirs(dest_crack_free, exist_ok=True)
    
    pairs = []
    for split in ['training', 'validation', 'test']:
        img_split_dir = os.path.join(src_images_dir, split)
        ann_split_dir = os.path.join(src_annotations_dir, split)
        if not os.path.exists(img_split_dir) or not os.path.exists(ann_split_dir): continue
        for fname in os.listdir(img_split_dir):
            if not fname.endswith(('.jpg', '.png', '.bmp')): continue
            base = os.path.splitext(fname)[0]
            ann_path_png = os.path.join(ann_split_dir, base + '.png')
            ann_path_jpg = os.path.join(ann_split_dir, base + '.jpg')
            if os.path.exists(ann_path_png):
                pairs.append((os.path.join(img_split_dir, fname), ann_path_png))
            elif os.path.exists(ann_path_jpg):
                pairs.append((os.path.join(img_split_dir, fname), ann_path_jpg))
                
    random.seed(42)
    random.shuffle(pairs)
    
    target_cracked = total_target // 2
    target_crack_free = total_target - target_cracked
    
    count_cracked = 0
    count_crack_free = 0
    
    for img_path, ann_path in tqdm(pairs):
        if count_cracked >= target_cracked and count_crack_free >= target_crack_free:
            break
            
        try:
            img = Image.open(img_path).convert('RGB')
            ann = Image.open(ann_path).convert('L')
            
            w, h = img.size
            if w < patch_size or h < patch_size:
                continue
                
            # Randomly sample patches per image
            for _ in range(10):
                if count_cracked >= target_cracked and count_crack_free >= target_crack_free:
                    break
                    
                x = random.randint(0, w - patch_size)
                y = random.randint(0, h - patch_size)
                
                img_patch = img.crop((x, y, x + patch_size, y + patch_size))
                ann_patch = ann.crop((x, y, x + patch_size, y + patch_size))
                
                ann_np = np.array(ann_patch)
                is_cracked = np.max(ann_np) > 0
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                patch_name = f"{base_name}_{x}_{y}.jpg"
                
                if is_cracked and count_cracked < target_cracked:
                    img_patch.save(os.path.join(dest_cracked, patch_name))
                    count_cracked += 1
                elif not is_cracked and count_crack_free < target_crack_free:
                    img_patch.save(os.path.join(dest_crack_free, patch_name))
                    count_crack_free += 1
                    
        except Exception as e:
            pass

    print(f"Saved {count_cracked} cracked patches and {count_crack_free} crack-free patches.")

if __name__ == "__main__":
    src_images = "/home/hieulc/avitech_13/omnicrack30k_data/images"
    src_annotations = "/home/hieulc/avitech_13/omnicrack30k_data/annotations"
    dest_cracked = "/home/hieulc/avitech_13/cycle_crack/data/cracked"
    dest_crack_free = "/home/hieulc/avitech_13/cycle_crack/data/crack_free"
    
    prepare_cycle_crack_patches(src_images, src_annotations, dest_cracked, dest_crack_free, total_target=3000)
