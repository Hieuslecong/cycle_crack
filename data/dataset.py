import os
from PIL import Image
from torch.utils.data import Dataset
import random

def is_image_file(filename):
    """Check if the filename has a valid image extension"""
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp'])

def make_dataset(dir_path):
    """Scan and return all image files in a directory"""
    images = []
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return images
        
    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class UnpairedCrackDataset(Dataset):
    """Unpaired loader for Cycle-Crack providing images from Domains N (normal) and C (cracked).
    
    Domain C: patches with >100 crack pixels (paper §IV-A).
    Domain N: patches with 0 crack pixels.
    Ambiguous (1-100 crack pixels) are discarded during data preparation.
    """
    
    def __init__(self, crack_dir, normal_dir, transform=None):
        """
        Args:
            crack_dir (str): Path to domain C images
            normal_dir (str): Path to domain N images
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.crack_dir = crack_dir
        self.normal_dir = normal_dir
        self.transform = transform

        self.crack_images = make_dataset(crack_dir)
        self.normal_images = make_dataset(normal_dir)

        if len(self.crack_images) == 0:
            print(f"Warning: No images found in crack directory {crack_dir}")
        if len(self.normal_images) == 0:
            print(f"Warning: No images found in normal directory {normal_dir}")

    def __len__(self):
        """Dataset length based on the largest domain size for complete traversal per epoch"""
        if len(self.crack_images) == 0 or len(self.normal_images) == 0:
            return 0
        return max(len(self.crack_images), len(self.normal_images))

    def __getitem__(self, idx):
        """Independently sample from both domains to maintain 'unpaired' properties"""
        crack_idx = idx % len(self.crack_images)
        # Randomize the normal_idx to prevent memorized pairs
        normal_idx = random.randint(0, len(self.normal_images) - 1)
        
        crack_path = self.crack_images[crack_idx]
        normal_path = self.normal_images[normal_idx]

        crack_img = Image.open(crack_path).convert('RGB')
        normal_img = Image.open(normal_path).convert('RGB')

        if self.transform is not None:
            crack_img = self.transform(crack_img)
            normal_img = self.transform(normal_img)

        return {'crack': crack_img, 'normal': normal_img, 'crack_path': crack_path, 'normal_path': normal_path}
