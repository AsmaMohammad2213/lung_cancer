import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "processed"      # your preprocessed CT images
OUTPUT_DIR = "masks_auto"    # auto-generated masks will be saved here
IMG_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_mask(img):
    """Generate a basic lung mask using thresholding + morphology."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold to separate lung area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert mask if lungs appear darker (typical in CT)
    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)
    # Morphological cleaning
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=3)
    return mask

def generate_all(input_dir, output_dir):
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for cls in classes:
        in_dir = os.path.join(input_dir, cls)
        out_dir = os.path.join(output_dir, cls)
        os.makedirs(out_dir, exist_ok=True)
        for fname in tqdm(os.listdir(in_dir), desc=f"Generating masks for {cls}"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(in_dir, fname)
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SIZE)
                mask = create_mask(img)
                save_path = os.path.join(out_dir, fname)
                cv2.imwrite(save_path, mask)

if __name__ == "__main__":
    generate_all(INPUT_DIR, OUTPUT_DIR)
    print("✅ Auto mask generation complete! Masks saved in 'masks_auto/'")
