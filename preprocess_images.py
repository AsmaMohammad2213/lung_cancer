import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm

# ------------ Parameters ------------
INPUT_DIR = "dataset"          # input dataset folder
OUTPUT_DIR = "processed"       # processed images saved here
IMG_EXT = ("*.jpg", "*.jpeg", "*.png")

# Rolling Guidance Filter parameters
RGF_ITER = 3
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Switched Median Filter parameters
MEDIAN_KSIZE = 3
VAR_THRESHOLD = 500.0  # local variance threshold (tune)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def rolling_guidance_filter(img, iterations=3, d=9, sigmaColor=75, sigmaSpace=75):
    """Iteratively apply bilateral filter."""
    img_f = img.copy()
    for _ in range(iterations):
        img_f = cv2.bilateralFilter(img_f, d, sigmaColor, sigmaSpace)
    return img_f

def switched_median_filter(img_gray, ksize=3, var_threshold=500.0):
    """Apply switched median filter on grayscale image."""
    padded = cv2.copyMakeBorder(img_gray, ksize//2, ksize//2, ksize//2, ksize//2, cv2.BORDER_REFLECT)
    out = img_gray.copy()
    h, w = img_gray.shape
    for i in range(h):
        for j in range(w):
            win = padded[i:i+ksize, j:j+ksize]
            if win.var() > var_threshold:
                out[i, j] = np.median(win)
    return out

def preprocess_image(path, save_path):
    img = cv2.imread(path)
    if img is None:
        return False
    rgf = rolling_guidance_filter(img)
    gray = cv2.cvtColor(rgf, cv2.COLOR_BGR2GRAY)
    smf = switched_median_filter(gray, ksize=MEDIAN_KSIZE, var_threshold=VAR_THRESHOLD)
    ycrcb = cv2.cvtColor(rgf, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = smf
    fused = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, fused)
    return True

def mirror_structure_and_process(input_dir, output_dir):
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for cls in classes:
        in_dir = os.path.join(input_dir, cls)
        out_dir = os.path.join(output_dir, cls)
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for ext in IMG_EXT:
            files.extend(glob(os.path.join(in_dir, ext)))
        for f in tqdm(files, desc=f"Processing {cls}"):
            fname = os.path.basename(f)
            save_path = os.path.join(out_dir, fname)
            preprocess_image(f, save_path)

if __name__ == "__main__":
    mirror_structure_and_process(INPUT_DIR, OUTPUT_DIR)
    print("✅ Preprocessing complete. Processed images saved in:", OUTPUT_DIR)
