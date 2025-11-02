import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import time
from tqdm import tqdm

# --- CONFIG ---
folder = "frames"  # folder containing extracted frames
output_file = "dissimilarity_matrix.npy"
window_size = 100  # can tweak this
resize_dim = (160, 120)  # smaller = faster
# ----------------

def load_frames(folder):
    frame_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                          if f.endswith(('.jpg', '.png'))])
    frames = []
    for f in tqdm(frame_files, desc="Loading frames"):
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.resize(img, resize_dim)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img_gray)
    return frames, frame_files

def compute_ssim_matrix(frames, window_size):
    n = len(frames)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n)):
        for j in range(i+1, min(i+window_size, n)):
            score = ssim(frames[i], frames[j])
            dist = 1 - score  # dissimilarity
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix

if __name__ == "__main__":
    start = time.time()
    frames, frame_files = load_frames(folder)
    print(f"Total frames: {len(frames)}, Window size: {window_size}")

    matrix = compute_ssim_matrix(frames, window_size)
    np.save(output_file, matrix)

    print("\n--- Matrix Computation Complete ---")
    print(f"Time taken: {time.time() - start:.2f} seconds")
    print(f"Saved to: {output_file}")
