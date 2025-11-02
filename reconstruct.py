#!/usr/bin/env python3
import os
import numpy as np
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def tic():
    return time.perf_counter()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_frame_paths(frames_dir: Path, exts=(".png", ".jpg", ".jpeg")):
    files = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    # The names are already jumbled; we just need a stable order to reference them
    files.sort()
    return files

# ------------------------------------------------------------
# Frame I/O
# ------------------------------------------------------------
def load_frames(frames_dir: Path):
    paths = list_frame_paths(frames_dir)
    frames = []
    for p in tqdm(paths, desc="Loading frames"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        frames.append((p.name, img))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")
    return frames

# ------------------------------------------------------------
# Similarity
# ------------------------------------------------------------
def prep_gray_small(img, small_size=(160, 90)):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, small_size, interpolation=cv2.INTER_AREA)
    return g

def build_similarity_matrix(frames, small_size=(160, 90)):
    """
    Pairwise SSIM on small grayscale versions for speed.
    Returns a symmetric NxN matrix with 1.0 on the diagonal.
    """
    n = len(frames)
    smalls = [prep_gray_small(frames[i][1], small_size) for i in range(n)]
    sim = np.zeros((n, n), dtype=np.float32)

    for i in tqdm(range(n), desc="Building similarity matrix"):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            score = ssim(smalls[i], smalls[j], data_range=255)
            sim[i, j] = score
            sim[j, i] = score
    return sim


def estimate_motion_direction(frames, order, sample_step=10):
    """
    Use optical flow to estimate whether the sequence runs forward or backward.
    Returns +1 for forward, -1 for backward.
    """
    total_dx = 0
    total_dy = 0
    for i in range(0, len(order) - 1, sample_step):
        idx1, idx2 = order[i], order[i + 1]
        prev_gray = cv2.cvtColor(frames[idx1][1], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[idx2][1], cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (160, 90))
        next_gray = cv2.resize(next_gray, (160, 90))
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                            pyr_scale=0.5, levels=1, winsize=15,
                                            iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
        dx, dy = np.mean(flow[..., 0]), np.mean(flow[..., 1])
        total_dx += dx
        total_dy += dy
    avg_motion = np.sqrt(total_dx**2 + total_dy**2)
    if abs(total_dx) + abs(total_dy) < 1e-5:
        return +1  # no clear motion → keep as forward
    # If optical flow vectors mostly point backward, invert
    return +1 if total_dx + total_dy > 0 else -1


# ------------------------------------------------------------
# Ordering (Greedy) + Direction Selection
# ------------------------------------------------------------
def choose_start(sim):
    """
    Pick a start node that is less likely to be in the middle:
    we use the node with the smallest 'max similarity to others'.
    """
    n = sim.shape[0]
    max_to_others = np.max(sim + np.eye(n, dtype=sim.dtype) * -np.inf, axis=1)  # ignore self
    start = int(np.argmin(max_to_others))
    return start

def greedy_order(sim):
    """
    Build a path by always picking the most similar unused next frame.
    """
    n = sim.shape[0]
    used = np.zeros(n, dtype=bool)

    start = choose_start(sim)
    order = [start]
    used[start] = True

    for _ in tqdm(range(n - 1), desc="Ordering frames"):
        last = order[-1]
        # pick best unused next
        row = sim[last].copy()
        row[used] = -1.0  # mask used
        nxt = int(np.argmax(row))
        if row[nxt] <= 0:
            # fallback: anything unused (should rarely happen)
            nxt = int(np.where(~used)[0][0])
        order.append(nxt)
        used[nxt] = True

    return order

def path_smoothness(sim, order):
    """
    Sum of consecutive SSIM scores along a path.
    Uses the precomputed similarity matrix (fast).
    """
    total = 0.0
    for i in range(len(order) - 1):
        total += float(sim[order[i], order[i + 1]])
    return total

def pick_direction(sim, order):
    """
    Compare forward vs reversed total smoothness and choose the better one.
    Prevents hardcoding a reversal while guaranteeing best continuity.
    """
    fwd = path_smoothness(sim, order)
    rev_order = order[::-1]
    rev = path_smoothness(sim, rev_order)
    if rev > fwd:
        return rev_order, "reverse"
    return order, "forward"

# ------------------------------------------------------------
# Video Writing
# ------------------------------------------------------------
def write_video(frames, order, out_path: Path, fps=30):
    h, w = frames[0][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for idx in tqdm(order, desc="Writing video"):
        vw.write(frames[idx][1])
    vw.release()

def save_order_csv(frames, order, out_csv: Path):
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("position,frame_index,filename\n")
        for pos, idx in enumerate(order):
            f.write(f"{pos},{idx},{frames[idx][0]}\n")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reconstruct an unjumbled video from shuffled frames.")
    parser.add_argument("--frames-dir", type=str, default="frames", help="Directory with frames (.png/.jpg)")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--small-w", type=int, default=160, help="Downscaled width for SSIM")
    parser.add_argument("--small-h", type=int, default=90, help="Downscaled height for SSIM")
    parser.add_argument("--out", type=str, default="unjumbled_output.mp4", help="Output video filename")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to save outputs")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_video = out_dir / args.out
    out_csv = out_dir / "order.csv"
    out_timing = out_dir / "timing_log.json"

    timing = {}
    t0 = tic()

    # 1️⃣ Load frames
    t = tic()
    frames = load_frames(frames_dir)
    timing["load_seconds"] = round(tic() - t, 4)
    print(f"Total frames loaded: {len(frames)}")

    # 2️⃣ Build similarity matrix (SSIM)
    t = tic()
    sim = build_similarity_matrix(frames, small_size=(args.small_w, args.small_h))
    timing["similarity_seconds"] = round(tic() - t, 4)

    # 3️⃣ Greedy frame ordering
    t = tic()
    order = greedy_order(sim)
    timing["greedy_order_seconds"] = round(tic() - t, 4)

    # 4️⃣ Automatic direction selection (forward vs reverse)
    t = tic()
    order, chosen_dir = pick_direction(sim, order)
    timing["direction_choice"] = chosen_dir
    timing["direction_select_seconds"] = round(tic() - t, 4)
    print(f"Chosen direction (SSIM-based): {chosen_dir}")

    # 4️⃣.5 Optical flow validation for motion direction
    print("\nAnalyzing optical flow direction (motion-based validation)...")
    motion_sign = estimate_motion_direction(frames, order)
    if motion_sign < 0:
        print("🔁 Optical flow indicates reversed motion → flipping order.")
        order = order[::-1]
        chosen_dir += "_optical_reversed"
    else:
        print("✅ Optical flow direction consistent → keeping order.")

    # 5️⃣ Save order to CSV
    save_order_csv(frames, order, out_csv)

    # 6️⃣ Write reconstructed video
    t = tic()
    write_video(frames, order, out_video, fps=args.fps)
    timing["write_video_seconds"] = round(tic() - t, 4)

    # 6️⃣.b Save a reversed version too (for verification and safety)
    reversed_video = out_dir / "unjumbled_output_reversed.mp4"
    print("Also saving reversed version for manual verification...")
    rev_order = order[::-1]
    write_video(frames, rev_order, reversed_video, fps=args.fps)


    # 7️⃣ Save timing log
    timing["total_seconds"] = round(tic() - t0, 4)
    with open(out_timing, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    # 8️⃣ Summary
    print("\n--- DONE ---")
    print(f"Video saved to: {out_video}")
    print(f"Order CSV saved to: {out_csv}")
    print(f"Timing log saved to: {out_timing}")
    print(f"Final chosen direction: {chosen_dir}")


if __name__ == "__main__":
    main()
