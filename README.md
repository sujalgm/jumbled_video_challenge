***Jumbled Frames Reconstruction — TECDIA Internship Challenge***

This repository contains my solution for the Jumbled Frames Reconstruction Challenge conducted by TECDIA.
The goal was to reconstruct a jumbled video (where all frames are shuffled) into its original, correct sequence using computer vision and similarity-based analysis.

The implementation was built entirely in Python, leveraging OpenCV, NumPy, and scikit-image.

**1. Project Overview**

The task was to design an algorithm that automatically reorders shuffled video frames into their natural sequence, producing a coherent reconstructed video.

As soon as I read the problem statement, I knew Python was the ideal language — it has been my primary tool since my first year of engineering and offers excellent support for image processing.
A friend had previously worked on a computer-vision project using OpenCV, which reminded me that cv2 would be perfect for handling frames efficiently.

The first logical step was to extract all video frames before attempting any ordering algorithm. Without frames, reconstruction is impossible.

**2. Step 1: Extracting Frames**

The process begins with extract_frames.py, which uses OpenCV to open the input video and extract each frame as a .png image.

Each frame is stored in a dedicated folder (frames/) like this:

frames/
 ├── frame_000.png
 ├── frame_001.png
 ├── frame_002.png
 ...


This forms the dataset of ~300 frames used for further analysis.

**3. Step 2: Building the Dissimilarity Matrix**

Next, I needed to quantify how different each frame is from every other frame — the dissimilarity matrix.
This is an N × N grid (where N = number of frames) storing the visual difference between each frame pair.

My first brute-force version worked but took over 12 minutes.
With 300 frames, that’s ~45,000 comparisons — computationally heavy even for a modern CPU.

To optimize:

Downscaled frames to 160×120 pixels (reduced workload, preserved structure).

Converted to grayscale before computing similarity.

Parallelized computations to utilize all CPU cores effectively.

This cut runtime significantly while maintaining excellent accuracy.

**4. Step 3: Choosing the Right Similarity Metric (SSIM)**

Different similarity metrics were evaluated:

Metric	Type	Accuracy	Speed	Comment
Mean Squared Error (MSE)	Pixel-wise	Low	Fast	Too sensitive to noise
Histogram Correlation	Color-based	Moderate	Fast	Ignores spatial structure
Structural Similarity Index (SSIM)	Structural	High	Moderate	Captures luminance, contrast & structure

I chose SSIM (Structural Similarity Index) because it models human visual perception and captures image structure much better than pixel-level metrics.

The script compute_matrix_ssim.py computes pairwise SSIM between nearby frames and stores
1 – SSIM as the dissimilarity score in dissimilarity_matrix.npy.

This matrix visually represents how closely each frame relates to others — forming the foundation for reconstruction.

**5. Step 4: Verifying the Matrix**

Before using the matrix, I validated it with verify_matrix.py, which checks:

   Correct shape and numeric range

   Symmetry (matrix[i][j] = matrix[j][i])

   Zero diagonal (each frame is identical to itself)

   This ensured clean, consistent data before reconstruction.

## 6. Step 5: Reconstructing the Video

Reconstructing required an algorithm that balances **accuracy**, **speed**, and **simplicity**.

| Algorithm | Approach | Time | Accuracy | Complexity |
|------------|-----------|-------|-----------|-------------|
| Random Shuffle | Baseline | Very Low | Very Low | Trivial |
| Greedy Nearest Neighbor | Local similarity | Fast | Good | Simple |
| 2-Way Greedy | Bidirectional refinement | Medium | Better | Moderate |
| TSP (Nearest Insertion) | Global optimization | Slow | Best | High |
| Graph DFS | Sequential flow | Medium | Fair | Moderate |

I selected the **Greedy SSIM-based ordering** because it produced near-perfect accuracy while being significantly faster than full TSP solvers.  
However, greedy orderings can occasionally yield a reversed sequence, so I added an **optical-flow-based direction validation** step using Farneback motion estimation.


**The script reconstruct.py:**

  Loads all frames

  Builds the similarity matrix

  Orders frames greedily based on SSIM closeness

  Compares forward vs reversed sequences and picks the smoother one

  Uses optical flow to validate motion direction

It saves:

outputs/
 ├── unjumbled_output.mp4
 ├── unjumbled_output_reversed.mp4
 ├── order.csv
 └── timing_log.json

**7. System Requirements**

**Hardware:**

CPU: Quad-core or higher

RAM: ≥ 8 GB (16 GB recommended)

**Software:**

Python 3.9 – 3.11
(Python 3.12+ or 3.14 may not yet have compatible NumPy wheels)

Tested on: Windows 11, macOS Sonoma, Ubuntu 22.04

**Dependencies:**

opencv-python
numpy
tqdm
scikit-image


**Install dependencies:**

pip install -r requirements.txt

**8. How to Run**
**Step 0 — Place Your Input Video**

Place the jumbled video file in the project root and name it:

jumbled_video.mp4


(Or update VIDEO_FILE in extract_frames.py.)

**Step 1 — Extract Frames**
python extract_frames.py

**Step 2 — Compute Dissimilarity Matrix**
python compute_matrix_ssim.py

**Step 3 — Verify Matrix**
python verify_matrix.py

**Step 4 — Reconstruct the Video**
python reconstruct.py --frames-dir frames --fps 30

**9. Key Learnings**

Downscaling + SSIM offered the best speed–accuracy balance.

Parallel computation fully utilized CPU resources.

Optical-flow validation eliminated reversed sequences.

Modular design simplified testing and debugging.

This project deepened my understanding of video analytics, image similarity metrics, and optimization.

**10. References**

Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE TIP, 2004

OpenCV Documentation

Scikit-Image Documentation

**11. Future Improvements**

GPU acceleration using CUDA-enabled OpenCV or CuPy

Hybrid similarity using SSIM + ORB feature matching

Real-time adaptation for live video reconstruction

**12. Execution Time Log (Example Output)**
{
  "load_seconds": 40.2311,
  "similarity_seconds": 213.6174,
  "greedy_order_seconds": 0.0145,
  "direction_choice": "forward",
  "direction_select_seconds": 0.0003,
  "write_video_seconds": 12.284,
  "total_seconds": 278.4051
}

**13. Unjumbled Video (Google Drive Link)**

https://drive.google.com/file/d/169dBJpBTTE_LanDsILmyCHnsC4J4CY6b/view?usp=sharing

Authored by

Sujal GM
For TECDIA Internship Challenge 2025
