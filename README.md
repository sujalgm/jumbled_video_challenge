***Jumbled Frames Reconstruction – TECDIA Internship Challenge***

This repository contains my solution for the Jumbled Frames Reconstruction Challenge conducted by TECDIA.
The goal was to reconstruct a jumbled video (where all frames are out of order) into its original, correct sequence using computer vision and intelligent similarity-based analysis.
The entire implementation was built in Python using OpenCV, NumPy, and scikit-image.

**1. Project Overview**

The objective was to design an algorithm that could automatically reorder shuffled video frames back into their natural sequence, producing a coherent reconstructed video.

As soon as I saw the problem, I knew I would use Python—it has been my primary language since my first year of engineering and offers strong support for image processing.
A friend of mine had earlier worked on a computer-vision project using OpenCV, and that reminded me that cv2 would be ideal for handling image data and frame operations.

The very first step I understood was that I needed to extract every frame from the given jumbled video before performing any reconstruction. Without this, no algorithmic ordering could be attempted.

**2. Step 1: Extracting Frames**

The process began with a custom Python script named extract_frames.py.
This script uses OpenCV to open the video file (jumbled_video.mp4) and save each frame sequentially as individual image files.

Each frame is stored in a dedicated folder (frames/) as:

frames/
 ├── frame_000.png
 ├── frame_001.png
 ├── frame_002.png
 ...


This gave me a structured dataset of roughly 300 frames to work with for further analysis.

**3. Step 2: Building the Dissimilarity Matrix**

Once all frames were available, I needed a way to quantify how different each frame is from every other—a dissimilarity matrix.
This matrix is an 
𝑁
×
𝑁
N×N grid (where N = number of frames) whose entries measure visual differences between pairs of frames.

My first attempt worked but took more than 12 minutes to complete. I realized that comparing every frame pair (300 × 299 / 2 ≈ 45 000 comparisons) was very time-consuming.
Running them sequentially or even with basic threading still left most of the CPU under-utilized.

To optimize:

I downscaled each frame to 160 × 120 pixels, drastically reducing computation per comparison while preserving structure.

I converted frames to grayscale before computing similarity.

I implemented parallel processing to utilize all CPU cores effectively.

This combination maintained high accuracy while cutting runtime by several minutes.

**4. Step 3: Choosing the Right Similarity Metric (SSIM)**

Various similarity metrics were considered:

Metric	Type	Accuracy	Speed	Comment
Mean Squared Error (MSE)	Pixel-wise	Low	Fast	Sensitive to noise
Histogram Correlation	Color-based	Moderate	Fast	Misses spatial structure
Structural Similarity Index (SSIM)	Structural	High	Moderate	Captures luminance, contrast & structure

I chose SSIM (Structural Similarity Index) from skimage.metrics because it models human visual perception and provides a more reliable notion of image similarity.
The script compute_matrix_ssim.py calculates pairwise SSIM for nearby frames and stores
1 – SSIM as the dissimilarity score in dissimilarity_matrix.npy.

This produced a clear visual “map” of how each frame relates to its neighbors.

**5. Step 4: Verifying the Matrix**

Before using the matrix, I verified its integrity with verify_matrix.py, which:

    Checks the matrix shape and numeric range.

    Confirms it is symmetric.

    Ensures diagonal entries are zero (each frame identical to itself).

    This verification helped catch inconsistencies early and guaranteed that the reconstruction stage started with clean data.

**6. Step 5: Reconstructing the Video**

Reconstructing the sequence required choosing an ordering algorithm that balances accuracy and speed. I evaluated several:

Algorithm	Approach	Time	Accuracy	Complexity
Random Shuffle	Baseline	Very Low	Very Low	Trivial
Greedy Nearest Neighbor	Local similarity	Fast	Good	Simple
2-Way Greedy	Bidirectional refinement	Medium	Better	Moderate
TSP (Nearest Insertion)	Global optimization	Slow	Best	High
Graph DFS Traversal	Sequential flow	Medium	Fair	Moderate

I selected the Greedy SSIM-based ordering because it produced near-perfect accuracy with far less computation than full TSP solvers.
However, greedy ordering alone can occasionally yield a reversed sequence.
To fix that, I added an optical-flow-based direction-validation step using Farneback motion estimation.

**All of this logic resides in reconstruct.py, which:**

    Loads all frames from frames/.

    Builds the similarity matrix.

    Orders frames greedily by SSIM closeness.

    Evaluates both forward and reverse directions and keeps the smoother one.

    Validates direction using optical flow.

**Saves:**

  unjumbled_output.mp4

  unjumbled_output_reversed.mp4 (for manual cross-check)

  order.csv (final frame order)

  timing_log.json (execution times)

**7. Requirements and Dependencies**

System Requirements

CPU : Quad-core or better

RAM : ≥ 8 GB (16 GB recommended)

Python 3.9 or newer

Tested on Windows 11, macOS Sonoma, Ubuntu 22.04

**Python packages required:**
opencv-python
numpy
tqdm
scikit-image

**install dependencies:**
pip install -r requirements.txt

***8. How to Run***

**Step 1 – Extract frames:
python extract_frames.py**

**Step 2 – Compute dissimilarity matrix:
python compute_matrix_ssim.py**

**Step 3 – Verify matrix:
python verify_matrix.py**

**Step 4 – Reconstruct the video:
python reconstruct.py --frames-dir frames --fps 30**

**Outputs will appear in:
outputs/
 ├── unjumbled_output.mp4
 ├── unjumbled_output_reversed.mp4
 ├── order.csv
 └── timing_log.json**

**9. Key Learnings and Observations**

Downscaling combined with SSIM provided the best trade-off between runtime and accuracy.

Parallel computation improved speed by utilizing all CPU cores.

Optical-flow validation eliminated reversed playback errors.

Modular design simplified debugging and incremental testing.

This project strengthened my understanding of video analytics, image-similarity metrics, and computational optimization.

**10. References**

Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Transactions on Image Processing, 2004.

OpenCV Documentation (https://docs.opencv.org
)

Scikit-Image Documentation (https://scikit-image.org
)

**11. Future Improvements and Next Steps**

While the current version performs efficiently and gives accurate reconstruction results, there are still a few ways it can be improved:

GPU-based Acceleration:
Using CUDA-enabled OpenCV or CuPy could reduce computation time for SSIM and optical-flow steps.

Hybrid Similarity Approach:
Combining SSIM with lightweight feature matching (like ORB) could handle scenes with higher motion or lighting variation.

Real-Time Extension:
Adapting the algorithm for continuous or live video input could make it suitable for real-time applications such as surveillance.


## Execution Time Log
This log was automatically generated after successful reconstruction:
{
  "load_seconds": 40.2311,
  "similarity_seconds": 213.6174,
  "greedy_order_seconds": 0.0145,
  "direction_choice": "forward",
  "direction_select_seconds": 0.0003,
  "write_video_seconds": 12.284,
  "total_seconds": 278.4051
}

**## Unjumbled video google drive link**

https://drive.google.com/file/d/169dBJpBTTE_LanDsILmyCHnsC4J4CY6b/view?usp=sharing

Authored by 
SUJAL GM 
for TECDIA Internship Challenge - 2025
