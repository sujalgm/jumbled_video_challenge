***Jumbled Frames Reconstruction — TECDIA Internship Challenge***

This repository contains my solution for the Jumbled Frames Reconstruction Challenge conducted by TECDIA.
The task was to reconstruct a jumbled video — where all frames are shuffled — back into its correct order using computer vision and similarity-based analysis.

The entire project was developed in Python, mainly using OpenCV, NumPy, and scikit-image.

**1. Project Overview**

The goal was to design an algorithm that can automatically rearrange shuffled video frames to restore the original sequence and create a smooth, logical video again.

As soon as I read the problem, I was sure that Python was the right language for this.
I’ve been using Python since my first year of engineering, and it’s perfect for handling image and video data.
A friend of mine had once worked on a project involving computer vision with OpenCV — that gave me the idea to use cv2 to read, analyze, and process the frames efficiently.

The first step was obvious — I had to extract all the individual frames from the given video before doing any kind of ordering or reconstruction.

**2. Step 1: Extracting Frames**

The process starts with extract_frames.py, which uses OpenCV to read the video and save each frame as a .png image inside a folder called frames/.

The folder looks like this:

```
frames/
 ├── frame_000.png
 ├── frame_001.png
 ├── frame_002.png
 ...
```


This gave me around 300 frames to work with — these became the base dataset for further analysis.

**3. Step 2: Building the Dissimilarity Matrix**

Once I had the frames, the next challenge was figuring out how “different” one frame is from another.
To do that, I built a dissimilarity matrix, which is basically an N × N grid (where N = number of frames) that measures the visual difference between every pair of frames.

At first, my brute-force version worked, but it took more than ~2hours to complete — since comparing 300 frames means almost 45,000 comparisons!

So I optimized it by:

Downscaling each frame to 160×120 pixels to reduce computation while keeping key details.

Converting frames to grayscale before comparison.

Using parallel processing to make full use of the CPU.

This brought the runtime down significantly without losing accuracy.

**4. Step 3: Choosing the Right Similarity Metric (SSIM)**

I tested a few different similarity metrics before finalizing one:

| Metric | Type | Accuracy | Speed | Comment |
|---------|------|-----------|--------|----------|
| Mean Squared Error (MSE) | Pixel-based | Low | Fast | Very sensitive to noise |
| Histogram Correlation | Color-based | Moderate | Fast | Ignores spatial structure |
| **Structural Similarity Index (SSIM)** | Structural | High | Moderate | Captures luminance, contrast, and structural details |

I finally chose SSIM (Structural Similarity Index) because it’s much closer to how humans perceive image similarity — it focuses on structure and texture, not just color.

The file compute_matrix_ssim.py calculates 1 - SSIM for each frame pair and saves the results in dissimilarity_matrix.npy.

This matrix became the foundation for figuring out the right order of frames.

**5. Step 4: Verifying the Matrix**

Before reconstruction, I used verify_matrix.py to make sure the dissimilarity matrix was valid.

It checks:

Whether the shape and values are correct

If it’s symmetric (matrix[i][j] = matrix[j][i])

If the diagonal values are zero (each frame is identical to itself)

This helped ensure that the reconstruction algorithm would start with clean, consistent data.

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


The main reconstruction script (reconstruct.py) does:

    Loads all frames

    Builds the SSIM-based similarity matrix

    Orders frames greedily based on their closeness

    Checks both forward and reverse directions

    Uses optical flow to confirm the correct direction

It produces the following outputs:

```
outputs/
 ├── unjumbled_output.mp4
 ├── unjumbled_output_reversed.mp4
 ├── order.csv
 └── timing_log.json
```

**7. System Requirements**

**Hardware:**

CPU: Quad-core or higher

RAM: At least 8 GB (16 GB recommended)

**Software:**

Python 3.9 to 3.11
(Python 3.12+ or 3.14 may cause dependency issues with NumPy)

**Tested On:**

Windows 11

macOS Sonoma

Ubuntu 22.04

**Dependencies:**
```

opencv-python
numpy
tqdm
scikit-image
```


**Install all dependencies with:**

pip install -r requirements.txt


Note (Installation Tip):
This project has been verified on Python 3.10 and 3.11.
If you face installation errors (especially NumPy build issues), create a virtual environment with Python 3.10 or 3.11 and run:
```
py -3.11 -m venv venv
venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

These versions are fully compatible and will install everything smoothly.

**8. How to Run**
**Step 0 — Place Your Input Video**

Put your jumbled video file in the project root and name it:

jumbled_video.mp4


(Or update VIDEO_FILE in extract_frames.py.)

**Step 1 — Extract Frames**
```
python extract_frames.py
```

**Step 2 — Compute Dissimilarity Matrix**
```
python compute_matrix_ssim.py
```

**Step 3 — Verify Matrix**
```
python verify_matrix.py
```

**Step 4 — Reconstruct the Video**
```
python reconstruct.py --frames-dir frames --fps 30
```
**9. Key Learnings**

Downscaling + SSIM gave the best balance between speed and accuracy.

Using multiple CPU cores improved performance drastically.

Optical flow validation solved the reversed video issue.

The modular design made testing and debugging much easier.

This project helped me deeply understand video analytics, image similarity metrics, and performance optimization in computer vision tasks.

**10. References**

Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE TIP, 2004.

OpenCV Documentation

Scikit-Image Documentation

**11. Future Improvements**

GPU acceleration using CUDA-enabled OpenCV or CuPy

Combining SSIM + ORB feature matching for better accuracy

Extending to handle real-time video reconstruction

**12. Example Execution Time Log**
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


**Authored by:**
```
Sujal GM
```
**for TECDIA Internship Challenge**
