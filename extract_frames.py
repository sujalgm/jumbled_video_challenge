import cv2
import os
import time

# --- Configuration ---
VIDEO_FILE = "jumbled_video.mp4"
OUTPUT_DIR = "frames"
# ---------------------

def extract_frames():
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    
    # Open the video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_FILE}")
        return

    frame_count = 0
    start_time = time.time()
    
    print(f"Starting frame extraction from {VIDEO_FILE}...")

    while True:
        # Read one frame
        ret, frame = cap.read()
        
        # If 'ret' is False, it means we've reached the end of the video
        if not ret:
            break
        
        # Define the output filename, e.g., "frames/frame_001.png"
        # We use zfill(3) to pad with zeros (001, 002, ..., 300)
        frame_filename = os.path.join(OUTPUT_DIR, f"frame_{str(frame_count).zfill(3)}.png")
        
        # Save the frame as a PNG image
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
        
        # Optional: Print progress
        if frame_count % 30 == 0:
            print(f"Extracted {frame_count} frames...")

    # Release the video capture object
    cap.release()
    
    end_time = time.time()
    print(f"\n--- Extraction Complete ---")
    print(f"Total frames extracted: {frame_count}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    extract_frames()