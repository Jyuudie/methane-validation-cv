import cv2
import os
import glob

# ================= CONFIGURATION =================
video_path = "raw_footage.mp4"       
annotation_folder = r"C:\path\to\cvat_exports"  
output_folder = "reconstructed_dataset"
# =================================================

os.makedirs(output_folder, exist_ok=True)

# 1. Load all the Frame Numbers we need into a list
print("Scanning annotation files...")
txt_files = glob.glob(os.path.join(annotation_folder, "*.txt"))

needed_frames = set() # Using a 'set' is faster for checking
file_map = {}         # To remember which txt file goes with which frame

for txt_path in txt_files:
    filename = os.path.basename(txt_path)
    try:
        # Extract number from "frame_000015.txt"
        frame_id = int(filename.replace("frame_", "").replace(".txt", ""))
        needed_frames.add(frame_id)
        file_map[frame_id] = txt_path
    except ValueError:
        pass

if not needed_frames:
    print("Error: No valid .txt files found! Check your folder path.")
    exit()

max_frame_needed = max(needed_frames)
print(f"Found {len(needed_frames)} annotations.")
print(f"Highest frame needed is: {max_frame_needed}")

# 2. Open Video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video. Check video_path spelling!")
    exit()

total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video Length: {total_video_frames} frames.")

if total_video_frames < max_frame_needed:
    print(" WARNING: Your text files ask for frames that don't exist!")
    print(f"   The video ends at {total_video_frames}, but you want Frame {max_frame_needed}.")
    print("   Are you sure this is the right video file?")

# 3. SEQUENTIAL PROCESSING (Safe Mode)
print("\nStarting extraction (Scanning video from start)...")
current_frame = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video
    
    # Check if we need this specific frame
    if current_frame in needed_frames:
        
        # Save Image
        image_name = f"frame_{current_frame:06d}.jpg"
        cv2.imwrite(os.path.join(output_folder, image_name), frame)
        
        # Save Text File (Copy it over)
        txt_source = file_map[current_frame]
        txt_dest_name = f"frame_{current_frame:06d}.txt"
        
        with open(txt_source, 'r') as f_in:
            data = f_in.read()
        with open(os.path.join(output_folder, txt_dest_name), 'w') as f_out:
            f_out.write(data)
            
        saved_count += 1
        print(f"Saved {saved_count}/{len(needed_frames)} images...", end='\r')

    # Optimization: Stop if we passed the last frame we need
    if current_frame > max_frame_needed:
        print("\nReached the last needed frame. Stopping early.")
        break

    current_frame += 1

cap.release()
print(f"\nDONE! Successfully extracted {saved_count} pairs.")

input("Press Enter to close...")
