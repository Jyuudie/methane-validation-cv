import os
#
# ================= CONFIGURATION =================
# Path to folder containing empty background images (Negative Samples)
FOLDER_PATH = "data/background_images"
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
# =================================================

def generate_empty_annotations():
    if not os.path.exists(FOLDER_PATH):
        print(f"Error: Folder '{FOLDER_PATH}' not found.")
        return

    count = 0
    files = os.listdir(FOLDER_PATH)
    
    print(f"Scanning {FOLDER_PATH} for images...")
    
    for filename in files:
        if filename.lower().endswith(IMAGE_EXTS):
            # Create a matching .txt file
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(FOLDER_PATH, txt_filename)
            
            # Only create if it doesn't exist to prevent overwriting
            if not os.path.exists(txt_path):
                with open(txt_path, 'w') as f:
                    pass # Create empty file (indicates "No Object" to YOLO)
                count += 1
                print(f"Generated negative label: {txt_filename}")
            else:
                print(f"Skipped (Exists): {txt_filename}")

    print(f"\nSuccess! Created {count} negative sample labels.")

if __name__ == "__main__":
    generate_empty_annotations()
