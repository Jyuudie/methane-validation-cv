import pandas as pd
import glob
import os

# ================= CONFIGURATION =================
input_folder = "output/individual_tracks"  # Folder with CSVs created from extract_tracts.py
output_file = "output/Master_Herd_Feeding_Index.csv"
ROLLING_WINDOW = 30  # 1 second smoothing
# =================================================

def merge_and_analyze(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        print("Error: No CSV files found.")
        return

    print(f"Merging {len(all_files)} sessions...")
    master_df = pd.DataFrame()

    for filename in all_files:
        print(f"Processing: {os.path.basename(filename)}...")
        
        try:
            df = pd.read_csv(filename)
        except:
            continue

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        if 'frame' in df.columns and 'y' in df.columns:
            # 1. Average all cows in this specific frame (Spatial Mean)
            daily_trend = df.groupby('frame')['y'].mean()
            
            # 2. Apply Smoothing (Temporal Mean)
            smoothed_trend = daily_trend.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            
            # 3. Store in Master DataFrame using filename as column header
            col_name = os.path.basename(filename).replace(".csv", "")
            master_df[col_name] = smoothed_trend

    master_df = master_df.sort_index()
    master_df.to_csv(output_file)
    print(f"Success! Aggregated data saved to {output_file}")

if __name__ == "__main__":
    merge_and_analyze(input_folder)
