import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================
csv_file = "output/cow_tracks.csv"
ROLLING_WINDOW = 30       # 30 frames = ~1 second smoothing
START_FRAME = 2000        
END_FRAME = 6000          
# =================================================

# Load and Filter Data
df = pd.read_csv(csv_file)
df = df[(df['frame'] >= START_FRAME) & (df['frame'] <= END_FRAME)]

# Calculate Herd Index (Mean Y-position of all visible cows)
herd_data = df.groupby('frame')['y'].mean().reset_index()

# Apply Smoothing
herd_data['y_smooth'] = herd_data['y'].rolling(window=ROLLING_WINDOW).mean()

# Visualization
plt.figure(figsize=(15, 6))
plt.style.use('ggplot')

sns.lineplot(data=herd_data, x='frame', y='y_smooth', color='darkblue', linewidth=2)

plt.title("Herd Feeding Index (Average Head Position)", fontsize=16)
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Inverted Y-Centroid (Pixels)", fontsize=12)
plt.gca().invert_yaxis() # Invert so "Down" = "Feeding"

plt.axhline(y=350, color='red', linestyle='--', alpha=0.5, label='Feeding Horizon') #threshold is based on estimatation via "find_threshold" script
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
