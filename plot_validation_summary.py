import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ================= CONFIGURATION =================
input_csv = "output/Master_Herd_Feeding_Index.csv"
START_FRAME = 0
END_FRAME = 13000

# Industry Standard Sampling Window (Almasi et al., 2025)
# 90s to 410s @ 30fps
WINDOW_START = 2700
WINDOW_END = 12300
# =================================================

def plot_validation_graph():
    # 1. Load Data
    try:
        df = pd.read_csv(input_csv, index_col=0)
    except FileNotFoundError:
        print("Error: Master CSV not found.")
        return
    
    df = df.loc[START_FRAME:END_FRAME]
    
    # 2. Normalize Data (MinMax Scaling)
    # Essential for overlaying varying camera angles
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # 3. Calculate Global Average (The "Signal")
    df_scaled['Global_Avg'] = df_scaled.mean(axis=1)

    # 4. K-Means Thresholding
    # Unsupervised clustering to find the dynamic feeding line
    X = df_scaled['Global_Avg'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    threshold = kmeans.cluster_centers_.flatten().mean()
    print(f"Algorithmically Determined Threshold: y={threshold:.4f}")

    # ================= VISUALIZATION =================
    plt.figure(figsize=(14, 7))

    # A. The "Target Window" (Green Zone)
    plt.axvspan(WINDOW_START, WINDOW_END, color='green', alpha=0.1, label='Almasi Window (90-410s)')

    # B. The "Spaghetti" (Individual Variability)
    video_cols = [c for c in df_scaled.columns if c != 'Global_Avg']
    for col in video_cols:
        plt.plot(df_scaled.index, df_scaled[col], color='gray', linewidth=1, alpha=0.3)

    # C. The "Global Signal" (Herd Behavior)
    plt.plot(df_scaled.index, df_scaled['Global_Avg'], color='#000080', linewidth=3, label='Global Herd Index')

    # D. The K-Means Threshold
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'K-Means Threshold (y={threshold:.2f})')

    # Styling
    plt.title(f'Biological Validation: Biphasic Feeding vs. Sampling Protocols')
    plt.xlabel('Frame Number (30fps)')
    plt.ylabel('Normalized Head Position (0=Up, 1=Down)')
    plt.gca().invert_yaxis()
    
    # Dummy plot for legend clarity
    plt.plot([], [], color='gray', linewidth=1, label='Individual Sessions')
    
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("Validation_Summary_Plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_validation_graph()
