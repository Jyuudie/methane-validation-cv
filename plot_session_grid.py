import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

# ================= CONFIGURATION =================
input_csv = "output/Master_Herd_Feeding_Index.csv"
START_FRAME = 0
END_FRAME = 13000
COLS_PER_ROW = 2
LINE_COLOR = '#000080'
# =================================================

def plot_individual_sessions():
    try:
        df = pd.read_csv(input_csv, index_col=0)
    except FileNotFoundError:
        print("Error: Master CSV not found.")
        return

    # Filter Frame Range
    df = df.loc[START_FRAME:END_FRAME]
    
    # Normalize Data (0 to 1 scaling for comparison)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Setup Grid
    num_plots = len(df_scaled.columns)
    num_rows = math.ceil(num_plots / COLS_PER_ROW)
    
    fig, axes = plt.subplots(num_rows, COLS_PER_ROW, figsize=(15, 4 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, col_name in enumerate(df_scaled.columns):
        ax = axes[i]
        ax.plot(df_scaled.index, df_scaled[col_name], color=LINE_COLOR, linewidth=1.5)
        
        # Invert Y so "Down" (1.0) is visually at the bottom
        ax.invert_yaxis()

        # Add "Almasi Window" Overlay (Green Box)
        ax.axvspan(2700, 12300, color='green', alpha=0.05)
        
        ax.set_title(f"Session: {col_name}", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Head Pos (Norm)")

    plt.tight_layout()
    plt.savefig("All_Sessions_Grid.png", dpi=300)
    print("Grid plot saved successfully.")

if __name__ == "__main__":
    plot_individual_sessions()
