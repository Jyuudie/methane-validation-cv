import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================
INPUT_CSV = "output/Master_Herd_Feeding_Index.csv"
# =================================================

def plot_histogram():
    df = pd.read_csv(INPUT_CSV, index_col=0)
    
    # Flatten all data into one big list of head positions
    all_positions = df.values.flatten()
    
    plt.figure(figsize=(10, 6))
    
    # Plot Histogram with Density Curve
    sns.histplot(all_positions, bins=50, kde=True, color='#000080', alpha=0.6)
    
    plt.title("Bimodal Distribution of Herd Head Positions")
    plt.xlabel("Vertical Position (y)")
    plt.ylabel("Frequency (Count)")
    plt.grid(True, alpha=0.3)
    
    # Add labels for the two "Humps"
    plt.text(x=300, y=100, s="State 1: Idling (Head Up)", fontsize=10, color='red')
    plt.text(x=800, y=100, s="State 2: Feeding (Head Down)", fontsize=10, color='green')
    
    plt.savefig("Bimodal_Distribution.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_histogram()
