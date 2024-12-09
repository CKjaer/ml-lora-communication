import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == "__main__":
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
    plt.rcParams['font.family'] = 'Palatino Linotype'
    fs = 20
    plt.rcParams.update({'font.size': fs})
    
    filepath = os.path.abspath(__file__)
    directory = os.path.abspath(os.path.join(filepath, "../sir_sims"))  # Change to your SIR directory
    test_time = "2024_11_27_13_21_06"  # Update to your desired timestamp
    
    # Initialize data_list as a list of dictionaries
    # SF, SIR, error count, simulated symbols, SER
    data_list = []
    
    for filename in os.listdir(directory):
        if filename.startswith(test_time) and filename.endswith('.txt'):  # Adjust for your file extension
            fp = os.path.join(directory, filename)
            with open(fp) as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("#"):  # Skip header comments
                        continue
                    sep = line.strip().split(',')
                    new_row = {
                        'SF': float(sep[0]),
                        'SIR': float(sep[1]),
                        'ErrorCount': float(sep[2]),
                        'SimulatedSymbols': float(sep[3]),
                        'SER': float(sep[4])
                    }
                    data_list.append(new_row)
                f.close()
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data_list)
    print(df)
    
    # Extract unique SF values (assuming one for simplicity)
    SF_values = pd.unique(df['SF'])
    if len(SF_values) != 1:
        raise TypeError("MULTIPLE SFs DETECTED!")
    
    SF = SF_values[0]
    
    # Plot SER curves as a function of SIR
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(
        df['SIR'].astype(float),
        df['SER'].astype(float),
        marker="v",
        label=f"SF{int(SF)}",
        color="black"
    )
    ax.set_yscale("log")
    ax.set_xlabel("SIR [dB]")
    ax.set_ylabel("SER")
    ax.set_xticks(np.arange(df['SIR'].min(), df['SIR'].max()+1, 2))
    ax.grid(True, which="both", alpha=0.5)
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(df['SIR'].min(), df['SIR'].max())
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(
        f"{directory}/SIR_simulations_results_SF{int(SF)}.pdf",
        format="pdf",
        bbox_inches="tight"
    )
