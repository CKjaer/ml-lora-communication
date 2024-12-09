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
    test_time = "2024_12_04_10_33_05"  # Update to your desired timestamp
    
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
    # print(df)
    
    current_dir=os.path.dirname(os.path.realpath(__file__))
    test_type= "SIR"
    df_CNN=pd.read_csv(os.path.join(current_dir, "test_data",f"test_{test_type}.csv"))
    rate_params=df.columns[1:]
    snr_params=np.linspace(-10,10,11)
    # snr_params=df.iloc[:,0]

    print(rate_params)
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
        label=f"Classical, λ=0.25",
        color="black"
    )
    #plot the CNN results
    ax.plot(
        snr_params,
        df_CNN.iloc[:,1],
        marker="s",
        label=f"CNN-FSD, λ=0.25",
        color="blue"
    )
    ax.set_yscale("log")
    ax.set_xlabel("SIR [dB]")
    ax.set_ylabel("SER")
    ax.grid(True, which="both", alpha=0.5)
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(df['SIR'].min(), df['SIR'].max())
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(
        f"sir_{test_type}_lam{0.25}.pdf",
        format="pdf",
        bbox_inches="tight"
    )

