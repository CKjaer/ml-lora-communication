import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# NB. do to errors individual mixed test are read from the .log file 

if __name__ == "__main__":
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
    plt.rcParams['font.family'] ='Palatino Linotype'
    fs = 20
    plt.rcParams.update({'font.size': fs})

    sir_values = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, 'mixed_test_sir_data')

    # SIR -6 dB mixed test extracted from the .log files
    ser_values_6 = np.array([0.13349609375, 0.0421328125, 0.03232421875, 0.04416015625, 0.16213671875, 
                            0.38101171875, 0.6534921875, 0.76808203125, 0.85131640625, 0.936171875, 0.916796875])

    # SIR 0 dB mixed test read from the csv file
    file_path = os.path.join(data_folder, "mixed_test_LoRaCNN_snr_0_rate_0.25.csv")
    ser_values_0 = pd.read_csv(file_path, sep=',')
    ser_values_0.columns = ['SNR', 'SER']
    
    # Classic CNN model
    file_path = os.path.join(data_folder, "2024_12_04_10_33_05_SIR_simulations_results_SF7_rate0.25.txt")
    ser_values_classic = pd.read_csv(file_path, sep=',', skiprows=1)
    print(ser_values_classic.iloc[:, -1])

    # Entire CNN model
    file_path = os.path.join(data_folder, "test_SIR.csv")
    ser_values_cnn = pd.read_csv(file_path, sep=',')
    print(ser_values_cnn.iloc[:, -1])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))       
    # 0 dB mixed test
    ax.plot(
        sir_values,
        ser_values_6,
        marker="o",
        label=r"$\mathrm{CNN\text{-}FSD_{-6 \, dB}}$",
        color = 'red'
    )
 
    # -6 dB mixed test
    ax.plot(
        sir_values,
        ser_values_0['SER'],
        marker="D",
        label=r"$\mathrm{CNN\text{-}FSD_{0 \, dB}}$",
        color = 'orange'

    )  
    # CNN decoder
    ax.plot(
        sir_values,
        ser_values_cnn.iloc[:, -1],
        marker="s",
        label="CNN-FSD",
        color = 'blue'
    )  

    # Classical decoder
    ax.plot(
        sir_values,
        ser_values_classic.iloc[:, -1],
        marker='v',
        label=f"Classical decoder",
        color = 'black', 
    )  
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_yscale("log")
    ax.set_xlabel("SIR [dB]")
    ax.set_ylabel("SER")
    ax.grid(True, which="both", alpha=0.5)
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-10, 10)
    ax.legend(loc='lower left')

    plt.savefig(
        os.path.join(data_folder, "sir_test_result.pdf"),
        format = "pdf",
        bbox_inches = "tight"
    )
