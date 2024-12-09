import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def read_data(prefix, folder):  
    data_list = {}
    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith('.csv'):
            fp = os.path.join(folder, filename)
            rate = filename.split('_')[-1].removeprefix('lam').removesuffix('.csv')  
            if rate == '1':
                rate = '1.0'             
            with open(fp) as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip the first line
                    sep = line.strip().split(',')
                    snr = int(sep[0]) 
                    ser = float(sep[1])  

                    if snr not in data_list:
                        data_list[snr] = {}

                    data_list[snr][rate] = ser
                f.close()
    return pd.DataFrame.from_dict(data_list, orient='index').reset_index()

if __name__ == "__main__":
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
    plt.rcParams['font.family'] ='Palatino Linotype'
    fs = 20
    plt.rcParams.update({'font.size': fs})
    print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_file_dir, "test_data", "mixed_test_snr_data")
    
    # Load data for -6 dB mixed test
    ser_df_6 = read_data('mixed_test_LoRaCNN_snr_-6', data_folder)  

    # Load data for 10 dB mixed test
    ser_lam_025 = np.array([0.7154017857142857, 0.5066964285714286, 0.2689732142857143, 
            0.08002804487179487, 0.030899439102564104, 0.01676007264464926])
    ser_df_10 = read_data('mixed_test_LoRaCNN_snr_-10', data_folder)
    ser_df_10.insert(1, '0.25', ser_lam_025)

    rate_params = ser_df_10.columns[1:]
    snr_vals = ser_df_10['index']   
    # print(ser_df_6)
    # print(ser_df_10)

    # Read SER from original CNN model for every rate parameter
    filename = "test_auto_scaled.csv"
    ser_cnn = pd.read_csv(os.path.join(current_file_dir, "test_data", filename))

    # Accuracy difference between the two models
    accuracy_diff_6 = np.zeros([len(snr_vals), len(rate_params)])
    accuracy_diff_10 = np.zeros([len(snr_vals), len(rate_params)])
    for i, rate_param in enumerate(rate_params):
        ser_diff_6 = np.array(1 - ser_cnn[rate_param]) - np.array(1 - ser_df_6[rate_param])
        print(ser_diff_6)
        accuracy_diff_6[:, i] =  ser_diff_6 * 100

    
        ser_diff_10 = np.array(1 - ser_cnn[rate_param]) - np.array(1 - ser_df_10[rate_param])
        accuracy_diff_10[:, i] = ser_diff_10 * 100

    print(accuracy_diff_6)
    print(accuracy_diff_10)

    
   # Create a LaTeX table for the accuracy differences using booktabs style
    with open(os.path.join(data_folder, "accuracy_diff_table.tex"), "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccc}\n")  # Remove vertical lines for booktabs style
        f.write("\\toprule\n")
        f.write("SNR [dB] & Rate & Accuracy Difference -6 dB [\\%] & Accuracy Difference -10 dB [\\%] \\\\\n")
        f.write("\\midrule\n")
        for i, snr in enumerate(snr_vals):
            first_row = True  # Flag for printing SNR only once per group
            for j, rate_param in enumerate(rate_params):
                if first_row:
                    f.write(f"{snr} & {rate_param} & {accuracy_diff_6[i, j]:.2f} & {accuracy_diff_10[i, j]:.2f} \\\\\n")
                    first_row = False
                else:
                    f.write(f" & {rate_param} & {accuracy_diff_6[i, j]:.2f} & {accuracy_diff_10[i, j]:.2f} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Accuracy differences between the original CNN model and the mixed test model for different SNR values and rate parameters.}\n")
        f.write("\\label{tab:accuracy_diff}\n")
        f.write("\\end{table}\n")





    markers = ['v', 's', 'o', 'D']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))           
    for i, rate_param in enumerate(rate_params):
        # Mixed test results -6 dB
        ax.plot(
            snr_vals,
            ser_df_6[rate_param],
            marker=markers[i],
            label=f"CNN-FSD, SNR=-6 dB, λ={rate_param}",
            color="#1f77b4",
        )
    
    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("SER")
    ax.grid(True, which="both", alpha=0.5)
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-16, -6)
    ax.legend(loc='lower left')

    plt.savefig(
        os.path.join(data_folder, "mixed_test_snr_06.pdf"),
        format = "pdf",
        bbox_inches = "tight"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))           

    for i, rate_param in enumerate(rate_params):
        # Mixed test results 10 dB
        ax.plot(
            snr_vals,
            ser_df_10[rate_param],
            marker=markers[i],
            label=f"CNN-FSD, SNR=-10 dB, λ={rate_param}",
            color="#ff7f0e",
        )

    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("SER")
    ax.grid(True, which="both", alpha=0.5)
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-16, -6)
    ax.legend(loc='lower left')

    plt.savefig(
        os.path.join(data_folder, "mixed_test_snr_10.pdf"),
        format = "pdf",
        bbox_inches = "tight"
    )
