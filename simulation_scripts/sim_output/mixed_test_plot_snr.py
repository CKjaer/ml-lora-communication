import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == "__main__":
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
        plt.rcParams['font.family'] ='Palatino Linotype'
        fs = 20
        plt.rcParams.update({'font.size': fs})


        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(current_file_dir, "test_data/mixed_test_snr_data")
        
        snr_vals = pd.Series([-16, -14, -12, -10, -8, -6])
        
        # lambda = 0.25
        ser_lam_025 = pd.DataFrame({
            '0.25': [0.7154017857142857, 0.5066964285714286, 0.2689732142857143, 
                0.08002804487179487, 0.030899439102564104, 0.01676007264464926]
        })

        # lambda = 0.5
        ser_lam_05 = pd.DataFrame({ 
            '0.5' : [0.7444196428571429, 0.5647321428571429, 0.30691964285714285, 
                               0.15915464743589744, 0.07868589743589743, 0.045240895417306706]
        })

        # lambda = 0.7       
        file_path = os.path.join(data_folder, 'mixed_test_LoRaCNN_snr_-10_rate_0.7.csv')
        ser_lam_07 = pd.read_csv(file_path, sep=',')

        # lambda = 1
        file_path = os.path.join(data_folder, 'mixed_test_LoRaCNN_snr_-10_rate_1.0.csv')
        ser_lam_1 = pd.read_csv(file_path, sep=',')

        combined_ser = pd.concat([snr_vals, ser_lam_025['0.25'], ser_lam_05['0.5'], ser_lam_07['0.7'], ser_lam_1['1']], axis=1)
        combined_ser.reset_index(level=0, inplace=True)

        rate_params = combined_ser.columns[2:]

        # Save the results to a .txt file for every rate parameter and create a plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
        for i, rate_param in enumerate(rate_params):
            
            # Mixed test results
            markers = ['v', 's', 'o', 'D']
            ax.plot(
                snr_vals,
                combined_ser[rate_param],
                marker=markers[i],
                label=f"SNR=-10 dB, λ={rate_param}",
                color="black",
            )
        ax.set_yscale("log")
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("SER")
        ax.grid(True, which="both", alpha=0.5)
        ax.set_ylim(1e-6, 1)
        ax.set_xlim(-16, -6)
        ax.legend(loc='lower right')

        plt.show()
        exit()

        plt.savefig(
            os.path.join(data_folder, "plots", f"snr_test_result_lam{rate_param}.pdf"),
            format = "pdf",
            bbox_inches = "tight"
        )
