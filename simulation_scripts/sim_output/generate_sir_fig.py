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
        data_folder = os.path.join(current_file_dir, "test_data")
        
        # Load data from the simulated cases
        data_list = {}
        for filename in os.listdir(data_folder):
            if filename.startswith('2024') and filename.endswith('.csv'):
                fp = os.path.join(data_folder, filename)
                rate = filename.split('_')[-1].removeprefix('lam').removesuffix('.csv')
                rate_str = f'{float(rate):.1f}'
                if rate_str == '0.2':
                    rate_str = '0.25'
                with open(fp) as f:
                    lines = f.readlines()
                    for line in lines:
                        sep = line.strip().split(';')
                        snr = int(sep[0]) 
                        ser = float(sep[1])  

                        if snr not in data_list:
                            data_list[snr] = {}

                        data_list[snr][rate_str] = ser
                    f.close()
        
        sim_df = pd.DataFrame.from_dict(data_list, orient='index').reset_index()
        sim_df = sim_df.rename(columns={'index': 'SNR'}) 
        sim_df = sim_df.sort_values(by=sim_df.columns[1], ascending=False)
        sim_df = sim_df.iloc[:-1]
        print(sim_df)
        
        # Load the data as a pandas dataframe for the tested model 
        filename = "test_SIR.csv"
        test_df = pd.read_csv(os.path.join(data_folder, filename))
        test_df.rename(columns={test_df.columns[0]: 'SNR'}, inplace=True)
        print(test_df)
        rate_params = test_df.columns[1:]
        snr_vals = test_df.iloc[:, 0]

        # print(sim_df[rate_params[0]])

        # Save the results to a .txt file for every rate parameter and create a plot
        for i, rate_param in enumerate(rate_params):
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Classical decoder
            # ax.plot(
            #     snr_vals,
            #     sim_df[rate_param],
            #     marker="v",
            #     label=f"Classical, λ={rate_param}",
            #     color="black",
            # )

            # Classical with Poisson distributed interferers
            ax.plot(
                snr_vals,
                test_df[rate_param],
                marker="s",
                label=f"CNN, λ={rate_param}",
                color="black",
            )  
        
            ax.set_yscale("log")
            ax.set_xlabel("SIR [dB]")
            ax.set_ylabel("SER")
            ax.grid(True, which="both", alpha=0.5)
            ax.set_ylim(1e-6, 1)
            ax.set_xlim(-10, 10)
            ax.legend(loc='upper right')


            
            plt.savefig(
                os.path.join(data_folder, f"{filename.replace(".csv", "").replace("test_", "")}","plots", f"sir_test_result_lam{rate_param}.pdf"),
                format = "pdf",
                bbox_inches = "tight"
            )
