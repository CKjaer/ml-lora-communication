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
        filepath = os.path.abspath(__file__)
        # print(filepath)
        directory = os.path.abspath(os.path.join(filepath, "../snr_sims"))
        # print(directory)
        test_time = "2024_11_27_13_05_32"
        # Initialize data_list as a list of dictionaries
        # SF, SNR, error count, simulated symbols, SER
        data_list = []
        for filename in os.listdir(directory):
            if filename.startswith(test_time) & filename.endswith('.csv'):
                fp = os.path.join(directory, filename)
                rate = filename.split('_')[-1].removeprefix('lam').removesuffix('.csv')
                with open(fp) as f:
                    lines = f.readlines()
                    for line in lines:
                        sep = line.strip().split(';')
                        new_row = {'Rate': float(rate), 'SNR': sep[0], 'SER': sep[1]}
                        data_list.append(new_row)
                    f.close()
        df_classical = pd.DataFrame(data_list)

        names=["exp_scaled","batch_scaled","auto_scaled"]
        df=[]
        for name in names:
            current_dir=os.path.dirname(os.path.realpath(__file__))
            test_type= name
            df.append(pd.read_csv(os.path.join(current_dir, "test_data",f"test_{test_type}.csv")))
        rate_params=df[0].columns[1:]
        # snr_params=df.iloc[:,0]
        
        
        # Save the results to a .txt file for every rate parameter and create a plot
        for i, rate_param in enumerate(rate_params):
            # Plot SER curves as function of SNR
            zero_data = df_classical[df_classical["Rate"]==float(0)]
            current_data = df_classical[df_classical["Rate"]==float(rate_param)]
            
            SF = [7]
            rate = float(rate_param)
            if (len(SF) != 1):
                 raise TypeError("MULTIPLE SFs DETECTED!")
            
            
            
            if i > -1:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                # Classic decoder with Posson distributed interfering users
                ax.plot(
                    current_data['SNR'].astype(float).astype(int),
                    current_data['SER'].astype(float),
                    marker="v",
                    # linestyle="dashed",
                    label=f"Classical, λ={rate:.2f}",
                    color="black",
                )
                legend_list = [f"Classical, λ={rate:.2f}"]

                for x in range(len(df)):
                    rate_params=df[x].columns[1:]
                    snr_params=df[x].iloc[:,0]
                    test_type=names[x]
                    if test_type=="exp_scaled":
                        color="red"
                        linestyle="dashdot"
                        test_name="SNR-based"
                    if test_type=="batch_scaled":
                        color="green"
                        linestyle="dotted"
                        test_name="CO-SNR"
                    if test_type=="auto_scaled":
                        color="blue"
                        linestyle="-"
                        test_name="Peak"
                    # with Poisson distributed interferers
                    ax.plot(
                        snr_params,
                        df[x].iloc[:,i+1],
                        marker="s",
                        label=f"CNN-FSD, λ={rate}",
                        linestyle=linestyle,
                        color=color,
                    )  # Poisson decoder with λ=rate_param
                    legend_list.append(f"{test_name} CNN-FSD, λ={rate:.2f}")

                ax.set_yscale("log")
                ax.set_xlabel("SNR [dB]")
                ax.set_ylabel("SER")
                ax.grid(True, which="both", alpha=0.5)
                ax.set_ylim(1e-5, 1)
                # ax.set_xlim(-16, -6)
                ax.set_xlim(-16, -6)
                ax.legend(legend_list,loc='lower left')
                # ax.legend([f"CNN λ={rate}"],loc='lower right')
                # ax.legend(["λ=0.00",f"λ={rate:.2f}"],loc='upper right')


                # Create an inset with the Poisson PMF stem plot
                # inset_ax = inset_axes(
                #     ax,
                #     width="30%",
                #     height="40%",
                #     loc="lower left",
                #     bbox_to_anchor=(0.1, 0.1, 1, 1),
                #     bbox_transform=ax.transAxes,
                # )
                # if rate!=0:
                #     l = np.linspace(0,10,11)
                #     poisson_dist = stats.poisson.pmf(l, mu=rate)
                #     # print(poisson_dist)
                #     mask = (poisson_dist >= 0.005)
                #     inset_ax.set_title(f"PMF, λ={rate:.2f}", fontsize = (fs - 2))
                #     inset_ax.set_xlabel(r"$\mathrm{N_i}$", labelpad=-4, fontsize = (fs - 2))
                #     inset_ax.set_xlim([0, 10])
                #     inset_ax.set_ylim([0, 0.8])
                #     inset_ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

                #     stem_inset = inset_ax.stem(
                #         l[mask],
                #         poisson_dist[mask],
                #         basefmt=" ",
                #         linefmt="k-",
                #     )
                #     # Allow clipping of the stem plot
                #     for artist in stem_inset.get_children():
                #         artist.set_clip_on(False)


                # plt.tight_layout()
                #plt.savefig(
                #    f"{filepath}/../snr_sims/{test_time}_SNR_simulations_results_SF{str(int(float(SF[0])))}_lam{rate_param}.png"
                #)
                plt.savefig(
                    os.path.join(current_dir, f"snr_combined_lam{rate_param}.pdf"),
                    
                    # f"{directory}/SNR_simulations_results_SF{str(int(float(SF[0])))}_lam{rate_param}.pdf",
                    format = "pdf",
                    bbox_inches = "tight"
                )