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
        print(filepath)
        directory = os.path.abspath(os.path.join(filepath, "../snr_sims"))
        print(directory)
        test_time = "2024_11_18_21_39_53"
        # Initialize data_list as a list of dictionaries
        # SF, SNR, error count, simulated symbols, SER

        current_dir=os.path.dirname(os.path.realpath(__file__))
        df=pd.read_csv(os.path.join(current_dir, f"testauto_scaled.csv"))
        rate_params=df.columns[1:]
        snr_params=df.iloc[:,0]
        

        # Save the results to a .txt file for every rate parameter and create a plot
        for i, rate_param in enumerate(rate_params):
            # Plot SER curves as function of SNR
            SF = [7]
            rate = rate_param
            if (len(SF) != 1):
                 raise TypeError("MULTIPLE SFs DETECTED!")
            
            
            print(i, rate_param)
            if i > -1:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                # Classic decoder without interfering users
                ax.plot(
                    snr_params,
                    df.iloc[:,1],
                    marker="v",
                    linestyle="dashed",
                    label=f"SF{SF}, λ=0.00",
                    color="black",
                )
                # Classical with Poisson distributed interferers
                ax.plot(
                    snr_params,
                    df.iloc[:,i+2],
                    marker="v",
                    label=f"SF{SF}, λ={rate}",
                    # label=f"SF{SF}, λ={rate:.2f}",
                    color="black",
                )  # Poisson decoder with λ=rate_param
                ax.set_yscale("log")
                ax.set_xlabel("SNR [dB]")
                ax.set_ylabel("SER")
                ax.grid(True, which="both", alpha=0.5)
                ax.set_ylim(1e-5, 1)
                ax.set_xlim(-16, -4)
                ax.legend(["λ=0.00",f"λ={rate}"],loc='upper right')
                # ax.legend(["λ=0.00",f"λ={rate:.2f}"],loc='upper right')

                # Create an inset with the Poisson PMF stem plot
                inset_ax = inset_axes(
                    ax,
                    width="30%",
                    height="40%",
                    loc="lower left",
                    bbox_to_anchor=(0.1, 0.1, 1, 1),
                    bbox_transform=ax.transAxes,
                )
                plt.show()
                exit()
                l = np.linspace(0,10,11)
                poisson_dist = stats.poisson.pmf(l, mu=rate_params[i+2])
                print(poisson_dist)
                mask = (poisson_dist >= 0.005)
                inset_ax.set_title(f"PMF, λ={rate:.2f}", fontsize = (fs - 2))
                inset_ax.set_xlabel(r"$\mathrm{N_i}$", labelpad=-4, fontsize = (fs - 2))
                inset_ax.set_xlim([0, 10])
                inset_ax.set_ylim([0, 0.8])
                inset_ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

                stem_inset = inset_ax.stem(
                    l[mask],
                    poisson_dist[mask],
                    basefmt=" ",
                    linefmt="k-",
                )
                # Allow clipping of the stem plot
                for artist in stem_inset.get_children():
                    artist.set_clip_on(False)

                plt.tight_layout()
                #plt.savefig(
                #    f"{filepath}/../snr_sims/{test_time}_SNR_simulations_results_SF{str(int(float(SF[0])))}_lam{rate_param}.png"
                #)
                plt.savefig(
                    os.path.join(current_dir, f"hohoh{rate_param}.pdf"),
                    # f"{directory}/SNR_simulations_results_SF{str(int(float(SF[0])))}_lam{rate_param}.pdf",
                    format = "pdf",
                    bbox_inches = "tight"
                )
