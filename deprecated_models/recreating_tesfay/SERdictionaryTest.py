import matplotlib
matplotlib.use('TkAgg')  # Change the backend to 'TkAgg' or another interactive backend
import matplotlib.pyplot as plt
from numpy import savetxt
import numpy as np
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import json



# Load the JSON data
with open("iq_cnn_output/test_IQCNN/symbol_error_rates.json", "r") as f:
    ser_data = json.load(f)

# load classical decoder data
with open("iq_cnn_output/test_IQCNN/classical_ser.json", "r") as f:
    classical_ser_data = json.load(f)

# Convert the keys of the outer dictionary to floats and the keys of the inner dictionaries to ints
ser_data = {float(outer_key): {int(inner_key): float(value) for inner_key, value in inner_dict.items()} for outer_key, inner_dict in ser_data.items()}
classical_ser_data = {float(outer_key): {int(inner_key): float(value) for inner_key, value in inner_dict.items()} for outer_key, inner_dict in classical_ser_data.items()}

if __name__ == "__main__":
    outputpath = "iq_cnn_output/test_IQCNN/plots"
    os.makedirs(outputpath, exist_ok=True)
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
    plt.rcParams['font.family'] ='Palatino Linotype'
    fs = 20
    plt.rcParams.update({'font.size': fs})
        
    for rate, values in ser_data.items():
        snr_values = sorted(values.keys())  # Just sort the keys directly
        ser_values = [values[snr] for snr in snr_values]  # loop through to not mix up the order
        classical_ser_values = [classical_ser_data[rate][snr] for snr in snr_values]
        
        print(f"ser_values: {ser_values}")
        print(f"classical_ser_values: {classical_ser_values}")
        
        #savetxt(os.path.join(outputpath,f'snr_vs_ser_rate_{rate}.csv'), np.array([snr_values, ser_values]).T, delimiter=';', fmt='%d;%.6f')

        if rate == 0:
            zero_ser_values = ser_values
            zero_snr_values = snr_values
            zero_classical_ser_values = classical_ser_values
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(
            zero_snr_values,
            zero_ser_values,
            label=f"CNN λ=0.00",
            linestyle="dashed",
            color="blue",
            marker="s"
        )
        ax.plot(
            snr_values,
            ser_values,
            marker="s",
            color="blue",
            label=f"CNN λ={rate:.2f}"
        )
        ax.plot(
            zero_snr_values,
            zero_classical_ser_values,
            label=f"Classical λ=0.00",
            linestyle="dashed",
            color="black",
            marker="v"
        )
        ax.plot(
            snr_values,
            classical_ser_values,
            marker="v",
            color="black",
            label=f"Classical λ={rate:.2f}"
        )
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('SER')
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(-16, -6)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(loc='lower left')
        
        # if rate != 0:
        #     # Create an inset with the Poisson PMF stem plot
        #     inset_ax = inset_axes(
        #         ax,
        #         width="30%",
        #         height="40%",
        #         loc="lower left",
        #         bbox_to_anchor=(0.1, 0.1, 1, 1),
        #         bbox_transform=ax.transAxes,
        #     )
        #     l = np.linspace(0,10,11)
        #     poisson_dist = stats.poisson.pmf(l, mu=rate)
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
                
        plt.savefig(
            os.path.join(outputpath, f"snr_vs_ser_rate_{rate}.pdf"),
            format = "pdf",
            bbox_inches = "tight"
        )

        #plt.tight_layout()
        plt.show()