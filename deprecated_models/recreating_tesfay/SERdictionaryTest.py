import matplotlib
matplotlib.use('TkAgg')  # Change the backend to 'TkAgg' or another interactive backend
import matplotlib.pyplot as plt
from numpy import savetxt
import numpy as np
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

# Sample data
ser_data = {
    0.0: {
        -16.0: 0.749844,
        -14.0: 0.512969,
        -12.0: 0.211875,
        -10.0: 0.042656,
        -8.0: 0.001563,
        -6.0: 0.000156,
        -4.0: 0.0,
    },
    0.25: {
        -16.0: 0.758437,
        -14.0: 0.504062,
        -12.0: 0.234844,
        -10.0: 0.061406,
        -8.0: 0.017656,
        -6.0: 0.0125,
        -4.0: 0.007812,
    },
    0.5: {
        -16.0: 0.745156,
        -14.0: 0.523594,
        -12.0: 0.254688,
        -10.0: 0.088594,
        -8.0: 0.042031,
        -6.0: 0.030625,
        -4.0: 0.01875,
    },
    0.7: {
        -16.0: 0.749219,
        -14.0: 0.529219,
        -12.0: 0.27,
        -10.0: 0.104375,
        -8.0: 0.060156,
        -6.0: 0.030625,
        -4.0: 0.029531,
    },
    1.0: {
        -16.0: 0.750938,
        -14.0: 0.528906,
        -12.0: 0.287656,
        -10.0: 0.13625,
        -8.0: 0.075938,
        -6.0: 0.059375,
        -4.0: 0.037187,
    },
}


if __name__ == "__main__":
    outputpath = "cnn_models/SERplots/no_scaling"
    os.makedirs(outputpath, exist_ok=True)
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'TeX Gyre Pagella'
    # plt.rcParams['font.family'] ='TeX Gyre Pagella'
    fs = 20
    plt.rcParams.update({'font.size': fs})
        
    for rate, values in ser_data.items():
        snr_values = sorted(values.keys())  # Just sort the keys directly
        ser_values = [values[snr] for snr in snr_values]  # loop through to not mix up the order
        
        savetxt(os.path.join(outputpath,f'snr_vs_ser_rate_{rate}.csv'), np.array([snr_values, ser_values]).T, delimiter=';', fmt='%d;%.6f')

        if rate == 0:
            zero_ser_values = ser_values
            zero_snr_values = snr_values
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(
            zero_snr_values,
            zero_ser_values,
            label=f"SF:{7} λ=0.00",
            linestyle="dashed",
            color="black",
            marker="v"
        )
        ax.plot(
            snr_values,
            ser_values,
            marker="v",
            color="black",
            label=f"SF:{7} λ={rate:.2f}"
        )
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('SER')
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(-16, -4)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(loc='upper right')
        
        if rate != 0:
            # Create an inset with the Poisson PMF stem plot
            inset_ax = inset_axes(
                ax,
                width="30%",
                height="40%",
                loc="lower left",
                bbox_to_anchor=(0.1, 0.1, 1, 1),
                bbox_transform=ax.transAxes,
            )
            l = np.linspace(0,10,11)
            poisson_dist = stats.poisson.pmf(l, mu=rate)
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
                
        plt.savefig(
            os.path.join(outputpath, f"snr_vs_ser_rate_{rate}.pdf"),
            format = "pdf",
            bbox_inches = "tight"
        )

        plt.tight_layout()
        plt.show()