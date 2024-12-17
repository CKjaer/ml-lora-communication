import matplotlib

matplotlib.use("TkAgg")  # Change the backend to 'TkAgg' or another interactive backend
import matplotlib.pyplot as plt
from numpy import savetxt
import numpy as np
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import json
from scipy.interpolate import interp1d

# Load cnn-tsd data
with open("cnn_output/test_IQCNN/symbol_error_rates.json", "r") as f:
    ser_data = json.load(f)

# load classical decoder data
with open("cnn_output/test_IQCNN/classical_ser.json", "r") as f:
    classical_ser_data = json.load(f)

# load cnn-fsd data
with open("cnn_output/test_IQCNN/cnn_fsd_ser.json", "r") as f:
    fsd_ser_data = json.load(f)

# Convert the keys of the outer dictionary to floats and the keys of the inner dictionaries to ints
ser_data = {float(outer_key): {int(inner_key): float(value) for inner_key, value in inner_dict.items()} for outer_key, inner_dict in ser_data.items()}
classical_ser_data = {float(outer_key): {int(inner_key): float(value) for inner_key, value in inner_dict.items()} for outer_key, inner_dict in classical_ser_data.items()}
fsd_ser_data = {float(outer_key): {int(inner_key): float(value) for inner_key, value in inner_dict.items()} for outer_key, inner_dict in fsd_ser_data.items()}


if __name__ == "__main__":
    outputpath = "cnn_output/test_IQCNN/plots"
    os.makedirs(outputpath, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Palatino Linotype"
    plt.rcParams["font.family"] = "Palatino Linotype"
    fs = 20
    plt.rcParams.update({"font.size": fs})

    for rate, values in ser_data.items():
        snr_values = sorted(values.keys())  # Just sort the keys directly
        ser_values = [values[snr] for snr in snr_values]  # loop through to not mix up the order after sorting
        classical_ser_values = [classical_ser_data[rate][snr] for snr in snr_values]
        fsd_ser_values = [fsd_ser_data[rate][snr] for snr in snr_values]

        print(f"ser_values: {ser_values}")
        print(f"classical_ser_values: {classical_ser_values}")
        print(f"fsd_ser_values: {fsd_ser_values}")

        # savetxt(os.path.join(outputpath,f'snr_vs_ser_rate_{rate}.csv'), np.array([snr_values, ser_values]).T, delimiter=';', fmt='%d;%.6f')

        # if rate == 0:
        #     zero_ser_values = ser_values
        #     zero_snr_values = snr_values
        #     zero_classical_ser_values = classical_ser_values
        #     zero_fsd_ser_values = fsd_ser_values

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.plot(zero_snr_values, zero_ser_values, label=f"CNN-TSD λ=0.00", linestyle="dashed", color="red", marker="o")
        ax.plot(snr_values, ser_values, marker="o", color="red", label=f"CNN-TSD λ={rate:.2f}")
        # ax.plot(zero_snr_values, zero_classical_ser_values, label=f"Classical λ=0.00", linestyle="dashed", color="black", marker="v")
        ax.plot(snr_values, classical_ser_values, marker="v", color="black", label=f"Classical λ={rate:.2f}")
        ax.plot(snr_values, fsd_ser_values, marker="s", color="blue", label=f"CNN-FSD λ={rate:.2f}")
        # ax.plot(zero_snr_values, zero_fsd_ser_values, label=f"CNN-FSD λ=0.00", linestyle="dashed", color="blue", marker="s")
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("SER")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(-16, -6)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(loc="lower left")

        # Define the target SER for comparison
        target_ser = 1e-3  # Example target SER

        # Interpolation function for each model
        interp_cnn_tsd = interp1d(ser_values, snr_values, kind="linear", bounds_error=False, fill_value="extrapolate")
        interp_classical = interp1d(classical_ser_values, snr_values, kind="linear", bounds_error=False, fill_value="extrapolate")
        interp_cnn_fsd = interp1d(fsd_ser_values, snr_values, kind="linear", bounds_error=False, fill_value="extrapolate")

        # Calculate the SNR for the target SER
        snr_cnn_tsd = interp_cnn_tsd(target_ser)
        snr_classical = interp_classical(target_ser)
        snr_cnn_fsd = interp_cnn_fsd(target_ser)

        # Calculate the dB gain for CNN-TSD over the classical decoder
        gain_tsd_vs_classical = snr_classical - snr_cnn_tsd
        gain_fsd_vs_classical = snr_classical - snr_cnn_fsd

        print(f"At target SER={target_ser}:")
        print(f"  SNR (CNN-TSD): {snr_cnn_tsd:.2f} dB")
        print(f"  SNR (Classical): {snr_classical:.2f} dB")
        print(f"  SNR (CNN-FSD): {snr_cnn_fsd:.2f} dB")
        print(f"  Gain (CNN-TSD vs Classical): {gain_tsd_vs_classical:.2f} dB")
        print(f"  Gain (CNN-FSD vs Classical): {gain_fsd_vs_classical:.2f} dB")

        plt.savefig(os.path.join(outputpath, f"snr_vs_ser_rate_{rate}.pdf"), format="pdf", bbox_inches="tight")

        # plt.tight_layout()
        plt.show()
