import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from ser_includes.lora_phy import create_basechirp, upchirp_lut, generate_interferer_symbols, process_batch, dechirp, generate_noise
from ser_includes.model_space import detect
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
from numpy import savetxt
import os
import tensorflow as tf

if __name__ == "__main__":
    # Check if GPU is available otherwise use CPU
    device = tf.device("")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("Found GPU, using that")
        device = tf.device("/device:GPU:0")
    else:
        print("GPU device not found, using CPU")
        device = tf.device("/device:CPU:0")

    with device:
        # LoRa PHY parameters value
        SF = 7  # Spreading factor
        BW = 250e3  # Bandwidth [Hz] (EU863-870 DR0 channel)
        M = int(2**SF)  # Number of symbols per chirp
        SIR_tuple = (200, 1000, True)  # Set to min=max for constant SIR

        # Create the basic chirp
        basic_chirp = create_basechirp(M)

        # Create a LUT for the upchirps for faster processing -
        # A zero row is created for the interferers to look up
        upchirp_lut = tf.concat(
            [
                upchirp_lut(M, basic_chirp),
                tf.zeros((1, M), dtype=tf.complex64),
            ],
            axis=0,
        )

        # Conjugate the basic chirp for basic dechirp
        basic_dechirp = tf.math.conj(basic_chirp)

        # Simulation parameters
        n_symbols = [2,2,2,2,2,2]
        nr_of_batches = 2
            # NB: n_symbols must be divisible by batch_size
        batch_size_list = [(n*M)//nr_of_batches for n in n_symbols]
        print(f"Batch sizes: {batch_size_list}")

        snr_values = tf.cast(tf.linspace(-6, -16, 6), dtype=tf.float64)
        rate_params = tf.constant([0.0, 0.25, 0.5, 0.7, 1.0], dtype=tf.float64)
        result_list = tf.zeros(
            (snr_values.shape[0], rate_params.shape[0]), dtype=tf.float64
        )

        # Noise formula based on thermal noise N0=k*T*B
        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant
        noise_power = tf.constant((k_b * 298.16 * BW), dtype=tf.float64)  # dB
        print(noise_power)

        print(f"Running sim for a total of {n_symbols} symbols per SNR")
        start_time = time.time()

        for i in tf.range(len(rate_params)):
            for j in tf.range(len(snr_values)):
                error_count = 0
                snr_start_time = time.time()
                for batch in tf.range(nr_of_batches):
                    #print(f"\t Batch {batch} of {nr_of_batches} for rate {rate_params[i]} in snr {snr_values[j]}")
                    # Generate the user message and look up the upchirps
                    batch_size = tf.cast(batch_size_list[j], dtype=tf.int32)
                    msg_tx = tf.random.uniform(
                        (batch_size,), minval=0, maxval=M, dtype=tf.int32
                    )

                    chirped_rx = process_batch(
                        upchirp_lut,
                        rate_params[i],
                        snr_values[j],
                        msg_tx,
                        batch_size,
                        M,
                        noise_power,
                        SIR_tuple,
                    )

                    # Dechirp by multiplying the upchirp with the basic dechirp
                    dechirped_rx = dechirp(chirped_rx, basic_dechirp)

                    # Run the FFT to demodulate
                    fft_result = tf.abs(tf.signal.fft(dechirped_rx))

                    # Decode the message using argmax
                    msg_rx = detect(fft_result, snr_values[j], M, noise_power)
                    if False:
                        snr = tf.cast(snr_values[j], dtype=tf.float64)
                        snr_linear = tf.pow(tf.cast(10.0, dtype=tf.float64), snr / 10.0)
                        user_amp = tf.sqrt(snr_linear * noise_power) * (tf.cast(M, dtype=tf.float64))
                        plt.stem(fft_result[0])
                        plt.axvline(msg_rx[0], color='r', linestyle='--')
                        plt.axvline(msg_tx[0], color='g', linestyle='--')
                        plt.axhline(user_amp, color='b', linestyle='--')
                        plt.legend(['FFT', 'RX', 'TX', 'SNR'])
                        plt.title(snr_values[j])
                        plt.show()

                    # Calculate the number of errors in batch
                    msg_tx = tf.squeeze(msg_tx)
                    batch_result = tf.math.count_nonzero(msg_tx != msg_rx)
                    error_count += batch_result

                # Update the result list
                result_list = tf.tensor_scatter_nd_add(
                    result_list,
                    indices=[[j, i]],
                    updates=[error_count],
                )
                print(
                    f"Rate: {rate_params[i]}, SNR: {snr_values[j]} dB, error count: {tf.gather_nd(result_list, [[j, i]])} SER: {result_list[j, i]/n_symbols[j]:E}"
                )
                print(f"SNR time: {time.time() - snr_start_time}")
        print(f"Simulation duration: {time.time() - start_time}")

        # Stack and cast the results to float64
        SF_list = tf.fill([len(snr_values)], tf.cast(SF, tf.float64))
        N_list = tf.convert_to_tensor(n_symbols)
        snr_list = tf.cast(snr_values, tf.float64)


        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        # Save the results to a .txt file for every rate parameter and create a plot
        for i, rate_param in enumerate(rate_params):  
            ser_list = tf.divide(result_list[:, i], n_symbols)

            file_path = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.abspath(
                os.path.join(file_path, "sim_output/snr_sims")
            )

            os.makedirs(output_path, exist_ok=True)
            file_name = f"{output_path}/{time_str}_classical_snr_vs_ser_rate_{rate_param.numpy():.2f}.csv"

            import numpy as np
            savetxt(file_name, np.array([snr_values, ser_list]).T, delimiter=';', fmt='%d;%.6f')
            
            if False:
                # Plot SER curves as function of SNR
                if i > 0:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                    # Classic decoder without interfering users
                    ax.plot(
                        snr_values,
                        result_list[:, 0] / n_symbols,
                        marker="o",
                        label=f"SF{SF}, λ=0.00",
                        color="black",
                    )

                    # Classical with Poisson distributed interferers
                    ax.plot(
                        snr_values,
                        result_list[:, i] / n_symbols,
                        marker="^",
                        linestyle="dashed",
                        label=f"SF{SF}, λ={rate_param.numpy():.2f}",
                        color="black",
                    )  # Poisson decoder with λ=rate_param
                    ax.set_yscale("log")
                    ax.set_xlabel("SNR [dB]")
                    ax.set_ylabel("SER")
                    ax.grid(True, which="both", alpha=0.5)
                    ax.set_ylim(1e-5, 1)
                    ax.set_xlim(-16, -4)

                    # Create an inset with the Poisson PMF stem plot
                    inset_ax = inset_axes(
                        ax,
                        width="40%",
                        height="45%",
                        loc="lower left",
                        bbox_to_anchor=(0.1, 0.1, 1, 1),
                        bbox_transform=ax.transAxes,
                    )
                    poisson_dist = tf.random.poisson(
                        [batch_size], rate_param, dtype=tf.int32
                    )
                    poisson_values, idx, poisson_counts = tf.unique_with_counts(
                        poisson_dist
                    )
                    poisson_count = poisson_counts / batch_size
                    inset_ax.set_title(f"PMF, λ={rate_param.numpy():.2f}")
                    inset_ax.set_xlabel(r"$N_i$", labelpad=-4)
                    inset_ax.set_xlim([0, 10])
                    inset_ax.set_ylim([0, 0.8])

                    stem_inset = inset_ax.stem(
                        poisson_values.numpy(),
                        poisson_count.numpy(),
                        basefmt=" ",
                        linefmt="k-",
                    )
                    # Allow clipping of the stem plot
                    for artist in stem_inset.get_children():
                        artist.set_clip_on(False)

                    plt.tight_layout()
                    plt.savefig(
                        f"{output_path}/{time_str}_SNR_simulations_results_SF{SF}_lam{rate_param.numpy()}.png"
                    )
