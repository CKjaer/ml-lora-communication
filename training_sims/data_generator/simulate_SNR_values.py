import os
import lora_phy as lora
import model_space as model
import matplotlib.pyplot as plt
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
        freq_eu = int(868e3)

        # Create the basic chirp
        basic_chirp = lora.create_basechirp(M, device)

        # Create a LUT for the upchirps for faster processing
        upchirp_lut = tf.concat(
            [
                lora.upchirp_lut(M, basic_chirp, device),
                tf.zeros((1, M), dtype=tf.complex64),
            ],
            axis=0,
        )

        # Conjugate the basic chirp for basic dechirp
        basic_dechirp = tf.math.conj(basic_chirp)

        # Simulation parameters, the number of symbols simulated results in a 5% tolerance for SER of 1e-5
        N = int(1e5)  # int(2e6)
        batch_size = int(1e4)  # Number of symbols per batch
        nr_of_batches = int(N / batch_size)  # NB: N must be divisible by batch_size

        snr_values = tf.cast(tf.linspace(-4, -16, 13), dtype=tf.float64)
        # rate_params = tf.constant([0.25, 0.5, 0.7, 1], dtype=tf.float64)
        rate_params = tf.constant([0.25], dtype=tf.float64)
        result_list = tf.zeros(
            (snr_values.shape[0], rate_params.shape[0]), dtype=tf.float64
        )

        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant
        noise_power = tf.constant((k_b * 298.16 * BW), dtype=tf.float64)  # dB

        start_time = time.time()

        # @tf.function
        def process_batch(rate_param, snr):
            # Generate the interfering users symbols and their distances
            inter_symbols, distance = lora.generate_interferer_symbols(
                batch_size, rate_params, M, upchirp_lut
            )

            # Generate the user message and look up the upchirps
            msg_tx = tf.random.uniform(
                (batch_size,), minval=0, maxval=M, dtype=tf.int32
            )
            user_chirp_tx = tf.gather(upchirp_lut, msg_tx, axis=0)

            # Generate complex AWGN channel
            noise_stddev = tf.cast(tf.sqrt(noise_power / 2.0), dtype=tf.float32)

            # Generate complex AWGN channel
            noise_real = tf.random.normal(
                shape=tf.shape(user_chirp_tx),
                mean=0.0,
                stddev=noise_stddev,
                dtype=tf.float32,
            )
            noise_imag = tf.random.normal(
                shape=tf.shape(user_chirp_tx),
                mean=0.0,
                stddev=noise_stddev,
                dtype=tf.float32,
            )
            complex_noise = tf.complex(noise_real, noise_imag)

            # Channel coefficients
            snr_linear = tf.pow(10.0, snr / 10.0)
            user_power = tf.sqrt(snr_linear * noise_power)  # W
            interferer_power = tf.sqrt(1 / (distance**2))  # W

            # Scale the interfering users by their power and sum to match
            inter_symbols_scaled = tf.reduce_sum(
                interferer_power * inter_symbols, axis=1
            )

            # Combine the signals and add noise
            upchirp_tx = (
                tf.cast(user_power, dtype=tf.complex64) * user_chirp_tx
                + inter_symbols_scaled
                + complex_noise
            )

            # Dechirp by multiplying the upchirp with the basic dechirp and run the FFT to demodulate
            dechirp_rx = tf.multiply(upchirp_tx, basic_dechirp)
            fft_result = tf.abs(tf.signal.fft(dechirp_rx))

            # Decode the message using argmax
            msg_rx = tf.argmax(fft_result, axis=1, output_type=tf.int32)

            # Calculate the number of errors in batch
            msg_tx = tf.squeeze(msg_tx)
            batch_result = tf.math.count_nonzero(msg_tx != msg_rx)

            return batch_result

        for i in tf.range(len(rate_params)):
            for j in tf.range(len(snr_values)):
                error_count = 0
                for batch in tf.range(nr_of_batches):
                    error_count += process_batch(rate_params[i], snr_values[j])

                # Update the result list
                result_list = tf.tensor_scatter_nd_add(
                    result_list,
                    indices=[[j, i]],
                    updates=[error_count],
                )
                print(
                    f"SNR: {snr_values[j]} dB, error count: {tf.gather_nd(result_list, [[j, i]])} SER: {result_list[j, i]/N:E}"
                )
        print(f"Simulation duration: {time.time() - start_time}")

        # Stack and cast the results to float64
        SF_list = tf.fill([len(snr_values)], tf.cast(SF, tf.float64))
        N_list = tf.fill([len(snr_values)], tf.cast(N, tf.float64))
        snr_list = tf.cast(snr_values, tf.float64)

        # Save the results to a .txt file for every rate parameter and create a plot
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()
        for i, rate_param in enumerate(rate_params):
            ser_list = tf.divide(result_list[:, i], N)
            output = tf.stack(
                [SF_list, snr_values, result_list[:, i], N_list, ser_list], axis=0
            )

            file_path = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.abspath(os.path.join(file_path, "output"))
            os.makedirs(output_path, exist_ok=True)
            time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
            file_name = f"{output_path}/{time_str}_SER_simulations_results_SF{SF}_lam{rate_param.numpy()}.txt"
            head = (
                f"Test done: {time_str} - "
                f"time taken: {time.time() - start_time} \n"
                f"SF, SNR, error count, simulated symbols, SER"
            )
            savetxt(file_name, output.numpy().T, delimiter=",", header=head)

            # Plot the results
            axs[i].plot(
                snr_values, result_list[:, i] / N, marker="^", linestyle="dashed"
            )
            axs[i].set_yscale("log")
            axs[i].set_xlabel("SNR [dB]")
            axs[i].set_ylabel("SER")
            axs[i].grid(True)
            axs[i].legend([f"SF{SF}, λ={rate_param.numpy():.1f}"])

        plt.tight_layout()
        plt.savefig(f"{output_path}/{time_str}_SER_simulations_results_SF{SF}.png")
