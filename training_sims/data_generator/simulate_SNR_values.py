import os
import lora_phy as lora
import model_space as model
import matplotlib.pyplot as plt
import time
from numpy import savetxt
import os
import tensorflow as tf

if __name__ == "__main__":
    # Check if GPU is available - if it is, tensor flow runs on the GPU
    # Otherwise, run it on a CPU :)
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

        # Create the basic chirp - Formula based on "Efficient Design of Chirp Spread Spectrum Modulation for Low-Power Wide-Area Networks" by Nguyen et al.
        basic_chirp = lora.create_basechirp(M, device)
        # Once the basic chirp is created, we can create the upchirp LUT, as it is faster than calculating it on the fly
        upchirp_lut = tf.concat(
            [
                lora.upchirp_lut(M, basic_chirp, device),
                tf.zeros((1, M), dtype=tf.complex64),
            ],
            axis=0,
        )

        # A dechirp is simply the conjugate of the upchirp - predefined to make the dechirping operation faster
        basic_dechirp = tf.math.conj(basic_chirp)

        # Simulation parameters, the number of symbol for 5% tolerance
        N = int(2e6)
        batch_size = int(1e4)  # Number of symbols per batch
        nr_of_batches = int(N / batch_size)  # NB: N must be divisible by batch_size

        snr_values = tf.linspace(-16, -4, 13)  # SNR values to test
        snr_values = tf.reverse(snr_values, axis=[0])

        rate_params = [0.25, 0.5, 0.7, 1]  # Poisson rate parameters
        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant
        noise_power = tf.constant((k_b * 298.16 * BW), dtype=tf.float64)  # dB
        result_list = tf.zeros(len(snr_values), len(rate_params), dtype=tf.int32)

        start_time = time.time()

        @tf.function
        def process_batch(snr, rate_param):
            # Generate the interfering users users symbols and their distances
            interfering_users_tx, distance = lora.generate_interferer_symbols(
                batch_size, rate_param, M, upchirp_lut
            )

            # Calculate the power of the user from the specified SNR
            user_power = 1 / tf.pow(10, snr / 10.0) * noise_power  # W

            # Generate the user message and look up the upchirp
            msg_tx = tf.random.uniform(
                (batch_size,), minval=0, maxval=M, dtype=tf.int32
            )
            user_chirp_tx = tf.squeeze(tf.gather(upchirp_lut, msg_tx, axis=0))

            # Generate complex AWGN channel
            complex_noise = tf.sqrt(noise_power / 2.0) * tf.complex(
                tf.random.normal((batch_size, M), dtype=tf.float64),
                tf.random.normal((batch_size, M), dtype=tf.float64),
            )

            # Compute the FSPL from the distannce of the interfering users
            fspl = (
                20 * tf.math.log(distance, 10) + 20 * tf.math.log(freq_eu, 10) - 147.55
            )

            # Combine the channel and interfering users
            upchirp_tx = (
                user_power * user_chirp_tx
                + tf.reduce_sum(interfering_users_tx, axis=0)
                + complex_noise
            )

            # Dechirp by multiplying the upchirp with the basic dechirp and run the FFT to get frequency components
            dechirp_rx = tf.multiply(upchirp_tx, basic_dechirp)
            fft_result = tf.abs(tf.signal.fft(dechirp_rx))

            # Decode the message using argmax, NB. should be replaced with CNN
            msg_rx = tf.argmax(fft_result, axis=1, output_type=tf.int32)

            # Calculate the number of errors in batch
            msg_tx = tf.squeeze(msg_tx)
            batch_result = tf.math.count_nonzero(msg_tx != msg_rx)

            return batch_result

        for i, rate_params in enumerate(rate_params):
            for j, snr in enumerate(snr_values):
                for batch in range(nr_of_batches):
                    print(
                        f"\tSNR {snr}: Batch {batch+1}/{nr_of_batches}. Total:{(nr_of_batches*i+(batch+1))}/{nr_of_batches*len(snr_values)} batches"
                    )
                    tf_snr = tf.constant(snr)
                    result_list[i, j] += process_batch(tf_snr, rate_params[i])
                    # Setup - Start the timer - mostly for fun
                print(
                    f"SNR: {snr}dB, error count: {result_list[i, j]}, SER: {result_list[i, j]/N:E}"
                )
        print(f"Time taken: {time.time() - start_time}")

        # Stacks and casts the results to a tensor for saving
        SF_list = tf.fill([len(snr_values)], tf.cast(SF, tf.float64))
        N_list = tf.fill([len(snr_values)], tf.cast(N, tf.float64))
        res_list = tf.stack(result_list)
        res_list = tf.cast(res_list, tf.float64)
        snr_list = tf.cast(snr_values, tf.float64)
        ser_list = tf.divide(res_list, N)
        output = tf.stack((SF_list, snr_values, res_list, N_list, ser_list), axis=0)

        output_path = os.path.abspath(os.path.join(file_path, "output"))
        os.makedirs(output_path, exist_ok=True)
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"{output_path}/{time_str}_SER_simulations_results_SF{SF}.txt"
        head = (
            f"Test done: {time_str} - "
            f"time taken: {time.time() - start_time} \n"
            f"SF, SNR, Error count, Number of symbols, SER"
        )
        savetxt(file_name, output.numpy().T, delimiter=",", header=head)

        plt.plot(snr_values, res_list / N, marker="^", linestyle="dashed")
        plt.yscale("log")
        plt.xlabel("SNR [dB]")
        plt.ylabel("SER")
        plt.legend(["SF" + str(SF)])
        plt.title("SER vs SNR")
        plt.savefig(f"{output_path}/{time_str}_SER_simulations_results_SF{SF}.png")
        plt.show()
