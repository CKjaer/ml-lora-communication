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

        # Create the basic chirp
        basic_chirp = lora.create_basechirp(M)

        # Create a LUT for the upchirps for faster processing - A final 0 row is created for intererers to look up
        upchirp_lut = tf.concat(
            [
                lora.upchirp_lut(M, basic_chirp),
                tf.zeros((1, M), dtype=tf.complex64),
            ],
            axis=0,
        )

        # Conjugate the basic chirp for basic dechirp
        basic_dechirp = tf.math.conj(basic_chirp)

        # Simulation parameters
        relative_error = 0.01
        max_ser = 1e-5
        n_symbols = int(tf.math.ceil(1 / (relative_error * max_ser)))
        batch_size = int(250e3)  # Number of symbols per batch
        nr_of_batches = int(n_symbols // batch_size)
        snr_val = tf.constant(-6, dtype=tf.float64)  # dB
        rate_param = tf.constant(0.25, dtype=tf.float64)  #
        sir_vals = tf.cast(tf.linspace(-10, 10, 11), dtype=tf.float64)  # dB
        result_list = tf.zeros(sir_vals.shape, dtype=tf.float64)

        # Noise formula based on thermal noise N0=k*T*B
        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant
        noise_power = tf.constant((k_b * 298.16 * BW), dtype=tf.float64)  # dB

        print(f"Running sim for a total of {n_symbols} symbols per SIR")

        start_time = time.time()

        for i in tf.range(len(sir_vals)):
            for batch in tf.range(nr_of_batches):
                sir_start_time = time.time()
                # Generate the user message and look up the upchirps
                msg_tx = tf.random.uniform(
                    (batch_size,), minval=0, maxval=M, dtype=tf.int32
                )

                sir_value = int(sir_vals[i].numpy())
                sir_tuple = (sir_value, sir_value, False)

                chirped_rx = lora.process_batch(
                    upchirp_lut,
                    rate_param,
                    snr_val,
                    msg_tx,
                    batch_size,
                    M,
                    noise_power,
                    sir_tuple,
                )

                # Dechirp by multiplying the upchirp with the basic dechirp
                dechirped_rx = lora.dechirp(chirped_rx, basic_dechirp)

                # Run the FFT to demodulate
                fft_result = tf.abs(tf.signal.fft(dechirped_rx))

                # Decode the message using argmax
                msg_rx = model.detect(fft_result, snr_val, M, noise_power)

                # Calculate the number of errors in batch
                msg_tx = tf.squeeze(msg_tx)
                batch_result = tf.math.count_nonzero(msg_tx != msg_rx)

                # Update the result list
                result_list = tf.tensor_scatter_nd_add(
                    result_list, [[i]], [batch_result]
                )
            print(
                f"SIR: {sir_vals[i]} dB, error count: {result_list[i]} SER: {result_list[i]/n_symbols:E}, time: {time.time() - sir_start_time}"
            )
        print(f"Simulation duration: {time.time() - start_time:.2f} s")

        # Stack and cast the results to float64
        SF_list = tf.fill([len(sir_vals)], tf.cast(SF, tf.float64))
        N_list = tf.fill([len(sir_vals)], tf.cast(n_symbols, tf.float64))
        ser_list = tf.divide(result_list, n_symbols)
        output = tf.stack([SF_list, sir_vals, result_list, N_list, ser_list], axis=0)

        # Save the results to a .txt file
        file_path = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.abspath(os.path.join(file_path, "sim_output/sir_sims"))
        os.makedirs(output_path, exist_ok=True)
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"{output_path}/{time_str}_SIR_simulations_results_SF{SF}_rate{rate_param.numpy()}.txt"
        head = (
            f"Test done: {time_str} - "
            f"time taken: {time.time() - start_time} \n"
            f"SF, SIR, error count, simulated symbols, SER"
        )
        savetxt(file_name, output.numpy().T, delimiter=",", header=head)
        
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Palatino Linotype'
        plt.rcParams['font.family'] ='Palatino Linotype'
        fs = 20
        plt.rcParams.update({'font.size': fs})


        # Plot SER curves as function of SIR
        figure = plt.figure(figsize=(8, 6))

        plt.plot(
            sir_vals,
            ser_list,
            marker="v",
            linestyle="dashed",
            color="black",
            label=f"SF{SF}, λ={rate_param.numpy():.2f}, SNR={snr_val} dB",
            clip_on=False,
            markevery=1,
        )

        plt.yscale("log")
        plt.xlabel("SIR [dB]")
        plt.ylabel("SER")
        plt.grid(True, which="both", alpha=0.5)
        plt.xlim(-10, 10)
        plt.ylim(1e-5, 1)
        plt.xticks(tf.range(-10, 12, 2))
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"{output_path}/{time_str}_SIR_simulations_results_SF{SF}_rate{rate_param.numpy()}.pdf",
            format = "pdf",
            bbox_inches = "tight"
        )
