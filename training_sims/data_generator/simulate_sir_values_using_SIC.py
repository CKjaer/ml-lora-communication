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

        # Conjugate the basic chirp for baic dechirp
        basic_dechirp = tf.math.conj(basic_chirp)

        # Simulation parameters, the number of symbols simulated results in a 5% tolerance for max. SER
        relative_error = 0.05
        max_ser = 1e-5
        n_symbols = int(tf.math.ceil(1 / (relative_error * max_ser)))
        batch_size = int(50e3)  # Number of symbols per batch
        nr_of_batches = int(n_symbols // batch_size)
        snr_val = tf.constant(-6.6, dtype=tf.float64)  # dB
        rate_param = tf.constant(0.25, dtype=tf.float64)  #
        #sir_vals = tf.cast(tf.linspace(6, 6, 1), dtype=tf.float64)  # dB
        sir_vals = tf.cast(tf.linspace(-10, 10, 11), dtype=tf.float64)  # dB
        result_list = tf.zeros(sir_vals.shape, dtype=tf.float64)

        # Noise formula based on thermal noise N0=k*T*B
        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant
        noise_power = tf.constant((k_b * 298.16 * BW), dtype=tf.float64)  # dB

        print(f"Running sim for a total of {n_symbols} symbols per SIR")

        start_time = time.time()

        for i in tf.range(len(sir_vals)):
            for batch in tf.range(nr_of_batches):
                print(f"SIR: {sir_vals[i]}, batch : {batch} of {nr_of_batches}")
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
                #fft_result = tf.abs(tf.signal.fft(dechirped_rx))
                s = tf.exp(tf.complex(0.0,0.0))
                s = tf.fill(dechirped_rx.shape,s)
                fft_result = tf.signal.fft(dechirped_rx)
                abs_fft_result = tf.abs(fft_result)

                # Decode the message using argmax
                msg_rx = model.detect(abs_fft_result, snr_val, M, noise_power)

                #msg_rx_str = msg_rx_str*tf.repeat(tf.sqrt(1/M))
                
                indices = tf.stack([tf.range(tf.shape(msg_rx)[0]), msg_rx], axis=1)
                msg_rx_str = tf.gather_nd(abs_fft_result, indices)
                #msg_rx_str = tf.repeat(msg_rx_str,M).reshape(batch_size,M)#*tf.sqrt(1/M)

                print(f"abs_fft_result_shapte {abs_fft_result.shape}")
                #msg_rx_str = tf.gather(abs_fft_result, msg_rx, axis = 0)
                print(f"gathered msg_rx: {msg_rx.shape}")
                print(f"gathered msG_rx_str: {msg_rx_str.shape}")
                print(f"Detected signal: {indices[0]}, {msg_rx[0]}")

                


                int_sigs = tf.gather(upchirp_lut, msg_rx)
                msg_rx_str = tf.cast(msg_rx_str, dtype=chirped_rx.dtype)
                int_sigs = (int_sigs * msg_rx_str)
                print(f"Detected signal: {msg_rx[0]}")
                print(f"Strength of signal {msg_rx[0]}: {msg_rx_str[0]}")
                chirped_rx2 = chirped_rx - int_sigs
                dechirped_rx2 = lora.dechirp(chirped_rx2, basic_dechirp)
                fft_result2 = tf.signal.fft(dechirped_rx2)
                abs_fft_result2 = tf.abs(fft_result2)
                msg_rx2 = model.detect(abs_fft_result2, snr_val, M, noise_power)

                
                int_sigs_dechirped = lora.dechirp(chirped_rx2, basic_dechirp)
                int_sigs_ffted = tf.abs(tf.signal.fft(int_sigs_dechirped))*0.05

                
                if True:
                    plt.subplots(1,2,figsize=(5,10))
                    plt.subplot(1,2,1)
                    plt.plot(abs_fft_result[0],'-o')
                    plt.plot(int_sigs_ffted[0],'-o')
                    m1 = tf.argmax(abs_fft_result[0])
                    plt.axvline(m1, c='r',linestyle='--')
                    plt.legend(["abs","desc"])

                    plt.subplot(1,2,2)
                    plt.plot(abs_fft_result2[0],'-o')
                    plt.plot(int_sigs_ffted[0],'-o')
                    m2 = tf.argmax(abs_fft_result2[0])
                    plt.axvline(m2, c='r',linestyle='--')
                    plt.legend(["abs","desc"])
                    plt.show()
                    print(f"Detections: from real: {m1}, from abs: {m2}")

                

                # Calculate the number of errors in batch
                msg_tx = tf.squeeze(msg_tx)
                batch_result = tf.math.count_nonzero(msg_tx != msg_rx)

                # Update the result list
                result_list = tf.tensor_scatter_nd_add(
                    result_list, [[i]], [batch_result]
                )
            print(
                f"SIR: {sir_vals[i]} dB, error count: {result_list[i]} SER: {result_list[i]/n_symbols:E}"
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

        # Plot SER curves as function of SIR
        figure = plt.figure(figsize=(10, 5))

        plt.plot(
            sir_vals,
            result_list / n_symbols,
            marker="^",
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
            f"{output_path}/{time_str}_SIR_simulations_results_SF{SF}_rate{rate_param.numpy()}.png"
        )
