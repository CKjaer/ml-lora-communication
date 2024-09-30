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

        # Create the basic chirp - Formula based on "Efficient Design of Chirp Spread Spectrum Modulation for Low-Power Wide-Area Networks" by Nguyen et al.
        basic_chirp = lora.create_basechirp(M, device)
        # Once the basic chirp is created, we can create the upchirp LUT, as it is faster than calculating it on the fly
        upchirp = tf.concat(
            [
                lora.upchirp_lut(M, basic_chirp, device),
                tf.zeros((1, M), dtype=tf.complex64),
            ],
            axis=0,
        )

        # A dechirp is simply the conjugate of the upchirp - predefined to make the dechirping operation faster
        basic_dechirp = tf.math.conj(basic_chirp)

        # Simulation parameters
        N = int(
            1e7
        )  # Number of symbols to simulate is greater than 1e-5 SER for better accuracy
        batch_size = int(1e6)  # Number of symbols per batch
        nr_of_batches = int(N / batch_size)  # NB: N must be divisible by batch_size

        snr_values = tf.linspace(-16, -4, 13)  # SNR values to test
        snr_values = tf.reverse(snr_values, axis=[0])
        result_list = [0] * len(snr_values)

        inner_radius = 200  # Inner radius of the circular uniform distribution with no interfering users
        outer_radius = 1000  # Outer radius of the circular uniform distribution
        rate_params = [0.25, 0.5, 0.7, 1]  # Poisson rate parameters
        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant

        @tf.function
        def process_batch(snr, rate_param):
            # Symbols for interfering users

            # Draw the number of interferers from a Poisson distribution
            n_interferers = tf.random.poisson(
                (batch_size,), rate_params[0], dtype=tf.int32
            )

            # Create a 2 random symbols for each interferer
            max_interferers = tf.reduce_max(n_interferers).numpy()
            rand_inter_symbols = tf.random.uniform(
                (batch_size, max_interferers), minval=0, maxval=M, dtype=tf.int32
            )

            # Create a mask to zero out the symbols that are not used if the number of interferers is less than the max
            mask = tf.sequence_mask(n_interferers, max_interferers, dtype=tf.int32)

            # Fill the masked symbols with 128 pointing to a zero symbol in the LUT
            masked_symbols = tf.where(
                (rand_inter_symbols * mask) == 0, M, (rand_inter_symbols * mask)
            )

            # Gather the symbols from the LUT and reshape to stack symbols for each interferer column-wise
            inter_symbols_lut = tf.gather(upchirp, masked_symbols, axis=0)
            shaped_symbols = tf.reshape(inter_symbols_lut, (batch_size, 256))

            # Shift the symbols to with a random arrival time
            rand_arrival = tf.random.uniform(
                (batch_size,), minval=0, maxval=M, dtype=tf.int32
            )
            shifted_inter = tf.roll(
                shaped_symbols, shift=-rand_arrival, axis=tf.ones(batch_size, tf.int32)
            )

            # Slice to a singular symbol per interferer
            num_cols = range(max_interferers * M)
            # sorted_slice = tf.sort(tf.concat([num_cols[0::M], num_cols[1::M]], axis=0))
            sliced_tensors = tf.gather(shifted_inter, [num_cols], axis=1)

            user_power = snr + 10 * tf.math.log10(k_b * 298.16 * BW)
            noise_power = tf.pow(10.0, snr / 10.0)

            # Circular AWGN channel noise
            real_noise = tf.random.normal(
                shape=(batch_size,), mean=0.0, stddev=tf.sqrt(noise_power / 2)
            )
            img_noise = tf.random.normal(
                shape=(batch_size,), mean=0.0, stddev=tf.sqrt(noise_power / 2)
            )

            msg_tx = tf.random.uniform(
                (batch_size,), minval=0, maxval=M, dtype=tf.int32
            )  # Create an array of random symbols

            uptable = tf.squeeze(tf.gather(upchirp, msg_tx, axis=0))

            modulated_signal = (
                user_power * uptable
                + tf.reduce_sum(sliced_tensors, axis=1)
                + tf.sqrt(real_)
            )
            # awgn = lora.channel_model_new(snr, batch_size, M, device)

            # upchirp_tx = tf.add(uptable, awgn)
            # Dechirp by multiplying the upchirp with the basic dechirp
            dechirp_rx = tf.multiply(upchirp_tx, basic_dechirp)
            fft_result = tf.abs(tf.signal.fft(dechirp_rx))
            # Use model to recognize the message
            msg_rx = model.detect(fft_result, device)
            # Calculate the number of errors
            msg_tx = tf.squeeze(msg_tx)
            batch_result = tf.math.count_nonzero(msg_tx != msg_rx)

            return batch_result

        for i, snr in enumerate(snr_values):
            for batch in range(nr_of_batches):
                beginning_time = time.time()
                print(
                    f"\tSNR {snr}: Batch {batch+1}/{nr_of_batches}. Total:{(nr_of_batches*i+(batch+1))}/{nr_of_batches*len(snr_values)} batches"
                )
                tf_snr = tf.constant(snr)
                result_list[i] += process_batch(tf_snr, rate_params[i])
                # Setup - Start the timer - mostly for fun
                end_time = time.time()
                print(f"\t\tBatch time: {end_time - beginning_time}")
            print(
                f"SNR: {snr}dB, error count: {result_list[i]}, SER: {result_list[i]/N:E}"
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
