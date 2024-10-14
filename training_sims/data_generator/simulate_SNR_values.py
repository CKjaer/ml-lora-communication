import os
import lora_phy as lora
import model_space as model
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
        SIR_tuple = (1, 10, True) # Set to min=max for constant SIR

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

        # Simulation parameters, the number of symbols simulated results in a 5% tolerance for SER of 1e-5
        N = int(1e2)  # int(2e6)
        batch_size = int(10)  # Number of symbols per batch
        nr_of_batches = int(N / batch_size)  # NB: N must be divisible by batch_size

        snr_values = tf.cast(tf.linspace(-4, -16, 7), dtype=tf.float64)
        rate_params = tf.constant([0, 0.25, 0.5, 0.7 ,1], dtype=tf.float64)
        result_list = tf.zeros((snr_values.shape[0], rate_params.shape[0]), dtype=tf.float64)

        #Noise formula based on thermal noise N0=k*T*B
        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant
        noise_power = tf.constant((k_b * 298.16 * BW), dtype=tf.float64)  # dB

        start_time = time.time()

        for i in tf.range(len(rate_params)):
            for j in tf.range(len(snr_values)):
                error_count = 0
                for batch in tf.range(nr_of_batches):
                    # Generate the user message and look up the upchirps
                    msg_tx = tf.random.uniform(
                        (batch_size,), minval=0, maxval=M, dtype=tf.int32
                    )

                    chirped_rx = lora.process_batch(
                        upchirp_lut,
                        rate_params[i],
                        snr_values[j],
                        msg_tx,
                        batch_size,
                        M,
                        noise_power,
                        SIR_tuple)
                    
                    #Dechirp by multiplying the upchirp with the basic dechirp
                    dechirped_rx = lora.dechirp(chirped_rx,basic_dechirp)

                    # Run the FFT to demodulate
                    fft_result = tf.abs(tf.signal.fft(dechirped_rx))

                    # Decode the message using argmax
                    msg_rx = model.detect(fft_result)

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
                    f"SNR: {snr_values[j]} dB, error count: {tf.gather_nd(result_list, [[j, i]])} SER: {result_list[j, i]/N:E}"
                )
        print(f"Simulation duration: {time.time() - start_time}")

        # Stack and cast the results to float64
        SF_list = tf.fill([len(snr_values)], tf.cast(SF, tf.float64))
        N_list = tf.fill([len(snr_values)], tf.cast(N, tf.float64))
        snr_list = tf.cast(snr_values, tf.float64)

        # Save the results to a .txt file for every rate parameter and create a plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for i, rate_param in enumerate(rate_params):
            # plt.subplot(2,2,i+1)
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
            
            # Plot SER curves as function of SNR
            if i > 0:
                axs = axs.flatten()
                ax = axs[i-1]
                ax.plot(snr_values, result_list[:, 0] / N, marker="o", linestyle="dashed", label=f"SF{SF}, λ=0.00", color='black') # Classical decoder with λ=0
                ax.plot(snr_values, result_list[:, i] / N, marker="^", linestyle="dashed", label=f"SF{SF}, λ={rate_param.numpy():.2f}", linefmt='k-', color='black') # Poisson decoder with λ=rate_param
                ax.set_yscale("log")
                ax.set_xlabel("SNR [dB]")
                ax.set_ylabel("SER")
                ax.grid(True, which='both', alpha=0.5)
                # ax.legend()
                ax.set_ylim(1e-5, 1)
                
                # Create an inset with the Poisson PMF
                inset_ax = inset_axes(ax, width="40%", height="30%", loc='lower left', bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
                poisson_dist = tf.random.poisson([batch_size], rate_param, dtype=tf.int32)
                poisson_values, idx, poisson_counts = tf.unique_with_counts(poisson_dist)
                poisson_count = poisson_counts / batch_size
                inset_ax.set_title(f"PMF, λ={rate_param.numpy():.2f}")
                inset_ax.set_xlabel(r'$N_i$', labelpad=-4)
                inset_ax.set_xlim([0, 10])
                inset_ax.set_ylim([0, 0.8])

                stem_inset = inset_ax.stem(poisson_values.numpy(), poisson_count.numpy(), basefmt=' ', linefmt='k-')
                for artist in stem_inset.get_children():
                    artist.set_clip_on(False)     



        plt.tight_layout()
        plt.savefig(f"{output_path}/{time_str}_SER_simulations_results_SF{SF}.png")
