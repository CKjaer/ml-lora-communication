import sys
import os
file_path = os.path.dirname(__file__)
include_dir = os.path.abspath(os.path.join(file_path,'../include'))
sys.path.append(include_dir)

import lora_phy as lora
import model_space as model

import matplotlib.pyplot as plt
import time
from numpy import savetxt
import os
import math as m

import tensorflow as tf


if __name__ == '__main__':
    # Check if GPU is available - if it is, tensor flow runs on the GPU
    # Otherwise, run it on a CPU :)
    device = tf.device('')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('Found GPU, using that')
        device = tf.device('/device:GPU:0')
    else:
        print('GPU device not found, using CPU')
        device = tf.device('/device:CPU:0')


    with device:
        # LoRa PHY parameters (based on EU863-870 DR0 channel)
        SF = 7 # Spreading factor 
        BW = 250e3 # Bandwidth [Hz] - Not used, but might be used if we use a different chirp calculation method
        M = int(2**SF) # Number of symbols per chirp
        
        # Create the basic chirp - Formula based on "Efficient Design of Chirp Spread Spectrum Modulation for Low-Power Wide-Area Networks" by Nguyen et al.
        basis_chirp = lora.create_basechirp(M,device)
        # Once the basic chirp is created, we can create the upchirp LUT, as it is faster than calculating it on the fly
        upchirp = lora.upchirp_lut(M,basis_chirp,device)
        # A dechirp is simply the conjugate of the upchirp - predefined to make the dechirping operation faster
        basic_dechirp = tf.math.conj(basis_chirp)

        # As the number of samples become quite large, the simulation is split into batches.
        # N is the number of symbols to simulate. At least 1e6 samples are nessecary for the simulation
        # to be accurate at high SNR values.
        # A reasonable batchsize value is 500e3
        N = int(1e7)
        batch_size = int(1e6)
        nr_of_batches = int(N/batch_size)
        #^Note: currently N MUST be divisible by N :)

        # Generate a list of SNR values to test for, alongside a list to store the results
        snr_values = tf.linspace(-16, -4, 13)
        print(f"Testing for following SNR values: {snr_values}")
        snr_values = tf.reverse(snr_values, axis=[0])
        result_list = [0]*len(snr_values)
        #Sets a start timer for the simulation
        start_time = time.time()

        @tf.function
        def process_batch(snr):
            #Create an array of random symbols
            msg_tx = tf.random.uniform((batch_size,), minval=0, maxval=M, dtype=tf.int32)

            #Chirp by selecting the message indexes from the lut, adding awgn and then dechirping
            #Gather indexes the list from the LUT. Squeeze removes an unnecessary dimension
            uptable = tf.squeeze(tf.transpose(tf.gather(upchirp, msg_tx, axis=1)))
            awgn = lora.channel_model(snr, batch_size, M, device)
            upchirp_tx = tf.add(uptable, awgn)
            #Dechirp by multiplying the upchirp with the basic dechirp
            dechirp_rx = tf.multiply(upchirp_tx, basic_dechirp)
            fft_result = tf.abs(tf.signal.fft(dechirp_rx))
            #Use model to recognize the message
            msg_rx = model.detect(fft_result,device)
            #Calculate the number of errors
            msg_tx = tf.squeeze(msg_tx)
            batch_result = tf.math.count_nonzero(msg_tx != msg_rx)

            return batch_result
        
        for i, snr in enumerate(snr_values):
            for batch in range(nr_of_batches):
                beginning_time = time.time()
                print(f"\tSNR {snr}: Batch {batch+1}/{nr_of_batches}. Total:{(nr_of_batches*i+(batch+1))}/{nr_of_batches*len(snr_values)} batches")
                tf_snr = tf.constant(snr)
                result_list[i] += process_batch(tf_snr)
                # Setup - Start the timer - mostly for fun
                end_time = time.time()
                print(f"\t\tBatch time: {end_time - beginning_time}")
            print(f'SNR: {snr}dB, error count: {result_list[i]}, SER: {result_list[i]/N:E}')
        print(f'Time taken: {time.time() - start_time}')


        #Stacks and casts the results to a tensor for saving
        SF_list = tf.fill([len(snr_values)],tf.cast(SF,tf.float64))
        N_list = tf.fill([len(snr_values)],tf.cast(N,tf.float64))
        res_list = tf.stack(result_list)
        res_list = tf.cast(res_list, tf.float64)
        snr_list = tf.cast(snr_values, tf.float64)
        ser_list = tf.divide(res_list,N)
        output = tf.stack((SF_list, snr_values, res_list, N_list, ser_list),axis=0)

        output_path = os.path.abspath(os.path.join(file_path,"output"))
        os.makedirs(output_path, exist_ok=True)
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"{output_path}/{time_str}_SER_simulations_results_SF{SF}.txt"
        head = (
            f"Test done: {time_str} - "
            f"time taken: {time.time() - start_time} \n"
            f"SF, SNR, Error count, Number of symbols, SER"
            )
        savetxt(file_name, output.numpy().T, delimiter=',',header=head)

        plt.plot(snr_values, res_list/N,marker='^',linestyle='dashed')
        plt.yscale('log')
        plt.xlabel('SNR [dB]')
        plt.ylabel('SER')
        plt.legend(['SF'+str(SF)])
        plt.title('SER vs SNR')
        plt.savefig(f"{output_path}/{time_str}_SER_simulations_results_SF{SF}.png")
        plt.show()