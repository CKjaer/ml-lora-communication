import matplotlib.pyplot as plt
import time
from numpy import savetxt
import os
import math as m
import lora_phy as lora
import model_space as model

import tensorflow as tf


if __name__ == '__main__':
    # Check if GPU is available - if it is, tensor flow runs on the GPU
    # Otherwise, run it on a CPU :)
    device = tf.device('')
    if tf.test.is_gpu_available('/device:GPU:0'):
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
        max_batch_size = int(500e3)
        nr_of_batches = int(N/max_batch_size)
        #^Note: currently N MUST be divisible by N :)

        # Generate a list of SNR values to test for, alongside a list to store the results
        snr_values = tf.linspace(-16, -4, 7)
        snr_values = tf.reverse(snr_values, axis=[0])
        result_list = [0]*len(snr_values)
        #Sets a start timer for the simulation
        start_time = time.time()

        for i, snr in enumerate(snr_values):
            for batch in range(nr_of_batches):
                print(f"SNR {snr}: Batch {batch+1}/{nr_of_batches}. Total:{(nr_of_batches*i+(batch+1))}/{nr_of_batches*len(snr_values)} batches")
                # Ensure that the last batch is the correct size
                batch_size = int(N/nr_of_batches)
                
                # Setup - Start the timer - mostly for fun
                beginning_time = time.time()

                #Create an array of random symbols
                msg_tx = tf.random.uniform((batch_size,), minval=0, maxval=M, dtype=tf.int32)

                #Chirp by selecting the message indexes from the lut, adding awgn and then dechirping
                #Gather indexes the list from the LUT. Squeeze removes an unnecessary dimension
                uptable = tf.squeeze(tf.transpose(tf.gather(upchirp, msg_tx, axis=1)))
                awgn = lora.channel_model(snr_values[i], batch_size, M, device)
                upchirp_tx = tf.add(uptable, awgn)
                #Dechirp by multiplying the upchirp with the basic dechirp
                dechirp_rx = tf.multiply(upchirp_tx, basic_dechirp)
                fft_result = tf.abs(tf.signal.fft(dechirp_rx))

                #Use model to recognize the message
                msg_rx = model.detect(fft_result,device)

                #Calculate the number of errors
                msg_tx = tf.squeeze(msg_tx)

                batch_result = tf.math.count_nonzero(msg_tx != msg_rx)
                result_list[i] += batch_result

                end_time = time.time()
                print(f"\tBatch result: error count: {batch_result}, SER: {batch_result/batch_size:E}, batch time: {end_time - beginning_time}")
            print(f'SNR: {snr}dB, error count: {result_list[i]}, SER: {result_list[i]/N:E}')
            print(f'SNR time taken: {time.time() - start_time}')

        #Stacks and casts the results to a tensor for saving
        res_list = tf.stack(result_list)
        SF_list = tf.stack([SF]*len(snr_values))
        SF_list = tf.cast(SF_list, tf.float64)
        N_list = tf.stack([N]*len(snr_values))
        res_list = tf.cast(res_list, tf.float64)
        N_list = tf.cast(N_list, tf.float64)
        output = tf.stack((SF_list, snr_values, res_list, N_list, tf.divide(res_list,N)),axis=0)
        path = os.getcwd()+"/snr_estimate_output/"
        os.mkdir(path)
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = path+time_str+"_SER_simulations_results_SF"+str(SF)+".txt"
        head = "Test done: " + time.strftime("%Y-%m-%d %H:%M:%S") +" - time taken: " + str(time.time() - start_time) + "\nSF, SNR, Error count, Number of symbols, SER"
        savetxt(file_name, output, delimiter=',',header=head)

        plt.plot(snr_values, res_list/N,marker='^',linestyle='dashed')
        plt.yscale('log')
        plt.xlabel('SNR [dB]')
        plt.ylabel('SER')
        plt.legend(['SF'+str(SF)])
        plt.title('SER vs SNR')
        plt.savefig(path+time_str+"_SER_simulations_results_SF"+str(SF)+".png")
        plt.show()