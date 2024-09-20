import matplotlib.pyplot as plt
import time
from numpy import savetxt
import os
import math as m
import lora_phy as lora
import model_space as model
from json_handler import json_handler
import tensorflow as tf

if __name__ == '__main__':
    setup_path = os.path.join(os.getcwd(), "test.json")
    setup_file = json_handler(setup_path)
    snr_values = setup_file.get_snr_values()
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
        SF = setup_file.get_spreading_factor() # Spreading factor 
        M = int(2**SF) # Number of symbols per chirp 
        
        # Create the basic chirp - Formula based on "Efficient Design of Chirp Spread Spectrum Modulation for Low-Power Wide-Area Networks" by Nguyen et al.
        basis_chirp = lora.create_basechirp(M,device)
        # Once the basic chirp is created, we can create the upchirp LUT, as it is faster than calculating it on the fly
        upchirp = lora.upchirp_lut(M,basis_chirp,device)
        # A dechirp is simply the conjugate of the upchirp - predefined to make the dechirping operation faster
        basic_dechirp = tf.math.conj(basis_chirp)

        #Setup file structure for saving the results
        test_id = setup_file.get_test_id()
        og_path = os.path.join(os.getcwd(), test_id)

        if not os.path.exists(og_path):
            os.mkdir(og_path)

        # As the number of samples become quite large, the simulation is split into batches.
        # N is the number of symbols to simulate. At least 1e6 samples are nessecary for the simulation
        # to be accurate at high SNR values.
        # A reasonable batchsize value is 500e3
        N_samp = setup_file.number_of_samples()
    
        snr_values = setup_file.get_snr_values() # SNR values to test
        N_snrs = len(snr_values)
        start_time = time.time()

        for i, snr in enumerate(snr_values):
            for j in range(M):
                print(f"SNR {snr}: Symbol {j}/{M}. Currently at:{(M*i+j)}/{M*N_snrs} snr/symbol combinations")                # Ensure that the last batch is the correct size

                # Setup - Start the timer - mostly for fun
                beginning_time = time.time()

                #Chirp by selecting the message indexes from the lut, adding awgn and then dechirping
                #Gather indexes the list from the LUT. Squeeze removes an unnecessary dimension
                uptable = tf.squeeze(tf.transpose(tf.gather(upchirp, tf.repeat(j,N_samp), axis=1)))
                awgn = lora.channel_model(snr_values[i], N_samp, M, device)
                upchirp_tx = tf.add(uptable, awgn)
                #Dechirp by multiplying the upchirp with the basic dechirp
                dechirp_rx = tf.multiply(upchirp_tx, basic_dechirp)
                fft_result = tf.abs(tf.signal.fft(dechirp_rx))
                
                file_name = og_path+"/"+f"snr_{snr}_symbol_{j}.csv"
                savetxt(file_name, fft_result, delimiter=',')

                end_time = time.time()
                print(f"Time taken for symbol {j} of SNR {snr_values[i]}: {end_time - beginning_time}")