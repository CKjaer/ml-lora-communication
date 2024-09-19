import lora_phy as lora
import model_space as model
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.fft as fft
from numpy import savetxt
import os


if __name__ == '__main__':
    # LoRa PHY parameters (based on EU863-870 DR0 channel)
    SF = 7 # Spreading factor 
    M = int(2**SF) # Number of symbols per chirp 

    print(f"Basic chirp specs: SF={SF} Hz, M={M}")
    basis_chirp = lora.create_basechirp(M)
    upchirp = lora.upchirp_lut(M,basis_chirp)
    basic_dechirp = basis_chirp.conj()

    time_str = "test_data_fft_"+time.strftime("%Y_%m_%d_%H_%M_%S")
    og_path = os.path.join(os.getcwd(), time_str)
    os.mkdir(og_path)

    N_samp = int(1e3) # Number of symbols per combination
    N_snrs = 2 # Number of SNR values to test
    snr_min_dB = -16
    snr_max_dB = -4

    # Generate random symbols
    snr_values = np.linspace(snr_min_dB, snr_max_dB, N_snrs) # SNR values to test
    start_time = time.time()

    for i, snr in enumerate(snr_values):
        #Batches used to simulate lighten ram usage
        for j in range(M):
            print(f"SNR {snr}: Symbol {j}/{M}. Currently at:{(M*i+j)}/{M*N_snrs} snr/symbol combinations")

            #Setup
            beginning_time = time.time()

            #Create random messages [M, N] array
            awgn = lora.gaussian_channel(snr_values[i], N_samp, M)

            #Chirp by selecting the message indexes from the lut, adding awgn and then dechirp
            upchirp_tx = upchirp[j, :] + awgn
            #Note: * = element wise multiplication - we want that, as upchirp_tx is [M, N] while dechirp is [M]
            dechirp_rx = upchirp_tx * basic_dechirp
            fft_result = np.abs(fft.fft(dechirp_rx))

            file_name = og_path+"/"+f"snr_{snr}_symbol_{j}.csv"

            savetxt(file_name, fft_result, delimiter=';')
            end_time = time.time()
