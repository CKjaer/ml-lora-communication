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
    BW = 250e3 # Bandwidth [Hz]
    M = int(2**SF) # Number of symbols per chirp 

    print(f"Basic chirp specs: SF={SF}, bandwidth={BW} Hz, M={M}")
    basis_chirp = lora.create_basechirp(M)
    upchirp = lora.upchirp_lut(M,basis_chirp)
    basic_dechirp = basis_chirp.conj()

    N = int(1e7) # Number of symbols to simulate
    max_batch_size = 500e3
    nr_of_batches = int(N/max_batch_size)
    
    # Generate random symbols
    snr_values = np.linspace(-16, -4, 7) # SNR values to test
    result_list = np.zeros(len(snr_values))
    start_time = time.time()

    for i, snr in enumerate(snr_values):
        #Batches used to simulate lighten ram usage
        for batch in range(nr_of_batches):
            print(f"SNR {snr}: Batch {batch+1}/{nr_of_batches}. Total:{(nr_of_batches*i+(batch+1))}/{nr_of_batches*len(snr_values)} batches")
            batch_size = int(N/nr_of_batches)
            #Setup
            beginning_time = time.time()

            #Create random messages [M, N] array
            msg_tx = np.random.randint(0, M, batch_size)
            awgn = lora.gaussian_channel(snr_values[i], batch_size, M)

            #Chirp by selecting the message indexes from the lut, adding awgn and then dechirp
            upchirp_tx = upchirp[msg_tx, :] + awgn
            #Note: * = element wise multiplication - we want that, as upchirp_tx is [M, N] while dechirp is [M]
            dechirp_rx = upchirp_tx * basic_dechirp
            fft_result = np.abs(fft.fft(dechirp_rx))

            #Use model to recognize the message
            msg_rx = model.detect(fft_result)

            #Calculate the number of errors
            batch_result = np.sum(msg_tx != msg_rx)
            result_list[i] += batch_result

            end_time = time.time()
            print(f"\tBatch result: error count: {batch_result}, SER: {batch_result/batch_size:E}, batch time: {end_time - beginning_time}")
        print(f'SNR: {snr}dB, error count: {result_list[i]}, SER: {result_list[i]/N:E}')

    #Save the results to a file
    output = np.stack(([SF]*len(snr_values), snr_values, result_list, [N]*len(snr_values), result_list/N),axis=1)
    path = os.getcwd()+"/snr_estimate_output/"
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = path+time_str+"_SER_simulations_results_SF"+str(SF)+".txt"
    head = "Test done: " + time.strftime("%Y-%m-%d %H:%M:%S") +" - time taken: " + str(time.time() - start_time) + "\nSF, SNR, Error count, Number of symbols, SER"
    savetxt(file_name, output, delimiter=',',header=head)

    plt.plot(snr_values, result_list/N,marker='^',linestyle='dashed')
    plt.yscale('log')
    plt.xlabel('SNR [dB]')
    plt.ylabel('SER')
    plt.legend(['SF'+str(SF)])
    plt.title('SER vs SNR')
    plt.savefig(path+time_str+"_SER_simulations_results_SF"+str(SF)+".png")
    plt.show()