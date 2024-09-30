import time
from numpy import savetxt
import os
import lora_phy as lora
from json_handler import json_handler
import tensorflow as tf
import argparse


def main():
    for i in range(10): print("\n")
    # Setup of the argument parser. Eases multiple runs of the program
    desc = (
        "This program generates test data for the LoRa PHY model.\n"
        "The data is saved as CSV files in the specified output directory.\n"
        "The program requires a setup file in JSON format to run.\n"
        "Note that the program overwrites any existing files with the same name.\n"
    )
    parser = argparse.ArgumentParser(prog="LoRa Phy gen",description=desc)
    parser.add_argument("-c","--config_file", help="A json file containing the setup for the process. The setup file should contain the following fields: 'test_id', 'number_of_samples', 'snr_values' and 'spreading_factor'.", type=str,required=True)
    parser.add_argument("-o","--output_dir", help=f"Allows for a specified outputdir. Otherwise the default directory is the name of the config file",default="DEFAULT",type=str,required=False)
    parser.add_argument("-v","--verbose", help="Allows printing of data",action="store_true",default=False)

    args = parser.parse_args()
    config_file = args.config_file
    output_dir = args.output_dir
    verbose = args.verbose
    
    # Load the setup file
    try:
        setup_path = os.path.join(os.getcwd(), config_file)
        setup_file = json_handler(setup_path)
    except FileNotFoundError:
        print(f"File {config_file} not found. Check the path and try again.")
        exit()

    # Loads and print the setup data
    setup_file.print_setup_data()

    N_samp_raw = setup_file.number_of_samples()
    snr_values_raw = setup_file.get_snr_values()
    SF_raw = setup_file.get_spreading_factor()

    #Setup file structure for saving the results
    if output_dir == "DEFAULT":
        output_dir = setup_file.get_test_id()
    og_path = os.path.join(os.getcwd(), output_dir)
    os.makedirs(og_path, exist_ok=True)

    # Check if GPU is available - if it is, tensor flow runs on the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if verbose: print('Found GPU, using that')
        device = tf.device('/device:GPU:0')
    else:
        if verbose: print('GPU device not found, using CPU')
        device = tf.device('/device:CPU:0')

    with device:
        # LoRa PHY parameters (based on EU863-870 DR0 channel)
        SF = tf.constant(SF_raw, dtype=tf.int32)
        M = tf.constant(tf.pow(2,SF), dtype=tf.int32)
        N_samp = tf.constant(N_samp_raw, dtype=tf.int32)
        snr_values = tf.constant(snr_values_raw, dtype=tf.int32)
        
        # Precompute chirps
        basis_chirp = lora.create_basechirp(M,device)
        upchirp = lora.upchirp_lut(M,basis_chirp,device)
        basic_dechirp = tf.math.conj(basis_chirp)

        # Start the timer
        start_time = time.time()

        @tf.function
        def process_batch(snr, symbol):
            """Processing function for a batch of LoRa symbols. Creates a batch of upchirps, adds AWGN and dechirps the signal."""
            #Chirp by selecting the message indexes from the lut, adding awgn and then dechirping
            #Gather indexes the list from the LUT. Squeeze removes an unnecessary dimension
            upchirps = tf.squeeze(tf.transpose(tf.gather(upchirp, tf.repeat(symbol,N_samp), axis=1)))
            awgn = lora.channel_model(snr, N_samp, M, device)
            upchirps_with_noise = tf.add(upchirps, awgn)
            #Dechirp by multiplying the upchirp with the basic dechirp
            dechirp_rx = tf.multiply(upchirps_with_noise, basic_dechirp)
            fft_result = tf.abs(tf.signal.fft(dechirp_rx))
            return fft_result
        
        for i, snr in enumerate(snr_values):
            tf_snr = tf.constant(snr, dtype=tf.int32)
            if verbose: print(f"Processing SNR {snr} ({i + 1}/{len(snr_values)})")

            for j in tf.range(M):
                # Setup - Start the timer - mostly for fun
                beginning_time = time.time()

                tf_symbol = tf.constant(j, dtype=tf.int32)
                fft_result = process_batch(tf_snr,tf_symbol)
                
                file_name = og_path+"/"+f"snr_{snr}_symbol_{j}.csv"
                savetxt(file_name, fft_result, delimiter=',')

                end_time = time.time()
                if verbose: print(f"Processed symbol {j}/{M} for SNR {snr} in {end_time - beginning_time:.4f} seconds - Total: {j+i*M}/{M*len(snr_values)}")

        if verbose: print(f"Total processing taken: {time.time() - start_time}")

if __name__ == '__main__':
    main()