import os
import time
from numpy import savetxt
if os.getcwd().endswith("training_sims"):
    import lora_phy as lora
else:
    import training_sims.data_generator.lora_phy as lora
import tensorflow as tf
import argparse
import json
import logging

def log_and_print(log:logging, message:str):
    log.debug(message)
    print(message)


def create_data_csvs(log:logging, N_samples:int, snr_values:int, SF:int, output_dir:str, lamb:float, verbose:bool=True):
    # Check if GPU is available - if it is, tensor flow runs on the GPU
    log.name = "LoRa Phy gen"
    log_and_print(log,"Starting the csv generation")
    log_and_print(log,f"Available physical devices: {tf.config.list_physical_devices('GPU')}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if verbose: log_and_print(log,'Found GPU, using that')
        device = tf.device('/device:GPU:0')
    else:
        if verbose: log_and_print(log,'GPU device not found, using CPU')
        device = tf.device('/device:CPU:0')

    with device:
        # LoRa PHY parameters (based on EU863-870 DR0 channel)
        SF = tf.constant(SF, dtype=tf.int32)
        M = tf.constant(tf.pow(2,SF), dtype=tf.int32)
        N_samp = tf.constant(N_samples, dtype=tf.int32)
        snr_values = tf.constant(snr_values, dtype=tf.int32)
        
        # Precompute chirps
        basis_chirp = lora.create_basechirp(M)
        upchirp = lora.upchirp_lut(M,basis_chirp)
        basic_dechirp = tf.math.conj(basis_chirp)

        # Start the timer
        start_time = time.time()

        @tf.function
        def process_batch(snr, symbol):
            """Processing function for a batch of LoRa symbols. Creates a batch of upchirps, adds AWGN and dechirps the signal."""
            #Chirp by selecting the message indexes from the lut, adding awgn and then dechirping
            #Gather indexes the list from the LUT. Squeeze removes an unnecessary dimension
            upchirps = tf.squeeze(tf.transpose(tf.gather(upchirp, tf.repeat(symbol,N_samp), axis=1)))
            awgn = lora.channel_model(snr, N_samp, M)
            upchirps_with_noise = tf.add(upchirps, awgn)
            #Dechirp by multiplying the upchirp with the basic dechirp
            dechirp_rx = tf.multiply(upchirps_with_noise, basic_dechirp)
            fft_result = tf.abs(tf.signal.fft(dechirp_rx))
            return fft_result
        
        for i, snr in enumerate(snr_values):
            tf_snr = tf.constant(snr, dtype=tf.int32)
            log_and_print(log,f"Processing SNR {snr} ({i + 1}/{len(snr_values)})")

            for j in tf.range(M):
                # Setup - Start the timer - mostly for fun
                beginning_time = time.time()

                tf_symbol = tf.constant(j, dtype=tf.int32)
                fft_result = process_batch(tf_snr,tf_symbol)
                
                file_name = output_dir+"/"+f"snr_{snr}_symbol_{j}.csv"
                savetxt(file_name, fft_result, delimiter=';')

                end_time = time.time()
                #log_and_print(log,f"\tProcessed symbol {j}/{M} for SNR {snr} in {end_time - beginning_time:.4f} seconds - Total: {j+i*M}/{M*len(snr_values)}")
        log_and_print(log,f"Total CSV creation time: {time.time() - start_time}")

if __name__ == '__main__':
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
    
    logfilename = "test.log"
    file_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.abspath(os.path.join(file_path,logfilename))
    logger = logging.getLogger("LoRa Phy gen")
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG)
    logger.debug("Starting the program")
    # Load the setup file
    try:
        setup_path = os.path.abspath(os.path.join(os.getcwd(), config_file))
        logger.debug(setup_path)
        setup_json = json.load(open(setup_path))
    except FileNotFoundError:
        logger.error(logger,f"File {config_file} not found. Check the path and try again.")
        exit()

    # Loads and print the setup data
    N_samp_raw = setup_json.get("number_of_samples")
    snr_values_raw = setup_json.get("snr_values")
    SF_raw = setup_json.get("spreading_factor")

    #Setup file structure for saving the results
    if output_dir == "DEFAULT":
        output_dir = setup_json.get("test_id")
    og_path = os.path.abspath(os.path.join(file_path, "output",output_dir))
    os.makedirs(og_path, exist_ok=True)

    create_data_csvs(logger, N_samp_raw, snr_values_raw, SF_raw, og_path,0)