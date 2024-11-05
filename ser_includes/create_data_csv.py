import os
import sys
import time
from numpy import savetxt
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
import ser_includes.lora_phy as lora
import tensorflow as tf
import argparse
import json
import logging

def log_and_print(log:logging, message:str):
    log.info(message)
    print(message)

def create_data_csvs(log:logging, N_samples:int, snr_values:int, SF:int, output_dir:str, rate, SIR_Random, SIR_setup, verbose:bool=True):
    # Check if GPU is available - if it is, tensor flow runs on the GPU
    log.name = "LoRa Phy gen"
    log_and_print(log,"Starting the csv generation")
    log_and_print(log,f"Available logical devices: {tf.config.list_logical_devices('GPU')}")

    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        if verbose: log_and_print(log,'Found GPU, using that')
        device = tf.device(gpus[0].name)
    else:
        if verbose: log_and_print(log,'GPU device not found, using CPU')
        device = tf.device('/device:CPU:0')

    with device:
        # LoRa PHY parameters (based on EU863-870 DR0 channel)
        SF = tf.constant(SF, dtype=tf.int32)
        M = tf.constant(tf.pow(2,SF), dtype=tf.int32)

        N_samp_array = tf.constant(N_samples, dtype=tf.int32)
        snr_values = tf.constant(snr_values, dtype=tf.int32)
        
        # Precompute chirps
        basic_chirp = lora.create_basechirp(M)
        upchirp_lut = tf.concat(
            [
                lora.upchirp_lut(M, basic_chirp),
                tf.zeros((1, M), dtype=tf.complex64),
            ],
            axis=0,
        )

        rate_params = tf.convert_to_tensor(rate,dtype=tf.float32)
        basic_dechirp = tf.math.conj(basic_chirp)
        
        # SIR tuple: (SIR_min, SIR_max, SIR_random)
        SIR_tuple = (SIR_setup[0], SIR_setup[1], SIR_Random)
        # Hardcoded for now, will change
        
        BW = 250e3
        k_b = tf.constant(1.380649e-23, dtype=tf.float64)  # Boltzmann constant
        noise_power = tf.constant((k_b * 298.16 * BW), dtype=tf.float64)  # dB
        # Start the timer
        start_time = time.time()
        for _, current_rate in enumerate(rate_params):
            log_and_print(log,f"Current Rate Param: {current_rate}")
            for i, snr in enumerate(snr_values):
                N_samp = N_samp_array[i]
                tf_snr = tf.constant(snr, dtype=tf.int32)
                log_and_print(log,f"Processing SNR {snr} ({i + 1}/{len(snr_values)})")
                for j in tf.range(M):
                    # Setup - Start the timer - mostly for fun
                    tf_symbol = tf.constant(j, dtype=tf.int32)

                    msg_tx = tf.fill((N_samp,),tf_symbol)

                    chirped_rx = lora.process_batch(upchirp_lut, current_rate, tf_snr, msg_tx, N_samp, M, noise_power, SIR_tuple)

                    #Dechirp by multiplying the upchirp with the basic dechirp
                    dechirped_rx = lora.dechirp(chirped_rx, basic_dechirp)

                    # Run the FFT to demodulate
                    fft_result = tf.abs(tf.signal.fft(dechirped_rx))
                    
                    file_name = output_dir+"/"+f"snr_{snr}_symbol_{j}_rate_{current_rate:.2}.csv"
                    savetxt(file_name, fft_result, delimiter=';')

                    #log_and_print(log,f"\tProcessed symbol {j}/{M} for SNR {snr} in {end_time - beginning_time:.4f} seconds - Total: {j+i*M}/{M*len(snr_values)}")
            log_and_print(log,f"Total CSV creation time: {time.time() - start_time}")

if __name__ == '__main__':

    output_dir = time.strftime("%Y%m%d-%H%M%S")

    logfilename = "generate.log"
    outputdir = os.path.join(os.path.dirname(__file__), "output", output_dir)
    os.makedirs(outputdir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(outputdir, logfilename))
    logger = logging.getLogger("LoRa Phy gen")
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO)
    logger.info("Starting the program")
    # Load the setup file
    N_samp_raw = [10]
    snr_values_raw = [-4,-8,-12,-16]
    SF_raw = 7
    rate = [0.5]
    SIR_Random = False

    og_path = os.path.join(outputdir,"csv")
    os.makedirs(og_path, exist_ok=True)

    if len(N_samp_raw) == 1:
        N_samp_raw = N_samp_raw * len(snr_values_raw)
    elif len(N_samp_raw) != len(snr_values_raw):
        raise TypeError(
            f"Number_of_samples has invalid dimensions: Must be either 1, or the same as snr_values ({len(snr_values_raw)})"
        )
    create_data_csvs(logger, N_samp_raw, snr_values_raw, SF_raw, og_path, rate, SIR_Random, [0,7.5], verbose=True)