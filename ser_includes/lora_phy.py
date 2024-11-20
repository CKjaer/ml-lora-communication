import tensorflow as tf
from math import pi
import numpy as np
import matplotlib.pyplot as plt


@tf.function
def create_basechirp(M):
    """Create the basic chirp

    Formula based on "Efficient Design of Chirp Spread Spectrum Modulation for Low-Power Wide-Area Networks" by Nguyen et al.
    The base chirp is calculated as exp(2*pi*1j*((n^2/(2*M)) - (n/2))).

    Returns:
        tf.complex64: A [M] tensor of complex64 values representing the basic chirp
    """
    n = tf.linspace(0, M - 1, M)
    n = tf.cast(n, tf.complex64)
    M_complex = tf.cast(M, tf.complex64)

    # We've split the calculation into two parts - mult1 handling 2*pi*1j and mult2 handling the rest of the formula
    mult1 = tf.constant(2 * pi * 1j, tf.complex64)

    # Mult 2 is calculated as (n^2/(2*M)) - (n/2)
    frac1 = tf.divide(tf.pow(n, 2), (2 * M_complex))
    frac2 = tf.divide(n, 2)
    mult2 = tf.subtract(frac1, frac2)
    return tf.exp(tf.multiply(mult1, mult2))


@tf.function
def upchirp_lut(M, basic_chirp):
    """Create a lookup table based on the basic chirp.
    Returns:
        tf.complex64: A [M, M] tensor of complex64 values representing the upchirps
    """
    # Sets up a tensor array with datatype of the basic chirp and size of M
    lut_array = tf.TensorArray(dtype=basic_chirp.dtype, size=M+1, dynamic_size=False)
    lut_array = lut_array.write(0, basic_chirp)  # Write the basic chirp

    for i in tf.range(1, M):
        rolled_chirp = tf.roll(basic_chirp, -i, axis=0)
        lut_array = lut_array.write(i, rolled_chirp)
    lut_array = lut_array.write(M, tf.zeros_like(basic_chirp))
    return lut_array.stack()


@tf.function
def generate_interferer_symbols(batch_size, rate_param, M, upchirp_lut, Pt, Pj, SIR_tuple):
    """
    Generate symbols for interferers based on a Poisson distribution.

    Args:
        batch_size (int): Number of batches to process.
        rate_param (float): Rate parameter for the Poisson distribution.
        M (int): Maximum value for random symbols and reference for LUT.
        upchirp_lut (tf.Tensor): LUT for symbols.
        user_amp (tf.complex64): Amplitude of the user signal.
        SIR_tuple (tuple): Tuple of (min SIR, max SIR, random SIR).
    
    Returns:
        tf.Tensor: Tensor of interferer power scaled by specified SIR.
    """
    (rmin, rmax, random) = SIR_tuple

    # Draw interferers from Poisson distribution
    if random:
        n_interferers = tf.random.poisson([batch_size], rate_param, dtype=tf.int32)
        max_interferers = tf.reduce_max(n_interferers)
            
        # Sample distances from a uniform distribution, and calculate uniform dist. over circle
        uniform_dist = tf.random.uniform([batch_size, max_interferers])
        dist = tf.sqrt(uniform_dist)*(rmax-rmin) + rmin
        #print(f"Distance limits: {rmin}, {rmax}")
        #print(f"Distance max value: {tf.reduce_max(dist)}, Distance min value: {tf.reduce_min(dist)}")

        # Generate Rayleigh fading coefficients
        real = tf.random.normal([batch_size, max_interferers], mean=0.0, stddev=1.0, dtype=tf.float32)
        imag = tf.random.normal([batch_size, max_interferers], mean=0.0, stddev=1.0, dtype=tf.float32)
        complex_gauss = tf.cast((1/tf.sqrt(tf.constant(2.0))), tf.complex64)*tf.complex(real, imag)
        ray = tf.abs(complex_gauss)
        #print(f"Ray min: {tf.reduce_min(ray)}, Ray max: {tf.reduce_max(ray)}, Ray mean: {tf.reduce_mean(ray)}, Ray sq mean: {tf.reduce_mean(tf.pow(ray, 2))}")

        eta = tf.constant(3.5, dtype=tf.float32)
        power_dist_loss = tf.pow(dist, -(eta))
        Pi = power_dist_loss * tf.pow(ray, 2.0)
        hi = tf.sqrt(Pi)
    
        # Calculate interferer amplitudes based on SIR
        Pt = tf.cast(Pt, tf.float32)
        interferer_amp = tf.cast(hi * tf.sqrt(Pt), tf.complex64)
    else:
        n_interferers = tf.fill([batch_size], 1)
        n_interferers = tf.cast(n_interferers, dtype=tf.int32)
        max_interferers = tf.reduce_max(n_interferers)

        SIRdB = tf.constant(rmax, dtype=tf.float64)
        SIR = tf.pow(tf.cast(10.0,tf.float64), SIRdB / 10.0)
        Pi = tf.cast(Pj / SIR, tf.complex64)
        interferer_amp = tf.fill([batch_size, 1], tf.sqrt(Pi))

    # Sequence mask creates an array of True and False based on how many interferers were drawn
    mask = tf.sequence_mask(n_interferers, max_interferers, dtype=tf.bool)

    # Generate random symbols and apply mask
    # Masking: Set symbols to M (indicating zero power in LUT) for unused positions
    # Gather symbols from the Look-Up Table (LUT)
    interferer_set_1 = tf.random.uniform([batch_size, max_interferers], minval=0, maxval=M, dtype=tf.int32)
    interferer_set_1 = tf.where(mask, interferer_set_1, M)
    interferer_set_1 = tf.gather(upchirp_lut, interferer_set_1, axis=0)

    #Repeat for interferer set 2
    interferer_set_2 = tf.random.uniform([batch_size, max_interferers], minval=0, maxval=M, dtype=tf.int32)
    interferer_set_2 = tf.where(mask, interferer_set_2, M)
    interferer_set_2 = tf.gather(upchirp_lut, interferer_set_2, axis=0)

    inter_symbols = tf.concat([interferer_set_1, interferer_set_2], axis = 2)

    del interferer_set_1, interferer_set_2, n_interferers

    # Generate random arrival times (shifts) for each batch
    rand_arrival = tf.random.uniform(
        [batch_size, max_interferers], minval=1, maxval=M - 1, dtype=tf.int32
    )

    half_shifted_inter = tf.zeros([batch_size, M], dtype=tf.complex64)

    # Scale and combine the interferer symbols
    for i in tf.range(max_interferers):
        cs = inter_symbols[:, i, :]
        ra = rand_arrival[:, i]
        shifted_symbol = tf.roll(cs, shift=-ra, axis=tf.ones_like(ra))[:,:M]
        chs = interferer_amp[:, i, tf.newaxis] * shifted_symbol
        half_shifted_inter += (chs)

    return half_shifted_inter


@tf.function
def process_batch(
    upchirp_lut, rate_param, snr, msg_tx, batch_size, M, PN, SIR_tuple
):
    """
    Processes a batch of LoRa symbols by adding noise and potential interference.
    Args:
        upchirp_lut (tf.Tensor): Lookup table for upchirp symbols.
        rate_param (int): Rate parameter for generating interfering symbols.
        snr (float): Signal-to-noise ratio.
        msg_tx (tf.Tensor): Tensor containing the transmitted message symbols.
        batch_size (int): Number of symbols in the batch.
        M (int): Symbols per chirtp.
        noise_power (float): Power of the noise to be added.
        SIR_tuple (tuple): Signal-to-interference ratio parameters.
    Returns:
        tf.Tensor: Tensor containing the processed symbols with noise and interference.
    """

    # Pick the contents of each symbol from the look up table
    user_chirp_tx = tf.gather(upchirp_lut, msg_tx, axis=0)

    complex_noise = generate_noise(tf.shape(user_chirp_tx), PN)
 
     # Transmission power - Adjusted to 62 dB - Change this value to adjust SIR curve.
    # Note that there are limits for which the model works
    Pt_dB = tf.constant(10.0**(-62.0/10.0))
    Pt = tf.cast(Pt_dB, dtype=tf.float64)

    # Channel coefficients
    snr = tf.cast(snr, dtype=tf.float32)
    snr_lin = tf.pow(10.0, snr / 10.0)
    # hj = sqrt((snr_lin * PN) / Pt)
    hj = tf.sqrt((tf.cast(snr_lin, tf.float64) * PN) / Pt)
    Pj = tf.pow(hj, 2.0) * Pt

    # Generate the interfering users symbols and their distances
    if rate_param > 0:
        inter_symbols_scaled = generate_interferer_symbols(
            batch_size, rate_param, M, upchirp_lut, Pt, Pj, SIR_tuple
        )
    else:
        inter_symbols_scaled = tf.zeros((batch_size, M), dtype=tf.complex64)

    # Combine the signals and add noise
    upchirp_tx = (
        tf.cast(hj, dtype=tf.complex64) * tf.cast(tf.sqrt(Pt),dtype=tf.complex64) * user_chirp_tx
        + inter_symbols_scaled
        + complex_noise
    )
    return upchirp_tx


@tf.function
def dechirp(upchirp_tx, basic_dechirp):
    dechirp_rx = tf.multiply(upchirp_tx, basic_dechirp)
    return dechirp_rx

@tf.function
def generate_noise(shape, noise_power):
    # P0 = k*T*B, #std = sqrt(var) - For white noise, PSD = var = P0
    # Because complex needs var, then var for 
    noise_stddev = tf.cast(tf.sqrt(noise_power / 2.0), dtype=tf.float32)

    # Generate complex AWGN channel
    noise_real = tf.random.normal(
        shape=shape,
        mean=0.0,
        stddev=noise_stddev,
        dtype=tf.float32,
    )

    noise_imag = tf.random.normal(
        shape=shape,
        mean=0.0,
        stddev=noise_stddev,
        dtype=tf.float32,
    )

    complex_noise = tf.complex(noise_real, noise_imag)
    return complex_noise

if __name__ == "__main__":
    M = 128
    basic_chirp = create_basechirp(M)
    upchirp_lutt = upchirp_lut(M, basic_chirp)
    rate_param = 1
    snr = -16
    batch_size = 50
    msg_tx = tf.random.uniform((batch_size,), minval=0, maxval=M, dtype=tf.int32)
    B = 250e3
    T = 298.16
    k = 1.38e-23
    noise_power = B*T*k
    #print(f"Noise power in dB: {10*np.log10(noise_power)}")
    SIR_tuple = (200, 1000, True)

    batch = process_batch(upchirp_lutt, rate_param, snr, msg_tx, batch_size, M, noise_power, SIR_tuple)
    basic_dechirp = tf.math.conj(basic_chirp)
    dechirped_bacth = dechirp(batch, basic_dechirp)
    plt.plot(np.abs(np.fft.fft(dechirped_bacth[0])))
    plt.show()
    #print("Done")