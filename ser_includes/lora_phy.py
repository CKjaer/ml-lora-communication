import tensorflow as tf
from math import pi


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
    lut_array = tf.TensorArray(dtype=basic_chirp.dtype, size=M, dynamic_size=False)
    lut_array = lut_array.write(0, basic_chirp)  # Write the basic chirp

    for i in tf.range(1, M):
        rolled_chirp = tf.roll(basic_chirp, -i, axis=0)
        lut_array = lut_array.write(i, rolled_chirp)

    return lut_array.stack()


@tf.function
def generate_interferer_symbols(batch_size, rate_param, M, upchirp_lut, user_amp, SIR_tuple):
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
    (SIR_min_dB, SIR_max_dB, random) = SIR_tuple

    # Draw interferers from Poisson distribution
    if random:
        n_interferers = tf.random.poisson([batch_size], rate_param, dtype=tf.int32)
    else:
        n_interferers = tf.fill([batch_size], 1)
        n_interferers = tf.cast(n_interferers, dtype=tf.int32)

    max_interferers = tf.reduce_max(n_interferers)
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

    # A random SIR value between min and max is sampled uniformly
    SIR_dB = tf.random.uniform([batch_size, max_interferers], SIR_min_dB, SIR_max_dB)
    SIR_lin = tf.pow(10.0, SIR_dB / 10.0)

    # Calculate interferer amplitudes based on SIR
    interferer_amp = tf.cast(user_amp, tf.complex64) / tf.sqrt(tf.cast(SIR_lin, tf.complex64))

    # Initialize the output tensor
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
    upchirp_lut, rate_param, snr, msg_tx, batch_size, M, noise_power, SIR_tuple
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
    
    complex_noise = generate_noise(tf.shape(user_chirp_tx), noise_power)
 
    # Channel coefficients
    snr = tf.cast(snr, dtype=tf.float64)
    snr_linear = tf.pow(tf.cast(10.0, dtype=tf.float64), snr / 10.0)
    user_amp = tf.sqrt(snr_linear * noise_power)

    # Generate the interfering users symbols and their distances
    if rate_param > 0:
        inter_symbols_scaled = generate_interferer_symbols(
            batch_size, rate_param, M, upchirp_lut, user_amp, SIR_tuple
        )
    else:
        inter_symbols_scaled = tf.zeros((batch_size, M), dtype=tf.complex64)

    # Combine the signals and add noise
    upchirp_tx = (
        tf.cast(user_amp, dtype=tf.complex64) * user_chirp_tx
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

