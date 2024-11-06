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
    # basic_chirp = tf.linspace(0, M-1, M)
    # basic_chirp = tf.cast(basic_chirp, tf.complex64)
    lut_array = tf.TensorArray(dtype=basic_chirp.dtype, size=M, dynamic_size=False)
    lut_array = lut_array.write(0, basic_chirp)  # Write the basic chirp

    for i in tf.range(1, M):
        rolled_chirp = tf.roll(basic_chirp, -i, axis=0)
        lut_array = lut_array.write(i, rolled_chirp)

    return lut_array.stack()


import tensorflow as tf

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

    # Generate random symbols and apply mask
    rand_symbols = tf.random.uniform(
        [batch_size, 2 * max_interferers], minval=0, maxval=M, dtype=tf.int32
    )

    # Sequence mask creates an array of True and False based on how many interferers were drawn
    mask = tf.sequence_mask(2 * n_interferers, 2 * max_interferers, dtype=tf.bool)

    # Masking: Set symbols to M (indicating zero power in LUT) for unused positions
    masked_symbols = tf.where(mask, rand_symbols, M)

    # Gather symbols from the Look-Up Table (LUT)
    inter_symbols = tf.gather(upchirp_lut, masked_symbols, axis=0)

    # Generate random arrival times (shifts) for each batch
    rand_arrival = tf.random.uniform(
        [batch_size, 2 * max_interferers], minval=1, maxval=M - 1, dtype=tf.int32
    )

    # Initialize TensorArray to store shifted symbols
    shifted_inter = tf.TensorArray(dtype=tf.complex64, size=batch_size)

    # Shift each symbol in the batch
    for b in tf.range(batch_size):
        batch_symbols = tf.TensorArray(dtype=tf.complex64, size=2 * max_interferers)
        for i in tf.range(2 * max_interferers):
            # Shift each symbol by its corresponding random arrival time
            shifted_symbol = tf.roll(inter_symbols[b, i, :], shift=-rand_arrival[b, i], axis=0)
            batch_symbols = batch_symbols.write(i, shifted_symbol)
        shifted_inter = shifted_inter.write(b, batch_symbols.stack())

    # Stack the shifted symbols to form a tensor
    shifted_inter = shifted_inter.stack()

    # A random SIR value between min and max is sampled uniformly
    SIR_dB = tf.random.uniform([batch_size, max_interferers], SIR_min_dB, SIR_max_dB)
    SIR_lin = tf.pow(10.0, SIR_dB / 10.0)

    # Calculate interferer amplitudes based on SIR
    interferer_amp = tf.cast(user_amp, tf.complex64) / tf.sqrt(tf.cast(SIR_lin, tf.complex64))

    # Initialize the output tensor
    half_shifted_inter = tf.zeros([batch_size, M], dtype=tf.complex64)

    # Scale and combine the interferer symbols
    for i in tf.range(max_interferers):
        half_shifted_inter += (
            tf.expand_dims(interferer_amp[:, i], axis=-1) * shifted_inter[:, i, :]
        )

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
    noise_stddev = tf.cast(tf.sqrt(noise_power / 2.0), dtype=tf.float32)

    # Generate complex AWGN channel
    noise_real = tf.random.normal(
        shape=tf.shape(user_chirp_tx),
        mean=0.0,
        stddev=noise_stddev,
        dtype=tf.float32,
    )

    noise_imag = tf.random.normal(
        shape=tf.shape(user_chirp_tx),
        mean=0.0,
        stddev=noise_stddev,
        dtype=tf.float32,
    )

    complex_noise = tf.complex(noise_real, noise_imag)

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
