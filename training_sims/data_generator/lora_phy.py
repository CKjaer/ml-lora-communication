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
    n = tf.linspace(0, M-1, M)
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
    #basic_chirp = tf.linspace(0, M-1, M)
    #basic_chirp = tf.cast(basic_chirp, tf.complex64)
    lut_array = tf.TensorArray(dtype=basic_chirp.dtype, size=M, dynamic_size=False)
    lut_array = lut_array.write(0, basic_chirp)  # Write the basic chirp

    for i in tf.range(1, M):
        rolled_chirp = tf.roll(basic_chirp, -i, axis=0)
        lut_array = lut_array.write(i, rolled_chirp)

    return lut_array.stack()


@tf.function
def channel_model(SNR, signal_length, M):
    """A circular AWGN channel as a function of SNR.
    Returns:
        tf.complex64: A [signal_length, M] tensor of complex64 values representing the noise
    """
    SNR_complex = tf.cast(SNR, tf.complex64)
    noise_power = 1 / tf.pow(tf.constant(10.0, tf.complex64), SNR_complex / 10.0)

    tfr1 = tf.random.normal((signal_length, M), dtype=tf.float32)
    tfr2 = tf.random.normal((signal_length, M), dtype=tf.float32)
    noise_complex = tf.complex(tfr1, tfr2)
    noise = noise_complex * tf.sqrt(noise_power / 2.0)
    return noise


@tf.function
def generate_interferer_symbols(batch_size, rate_param, M, upchirp_lut, user_amp):
    """
    Generate symbols for interferers based on a Poisson distribution.

    Args:
        batch_size (int): Number of batches to process.
        rate_param (float): Rate parameter for the Poisson distribution.
        M (int): Maximum value for random symbols and reference for LUT.
        upchirp_lut (tf.Tensor): LUT for symbols.

    Returns:
        tf.complex64: Tensor of shifted complex symbols for each batch.
        tf.complex64: Tensor of radii for each interferer.
    """
    # Draw interferers from Poisson distribution
    n_interferers = tf.random.poisson([batch_size], rate_param, dtype=tf.int32)
    max_interferers = tf.reduce_max(n_interferers)

    # Generate random symbols and apply mask
    rand_symbols = tf.random.uniform(
        [batch_size, 2 * max_interferers], minval=0, maxval=M, dtype=tf.int32
    )
    
    # Sequence mask creates an array of true and false depending on how many u
    mask = tf.sequence_mask(2 * n_interferers, 2 * max_interferers, dtype=tf.bool)

    #Note: Setting the symbols to M makes the LUT return all zeros
    masked_symbols = tf.where(mask, rand_symbols, M)

    # Gather symbols from LUT and shift with random arrival times
    inter_symbols = tf.gather(upchirp_lut, masked_symbols, axis=0)
    rand_arrival = tf.random.uniform([batch_size], minval=0, maxval=M, dtype=tf.int32)

    shifted_inter = tf.roll(
        inter_symbols,
        shift=-rand_arrival,
        axis= 2 * tf.ones([batch_size], dtype=tf.int32),
    )
    half_shifted_inter = shifted_inter[:, : tf.shape(shifted_inter)[1] // 2]

    # 0 or 1?
    SIR_min_dB = 1
    SIR_max_dB = 10
    delta_SIR = SIR_max_dB - SIR_min_dB
    SIR_dB = tf.random.uniform((batch_size,max_interferers),0,1)*delta_SIR + SIR_min_dB

    SIR_dB = tf.cast(SIR_dB,dtype=tf.complex64)
    SIR_lin = tf.pow(tf.cast(10.0,dtype=SIR_dB.dtype),SIR_dB/10.0)

    interferer_amp = tf.cast(user_amp, dtype=tf.complex64) / tf.sqrt(SIR_lin)
    interferer_amp = tf.repeat(interferer_amp, M)
    interferer_amp = tf.reshape(interferer_amp, (batch_size, max_interferers, M))
    interferer_amp = tf.cast(interferer_amp, dtype=tf.complex64)
    user_amp = tf.cast(user_amp, dtype=tf.complex64)

    inter_symbols_scaled = tf.reduce_sum(
        interferer_amp * half_shifted_inter, axis=1
    )
    return inter_symbols_scaled

@tf.function
def process_batch(upchirp_lut,  rate_params, snr, msg_tx, batch_size, M, noise_power):
    user_chirp_tx = tf.gather(upchirp_lut, msg_tx, axis=0)

    # Generate complex AWGN channel
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
    snr = tf.cast(snr,dtype=tf.float64)
    snr_linear = tf.pow(tf.cast(10.0,dtype=tf.float64), snr / 10.0)
    user_amp = tf.sqrt(snr_linear * noise_power)
        # Generate the interfering users symbols and their distances
    inter_symbols_scaled = generate_interferer_symbols(
        batch_size, rate_params, M, upchirp_lut, user_amp
    )
    #print(f"Inter_symbols_scaled: {inter_symbols_scaled}")

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
def log10(x):
    num = tf.math.log(x)
    denom = tf.math.log(tf.constant(10,dtype=num.dtype))
    return num / denom