import tensorflow as tf
from math import pi


@tf.function
def create_basechirp(M, device):
    """Create the basic LoRa chirp

    Formula based on "Efficient Design of Chirp Spread Spectrum Modulation for Low-Power Wide-Area Networks" by Nguyen et al.
    The base chirp is calculated as exp(2*pi*1j*((n^2/(2*M)) - (n/2))).

    Returns:
        tf.complex64: A [M] tensor of complex64 values representing the basic chirp
    """
    with device:
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
def upchirp_lut(M, basic_chirp, device):
    """Create a lookup table based on the basic chirp.
    Returns:
        tf.complex64: A [M, M] tensor of complex64 values representing the upchirps
    """
    with device:
        # Sets up a tensor array with datatype of the basic chirp and size of M
        lut_array = tf.TensorArray(dtype=basic_chirp.dtype, size=M, dynamic_size=False)
        lut_array = lut_array.write(0, basic_chirp)  # Write the basic chirp

        for i in tf.range(1, M):
            rolled_chirp = tf.roll(basic_chirp, -i, axis=0)
            lut_array = lut_array.write(i, rolled_chirp)

        return lut_array.stack()


@tf.function
def channel_model(SNR, signal_length, M, device):
    """A circular AWGN channel as a function of SNR.
    Returns:
        tf.complex64: A [signal_length, M] tensor of complex64 values representing the noise
    """
    with device:
        SNR_complex = tf.cast(SNR, tf.complex64)
        noise_power = 1 / tf.pow(tf.constant(10.0, tf.complex64), SNR_complex / 10.0)

        tfr1 = tf.random.normal((signal_length, M), dtype=tf.float32)
        tfr2 = tf.random.normal((signal_length, M), dtype=tf.float32)
        noise_complex = tf.complex(tfr1, tfr2)
        noise = noise_complex * tf.sqrt(noise_power / 2.0)
        return noise


@tf.function
def generate_interferer_symbols(batch_size, rate_param, M, upchirp_lut):
    """
    Generate symbols for interferers based on a Poisson distribution.

    This function draws a number of interferers from a Poisson distribution, generates random symbols for
    each interferer, and shifts them with a random arrival time.

    Args:
        batch_size (int): The number of batches to process.
        rate_param (float): The rate parameter for the Poisson distribution.
        M (int): The maximum value for random symbols and a reference for masking.
        upchirp (tf.Tensor): A tensor representing the look-up table (LUT) for symbols.

    Returns:
        tf.complex64: A tensor containing the shifted complex symbols for each batch.
        tf.complex64: A tensor containing the radii for each interferer casted to complex64.
    """

    # Draw the number of interferers from a Poisson distribution
    n_interferers = tf.random.poisson((batch_size,), rate_param, dtype=tf.int32)

    # Create 2 random symbols for each interferer
    max_interferers = tf.reduce_max(n_interferers)
    rand_inter_symbols = tf.random.uniform(
        (batch_size, 2 * max_interferers), minval=0, maxval=M, dtype=tf.int32
    )

    # Create a mask to zero out the symbols that are not used if the number of interferers is less than the max
    mask = tf.sequence_mask(n_interferers, 2 * max_interferers, dtype=tf.int32)

    # Fill the masked symbols with 128 pointing to a zero symbol in the LUT
    masked_symbols = tf.where(
        (rand_inter_symbols * mask) == 0, M, (rand_inter_symbols * mask)
    )

    # Gather the symbols from the LUT and reshape to stack symbols for each interferer column-wise
    inter_symbols_lut = tf.gather(upchirp_lut, masked_symbols, axis=0)
    shaped_symbols = tf.reshape(
        inter_symbols_lut, (batch_size, 2 * max_interferers * M)
    )

    # Shift the symbols with a random arrival time
    rand_arrival = tf.random.uniform((batch_size,), minval=0, maxval=M, dtype=tf.int32)
    shifted_inter = tf.roll(
        shaped_symbols,
        shift=-rand_arrival,
        axis=tf.ones((batch_size,), tf.int32),
    )

    start_indices = tf.range(0, M * max_interferers, M * 2)
    tiled_indices = tf.tile(
        tf.expand_dims(tf.range(M), 0), [tf.shape(start_indices)[0], 1]
    )
    indices = tf.reshape(tiled_indices + tf.expand_dims(start_indices, 1), [-1])

    sliced_inter = tf.squeeze(tf.gather(shifted_inter, [indices], axis=1))

    # Map the interfering users to a distance (annulus between 200 and 1000 m)
    random_radii = tf.sqrt(
        tf.random.uniform(
            (n_interferers.shape[0], max_interferers),
            minval=200,
            maxval=1000,
            dtype=tf.float32,
        )
    )
    # Repeat the radii for symbols for element-wise multiplication
    radii_repeated = tf.repeat(random_radii, repeats=M, axis=1)
    print(radii_repeated.shape)
    return sliced_inter, tf.cast(radii_repeated, dtype=tf.complex64), max_interferers
