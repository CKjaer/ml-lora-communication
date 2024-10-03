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
    mask = tf.sequence_mask(batch_size, 2 * max_interferers, dtype=tf.bool)
    masked_symbols = tf.where(mask, rand_symbols, 128)

    # Gather symbols from LUT and shift with random arrival times
    inter_symbols = tf.gather(upchirp_lut, masked_symbols, axis=0)
    rand_arrival = tf.random.uniform([batch_size], minval=0, maxval=M, dtype=tf.int32)
    shifted_inter = tf.roll(
        inter_symbols,
        shift=-rand_arrival,
        axis=2 * tf.ones([batch_size], dtype=tf.int32),
    )

    # half_shifted_inter = shifted_inter[:, : shifted_inter.shape[1] // 2]
    half_shifted_inter = shifted_inter[:, : tf.shape(shifted_inter)[1] // 2]

    # Generate radii for interferers (distance in the annulus)
    radii = tf.sqrt(
        tf.random.uniform(
            tf.shape(half_shifted_inter), minval=200, maxval=1000, dtype=tf.float32
        )
    )
    # radii_repeated = tf.repeat(radii, repeats=M, axis=1)

    # Print the data to a text file
    # with open("interferer_symbols.txt", "w") as f:
    #     # Convert tensors to numpy and write them to the file
    #     f.write("Masked Symbols:\n")
    #     f.write(str(masked_symbols.numpy()) + "\n\n")
    #     f.write("Inter Symbols:\n")
    #     f.write(str(inter_symbols.numpy()) + "\n\n")
    #     f.write("Shifted Inter:\n")
    #     f.write(str(shifted_inter.numpy()) + "\n\n")
    #     f.write("X rolled\n")
    #     f.write(str(half_shifted_inter.numpy()) + "\n\n")
    #     f.write("Radii:\n")
    #     f.write(str(tf.cast(tf.sqrt(radii), dtype=tf.complex64).numpy()) + "\n")

    return half_shifted_inter, tf.cast(radii, dtype=tf.complex64)
