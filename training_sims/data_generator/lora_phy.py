import tensorflow as tf
from math import pi

# Create chirp
@tf.function
def create_basechirp(M,device):
    """Create the basic chirp
     
    Formula based on "Efficient Design of Chirp Spread Spectrum Modulation for Low-Power Wide-Area Networks" by Nguyen et al.
    The base chirp is calculated as exp(2*pi*1j*((n^2/(2*M)) - (n/2))).

    Returns:
        tf.complex64: A [M] tensor of complex64 values representing the basic chirp
    """
    with device:
        n = tf.linspace(0, M-1, M)
        n = tf.cast(n, tf.complex64)
        M_complex = tf.cast(M, tf.complex64)

        #We've split the calculation into two parts - mult1 handling 2*pi*1j and mult2 handling the rest of the formula
        mult1 = tf.constant(2*pi*1j, tf.complex64)

        # Mult 2 is calculated as (n^2/(2*M)) - (n/2)
        frac1 = tf.divide(tf.pow(n,2), (2*M_complex))
        frac2 = tf.divide(n,2)
        mult2 = tf.subtract(frac1,frac2)
        return tf.exp(tf.multiply(mult1, mult2))
    

# Generate a LUT for the upchirps
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
        
        for i in tf.range(1,M):
            rolled_chirp = tf.roll(basic_chirp, -i, axis=0)
            lut_array = lut_array.write(i, rolled_chirp)
        return lut_array.stack()


# Symmetrical circular AWGN channel
@tf.function
def channel_model(SNR, signal_length, M, device):
    """Creates a channel model, in this case AWGN.
    Returns:
        tf.complex64: A [signal_length, M] tensor of complex64 values representing the noise
    """
    with device:
        SNR_complex = tf.cast(SNR, tf.complex64)
        noise_power = 1/tf.pow(tf.constant(10.0,tf.complex64),SNR_complex/10.0)

        tfr1 = tf.random.normal((signal_length, M),dtype=tf.float32)
        tfr2 = tf.random.normal((signal_length, M),dtype=tf.float32)
        noise_complex = tf.complex(tfr1,tfr2)
        noise = noise_complex*tf.sqrt(noise_power/2.0)
        return noise

#@tf.function
#def generate_intererer_table(lut, M, N, device):
#    """Generates the interference table for the LoRa PHY.
#    Returns:
#        tf.complex64: A [M, M] tensor of complex64 values representing the interference table
#    """
#    with device:
#        # Create the basic chirp
#        int_vec = tf.random.uniform((2,N), minval=0, maxval=M, dtype=tf.int32)
#        int_vec = tf.squeeze(int_vec)
#        symbol1 = tf.gather(lut, int_vec, axis=1)
#
        #return interference_table
