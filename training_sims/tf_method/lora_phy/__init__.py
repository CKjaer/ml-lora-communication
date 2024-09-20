import tensorflow as tf
from math import pi
import numpy as np

# Generate a LUT for the upchirps
def upchirp_lut(M, basic_chirp, device):
    with device:
        lut_list = [basic_chirp]  # Initialize with the basic chirp
        for i in range(1,M):
            rolled_chirp = tf.roll(basic_chirp, -i, axis=0)
            lut_list.append(rolled_chirp)
        upchirp_lut = tf.stack(lut_list, axis=1)
        return upchirp_lut

# Create chirp
def create_basechirp(M,device):
    with device:
        n = tf.linspace(0, M-1, M)
        mult1 = 2*pi*1j
        mult2 = tf.cast(((n**2 / (2 * M)) - (n / 2)), tf.complex64)
        basis_chirp = tf.exp(tf.multiply(mult1, mult2))
        return basis_chirp

# Symmetrical circular AWGN channel
def channel_model(SNR, signal_length, M, device):
  with device:
    tfr1 = tf.random.normal((signal_length, M))
    tfr2 = tf.random.normal((signal_length, M))
    tfr1 = tf.complex(tfr1,tfr2)
    noise_power = tf.divide(1,tf.pow(10,tf.divide(SNR,10)))
    noise_power = tf.cast(noise_power, tf.complex64)
    noise = tf.sqrt(noise_power/2)
    noise = noise * tfr1
    return noise 