import numpy as np
from numba import njit

# Generate a LUT for the upchirps
@njit(parallel=True)
def upchirp_lut(M, basic_chirp):
    upchirp_lut = np.zeros((M, M), dtype=np.complex128)
    for i in range(M):
        upchirp_lut[i, :] = np.roll(basic_chirp, -i)
    return upchirp_lut

# Create chirp
@njit(parallel=True)
def create_basechirp(M):
    n = np.linspace(0, M-1, M)
    basis_chirp = np.exp(2 * 1j * np.pi * ((n**2 / (2 * M)) - (n / 2)))
    return basis_chirp

# Symmetrical circular AWGN channel
@njit(parallel=True)
def gaussian_channel(SNR, signal_length, M):
    noise_power = 1 / (10**(SNR/10))
    noise =  np.sqrt(noise_power/2) * (np.random.randn(signal_length, M) + 1j * np.random.randn(signal_length, M))
    return noise
