import numpy as np
import matplotlib.pyplot as plt
import timeit

# LoRa PHY parameters (based on EU863-870 DR0 channel)
SF = 7 # Spreading factor 
BW = 250e3 # Bandwidth [Hz]
M = int(2**SF) # Number of symbols per chirp 
samp_rate = BW # [Hz]
symbol_duration = M / BW # [s]
t = np.linspace(0, symbol_duration, int(samp_rate), endpoint=False)

print(f"Basic chirp specs: SF={SF}, bandwidth={BW} Hz, M={M}")

# Generate a LUT for the upchirps 
def upchirp_lut(M):
    n = np.linspace(0, M-1, M)
    basic_chirp = np.exp(2 * 1j * np.pi * ((n**2 / (2 * M)) - (n / 2)))
    upchirp_lut = np.zeros((M, M), dtype=complex)
    for i in range(M):
        upchirp_lut[i, :] = np.roll(basic_chirp, -i)
    return upchirp_lut

# Symmetrical circular AWGN channel
def gaussian_channel(SNR, signal_length, M):
    noise_power = 1 / (10**(SNR/10))
    noise = noise_power * np.sqrt(1/2) * (np.random.randn(signal_length, M) + 1j * np.random.randn(signal_length, M))
    return noise

# Downchirp received signal
def downchirp(upchirp, base_chirp):
    return upchirp * np.conj(base_chirp)


if __name__ == '__main__':
    
    upchirp = upchirp_lut(M)
    msg_size = int(1e7) # Number of symbols
    
    # Generate random symbols
    snr_vals = np.linspace(-16, 4, 2) # SNR values to test
    n_symbols = np.zeros(len(snr_vals))
    for i in range(len(snr_vals)):

        msg_tx = np.random.randint(0, M, msg_size)
        upchirp_tx = upchirp[msg_tx, :] 
        awgn = gaussian_channel(snr_vals[i], msg_size, M) 
        dechirp_rx = downchirp(upchirp_tx, upchirp[msg_tx, :]) + awgn
        fft_result = np.abs(np.fft.fft(dechirp_rx))
        msg_rx = np.argmax(fft_result, axis=1) 
        n_symbols[i] = np.mean(msg_tx != msg_rx)


                          
    plt.plot(snr_vals, n_symbols)
    plt.xlabel('SNR [dB]')
    plt.ylabel('SER')
    plt.show()
   
   