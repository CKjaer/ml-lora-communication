import numpy as np


def n_symbol_estimation(max_ser, relative_error):
    n_symbols = int(np.ceil((1 - max_ser) / (relative_error * max_ser)))
    return n_symbols


# We disregard -4 dB
snr_vals = np.array([-16, -14, -12, -10, -8, -6])
ser_vals = np.array([1e-1, 1e-1, 1e-1, 1e-2, 1e-3, 1e-4])

total_symbols = 0
for i in range(len(snr_vals)):
    n_symbols = n_symbol_estimation(ser_vals[i], 0.01)
    total_symbols += n_symbols
    print(f"SNR: {snr_vals[i]}, n_symbols: {n_symbols}")

print(
    f"Total symbols: {total_symbols} Total time: {(total_symbols * 6e-3) / 3600:.2f} hours"
)
