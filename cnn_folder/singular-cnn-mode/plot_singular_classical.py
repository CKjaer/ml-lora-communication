import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys

SNR = [-6, -8, -10, -12, -14, -16]
SER_classical = [0.000005, 0.001608, 0.038067, 0.203010, 0.468787, 0.698075]
SER_singular = [0.042525, 0.063476, 0.154187, 0.368527, 0.607143, 0.784152]
SER_CNN_FSD = []
# plot classical in black with triangle markers and singular in blue with square markers
plt.semilogy(SNR, SER_classical, 'k-', marker='^', label='Classical $\lambda=0.00$')
plt.semilogy(SNR, SER_singular, 'b-', marker='s', label='Singular $\lambda=0.00$')
#legend down left, write classical lambda=0 and Singular CNN lambda=0
plt.legend(loc='lower left')




plt.xlabel('SNR [dB]')
plt.ylabel('Symbol Error Rate [SER]')
plt.ylim(1e-5, 1e0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))


plt.savefig('singular_cnn_ser_vs_snr.png')
plt.show()