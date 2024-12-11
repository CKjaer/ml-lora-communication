import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys

SNR = [-6, -8, -10, -12, -14, -16]

SER_classical_0 = [0.000006, 0.001611, 0.037937, 0.203298, 0.468955, 0.698191]
SER_classical_025=[0.017987, 0.030889, 0.082529, 0.251476, 0.504485, 0.72121]
SER_classical_05=[0.0346, 0.058052, 0.12541, 0.292114, 0.541924, 0.741722]
SER_classical_07=[0.049354, 0.083618, 0.153726, 0.243079, 0.487828, 0.703728]
SER_classical_1=[0.070218, 0.110132, 0.198948,0.381264,0.604959,0.776125 ]

SER_singular_0 = [0.014213, 0.021575, 0.090745, 0.314732, 0.553571, 0.775670]
SER_singular_025 = [0.027453, 0.042788, 0.121194, 0.343750, 0.591518, 0.750000]
SER_singular_05 =[0.043139, 0.064844, 0.162861, 0.366071, 0.592634, 0.794643]
SER_singular_07 =[0.054546, 0.081450, 0.182893, 0.373884, 0.623884, 0.777902]
SER_singular_1 = [0.073288, 0.106741, 0.213341, 0.444196, 0.674107, 0.822545]

SER_CNN_FSD_0 = [0.00001, 0.0021, 0.0447, 0.2399, 0.5179, 0.7824]
SER_CNNN_FSD_025 = [0.0073, 0.0154, 0.07, 0.2656, 0.5335, 0.8002]
SER_CNNN_FSD_05 = [0.014, 0.0277, 0.0897, 0.3147, 0.597,0.827]
SER_CNNN_FSD_07 = [0.0224, 0.0387, 0.1089, 0.2991, 0.5725, 0.8058]
SER_CNNN_FSD_1 = [0.0251, 0.0525, 0.1342, 0.3292, 0.6272, 0.8136]

IQ_CNN_0 = [0.000625, 0.0121875, 0.08640625, 0.2803125, 0.5503125, 0.775]
IQ_CNN_025 = [0.023125, 0.04640625,0.14546875,  0.359375, 0.62703125, 0.82078125]
IQ_CNN_05 = [0.04609375, 0.08828125,0.2015625,0.435625,0.67828125,0.84484375, ]
IQ_CNN_07 = [0.0546875, 0.1090625, 0.2465625, 0.47421875,0.71109375, 0.86875]
IQ_CNN_1 = [0.0815625, 0.145, 0.29578125, 0.52453125, 0.7490625, 0.88]

#FIRST PLOT
plt.semilogy(SNR, SER_classical_0, 'k-', linestyle='--', marker='^', label='Classical $\lambda=0.00$')
plt.semilogy(SNR, SER_classical_025, 'k-', marker='^', label='Classical $\lambda=0.25$')

#SER_CNN_FSD should in uncontinous line
plt.semilogy(SNR, SER_CNN_FSD_0, 'b-', linestyle='--', marker='v', label='CNN FSD $\lambda=0.00$')
plt.semilogy(SNR, SER_CNNN_FSD_025, 'b-', marker='v', label='CNN FSD $\lambda=0.25$')

plt.semilogy(SNR, IQ_CNN_0, 'r-',linestyle='--', marker='o', label='CNN TSD $\lambda=0.00$')
plt.semilogy(SNR, IQ_CNN_025, 'r-', marker='o', label='CNN TSD $\lambda=0.25$')

plt.semilogy(SNR, SER_singular_0, 'g-', linestyle='--',marker='s', label='Singular CNN $\lambda=0.00$')
plt.semilogy(SNR, SER_singular_025, 'g-',marker='s', label='Singular CNN $\lambda=0.25$')

#legend down left, write classical lambda=0 and Singular CNN lambda=0
plt.legend(loc='lower left')


plt.xlabel('SNR [dB]')
plt.ylabel('SER')
plt.ylim(1e-5, 1e0)
plt.xlim(-16, -6)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))

plt.show()

#SECONND PLOT
# plot classical in black with triangle markers and singular in blue with square markers
plt.semilogy(SNR, SER_classical_0, 'k-',linestyle='--', marker='^', label='Classical $\lambda=0.00$')
plt.semilogy(SNR, SER_classical_05, 'k-', marker='^', label='Classical $\lambda=0.5$')

#SER_CNN_FSD should in uncontinous line
plt.semilogy(SNR, SER_CNN_FSD_0, 'b-', linestyle='--', marker='v', label='CNN FSD $\lambda=0.00$')
plt.semilogy(SNR, SER_CNNN_FSD_05, 'b-', marker='v', label='CNN FSD $\lambda=0.5$')

plt.semilogy(SNR, IQ_CNN_0, 'r-',linestyle='--', marker='o', label='CNN TSD $\lambda=0.00$')
plt.semilogy(SNR, IQ_CNN_05, 'r-', marker='o', label='CNN TSD $\lambda=0.5$')

plt.semilogy(SNR, SER_singular_0, 'g-', linestyle='--',marker='s', label='Singular CNN $\lambda=0.00$')
plt.semilogy(SNR, SER_singular_05, 'g-',marker='s', label='Singular CNN $\lambda=0.5$')

#legend down left, write classical lambda=0 and Singular CNN lambda=0
plt.legend(loc='lower left')


plt.xlabel('SNR [dB]')
plt.ylabel('SER')
plt.ylim(1e-5, 1e0)
plt.xlim(-16, -6)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))

plt.show()

#THIRD PLOT
# plot classical in black with triangle markers and singular in blue with square markers
plt.semilogy(SNR, SER_classical_0, 'k-',linestyle='--', marker='^', label='Classical $\lambda=0.00$')
plt.semilogy(SNR, SER_classical_07, 'k-', marker='^', label='Classical $\lambda=0.7$')

#SER_CNN_FSD should in uncontinous line
plt.semilogy(SNR, SER_CNN_FSD_0, 'b-', linestyle='--', marker='v', label='CNN FSD $\lambda=0.00$')
plt.semilogy(SNR, SER_CNNN_FSD_07, 'b-', marker='v', label='CNN FSD $\lambda=0.7$')

plt.semilogy(SNR, IQ_CNN_0, 'r-',linestyle='--', marker='o', label='CNN TSD $\lambda=0.00$')
plt.semilogy(SNR, IQ_CNN_07, 'r-', marker='o', label='CNN TSD $\lambda=0.7$')

plt.semilogy(SNR, SER_singular_0, 'g-', linestyle='--',marker='s', label='Singular CNN $\lambda=0.00$')
plt.semilogy(SNR, SER_singular_07, 'g-',marker='s', label='Singular CNN $\lambda=0.7$')

#legend down left, write classical lambda=0 and Singular CNN lambda=0
plt.legend(loc='lower left')


plt.xlabel('SNR [dB]')
plt.ylabel('SER')
plt.ylim(1e-5, 1e0)
plt.xlim(-16, -6)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))

plt.show()

#FOURTH PLOT
# plot classical in black with triangle markers and singular in blue with square markers
plt.semilogy(SNR, SER_classical_0, 'k-',linestyle='--', marker='^', label='Classical $\lambda=0.00$')
plt.semilogy(SNR, SER_classical_1, 'k-', marker='^', label='Classical $\lambda=1.0$')

#SER_CNN_FSD should in uncontinous line
plt.semilogy(SNR, SER_CNN_FSD_0, 'b-', linestyle='--', marker='v', label='CNN FSD $\lambda=0.00$')
plt.semilogy(SNR, SER_CNNN_FSD_1, 'b-', marker='v', label='CNN FSD $\lambda=1.0$')

plt.semilogy(SNR, IQ_CNN_0, 'r-',linestyle='--', marker='o', label='CNN TSD $\lambda=0.00$')
plt.semilogy(SNR, IQ_CNN_1, 'r-', marker='o', label='CNN TSD $\lambda=1.0$')

plt.semilogy(SNR, SER_singular_0, 'g-', linestyle='--',marker='s', label='Singular CNN $\lambda=0.00$')
plt.semilogy(SNR, SER_singular_1, 'g-',marker='s', label='Singular CNN $\lambda=1.0$')

#legend down left, write classical lambda=0 and Singular CNN lambda=0
plt.legend(loc='lower left')


plt.xlabel('SNR [dB]')
plt.ylabel('SER')
plt.ylim(1e-5, 1e0)
plt.xlim(-16, -6)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))

plt.show()