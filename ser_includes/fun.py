import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.random as tfr

import matplotlib.pyplot as plt

k = 1.38e-23
B = int(250e3)
T = 298.16
PN0 = k * T * B
SNR = np.linspace(-16, -4, 7)
print(SNR)

Pj = (10**(SNR/10) * PN0)
hj = np.sqrt(Pj)
print(Pj)
print(hj)

path_loss_exponent = -3.5/2

ray = tfr.rayleigh((B,), scale=1, dtype=tf.float32)
E_x = np.mean(ray)
print(f"r mean: {np.mean(ray)}, r std: {np.std(ray)}, r var: {np.var(ray)}, r max: {np.max(ray)}, r min: {np.min(ray)}")

d = np.power(hj/E_x, -1/path_loss_exponent)
print(d)


d = np.arange(200,1000)
hi = np.power(d, path_loss_exponent) * E_x
print(20*np.log10(hi))

plt.plot(d, hi)
for i in range(len(hj)):
    plt.axhline(hj[i], color='r')
plt.show()