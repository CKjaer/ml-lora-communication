# Setup imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.random as tfr
import matplotlib.pyplot as plt


# Create a function to calculate the dB value of a number, and the number from a dB value
def calcTodB(x):
    return 10 * np.log10(x)


def calcFromdB(x):
    return np.power(10, x / 10)


# Calculate noise setup
k = 1.38 * np.power(10.0, -23)  # [J/K]
B = int(250e3)  # [Hz]
T = 298.16  # [K]
PN0 = k * T * B  # [J*Hz] = [W]
# PN0 = -149
# PN0 = calcFromdB(PN0) # [W]
print(f"Noise power: {PN0}, dB: {calcTodB(PN0)}")


def getRayleighSamples(n: int, v: bool = False):
    real = tf.random.normal([n], mean=0.0, stddev=1.0, dtype=tf.float32)
    imag = tf.random.normal([n], mean=0.0, stddev=1.0, dtype=tf.float32)
    ray = tf.abs(
        (1 / tf.sqrt(tf.constant(2, dtype=tf.complex64))) * tf.complex(real, imag)
    )
    return ray


def getPSfromSNR(SNR, Pt, v: bool = False):
    # To estimate the distance accurately, we pull alot of Rayleig samples and average them
    SNR_lin = calcFromdB(SNR)
    hj = np.sqrt((SNR_lin * PN0) / (Pt))
    Ps = np.power(hj, 2.0) * Pt
    if v:
        print(f"Length of Ps: {len(Ps)}")
        print(f"Ps spans {np.min(Ps)} and {np.max(Ps)}")
    return Ps


def calcDfromPs(Ps, eta, tp, v: bool = False):
    d = np.power(Ps / (tp), -1 / eta)
    if v:
        print(f"Length of d: {len(d)}")
        print(f"D spans {np.min(d)} and {np.max(d)}")
    return d


def calcPSfromD(d, eta, ray_scale: bool, tp, v: bool = False):
    ray = getRayleighSamples(len(d), v=v)
    dist_loss = np.power(d, -eta)
    Ps = dist_loss * tp
    if ray_scale:
        Ps = Ps  # * ray**2
    else:
        Ps = Ps  # * np.mean(ray**2)
    print(f"mean ray: {np.mean(ray**2)}")
    if v:
        print(f"Length of Ps: {len(Ps)}")
        print(f"Ps spans {np.min(Ps)} and {np.max(Ps)}")
    return Ps


# Get SNRS in lin scale
SNR = np.linspace(-4, -16, 7)  # [dB]
# Calculate Transmit Power - According to PhD then 14 dBm is the transmit power
tp = calcFromdB(-62)  # [W]
eta = 3.5
Ps = getPSfromSNR(SNR, tp, v=True)
d = calcDfromPs(Ps, eta, tp, v=True)
for i, snr in enumerate(SNR):
    print(
        f"SNR: {snr}, Ps: {calcTodB(Ps[i])}, SNR: {calcTodB(Ps[i]/PN0):.2f} distance: {d[i]}"
    )

ray_test = getRayleighSamples(int(1e5), v=True)
print(np.mean(ray_test**2))


def plot(d_in, Ps_in, eta, transmit_power, SNR, rmin, rmax, ray_scale: bool = False):
    distance = np.linspace(rmin, rmax, 1000)
    Ps = calcPSfromD(distance, eta, ray_scale, transmit_power)
    plt.plot(distance, calcTodB(Ps))
    plt.scatter(d_in, calcTodB(Ps_in))
    for i in range(len(SNR)):
        plt.axhline(calcTodB(Ps_in[i]), color="r")
    print(
        f"PU min: {np.min(calcTodB(Ps_in))}, max: {np.max(calcTodB(Ps_in))}, mean: {np.mean(calcTodB(Ps_in))}"
    )
    print(
        f"PI min: {np.min(calcTodB(Ps))}, max: {np.max(calcTodB(Ps))}, mean: {np.mean(calcTodB(Ps))}"
    )

    plt.legend(["Calculated distance @ SNRs", "Path loss", "Power at SNR"])
    plt.show()


tp = calcFromdB(-62)  # [W]
d = calcDfromPs(Ps, eta, tp, v=True)
plot(d, Ps, eta, tp, SNR, 200, 1000, False)


N = int(2e3)
R = np.sqrt(np.random.uniform(size=N)) * 800 + 200
theta = np.random.uniform(size=N) * 2 * np.pi
x = R * np.cos(theta)
y = R * np.sin(theta)
d_i = np.sqrt(x**2 + y**2)
Pi = calcPSfromD(d_i, eta, True, tp)
plt.scatter(x + 1000, y + 1000, c=calcTodB(Pi))
for i in range(len(SNR)):
    plt.gca().add_patch(plt.Circle((1000, 1000), d[i], edgecolor="r", facecolor="none"))
plt.gca().add_patch(
    plt.Circle((1000, 1000), calcTodB(PN0), edgecolor="pink", facecolor="none")
)
plt.colorbar()
plt.show()
