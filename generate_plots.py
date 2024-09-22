import numpy as np
import matplotlib.pyplot as plt
from load_files import load_data

data = load_data("test")
def generate_plots(data):
    for i in range(len(data)):
        freqs = list(map(float, data["freqs"].iloc[i].split(';'))) # split string and convert to float
        freqs_idx = np.arange(0, len(freqs), 1)
        print(data["symbol"].iloc[i])

        fig = plt.figure(figsize=(2,2))
        plt.axis('off')
        fig.set_facecolor('black')
        plt.plot(freqs_idx, freqs, color = 'white', linewidth=1)
        plt.close(fig)
        fig.savefig(f"data/snr_{data['snr'].iloc[i]}_symbol_{data['symbol'].iloc[i]}.png", bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    
    # # create fft plot
    # plt.figure()
    # plt.plot(freqs_idx, freqs)
    # plt.title("FFT")
    # plt.xlabel("Frequency")
    # plt.ylabel("Magnitude")
    # plt.show()
    pass
        
    
    