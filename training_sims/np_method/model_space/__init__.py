import numpy as np


#Currently defined as a function, though not nessessary,
#however it might easy swapping it out for a more complex model in the future
def detect(fft_input):
    output = np.argmax(fft_input, axis=1)
    return output
