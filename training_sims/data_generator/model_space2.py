import tensorflow as tf
import matplotlib.pyplot as plt


#Currently defined as a function, though not nessessary,
#however it might easy swapping it out for a more complex model in the future
@tf.function
def detect(fft_input, snr, M, noise_power):
    """Detects the symbol from the FFT output.

    Args:
        fft_input (tf.complex64): The FFT output
        device (tf.device): The currently running device

    Returns:
        tf.int32: Returns the most likely symbol
    """
    SF = 7
    processing_gain =  10 * tf.math.log(tf.cast(SF, dtype=tf.float64)) / tf.cast(tf.math.log(10.0), dtype=tf.float64)
    #print(f"pg: {processing_gain}")
    snr_gained = snr + processing_gain + 20
    snr_linear = tf.pow(tf.cast(10.0,dtype=tf.float64), snr_gained / 10.0)
    s = tf.sqrt(snr_linear * noise_power)
    s = tf.constant(2.075e-06,dtype=tf.float64)
    #print(f"SNR: {snr}, snr_lin: {snr_linear}, np: {noise_power}")
    #print(f"Value of s: {s}")
    s_arr = tf.fill(fft_input.shape, s)
    s_arr = tf.cast(s_arr, dtype = fft_input.dtype)
    x_dat = tf.linspace(0,M-1,M)
    fft_diff = tf.pow(fft_input  - s_arr,2)

    #plt.subplots(1,2,figsize =(5,10))
    #plt.subplot(1,2,1)
    #plt.plot(x_dat,fft_input[0,:].numpy(),'-o')
    #plt.plot(x_dat,s_arr[0,:].numpy(),'--')
    #plt.legend(["fft","s_arr"])
    #plt.title("fft orig")
    
    #plt.subplot(1,2,2)
    #plt.plot(x_dat,fft_diff[0,:].numpy(),'-o')
    #plt.title("fft min")
    #plt.show()
    

    output = tf.argmin(fft_diff, axis=1, output_type=tf.int32)
    return output
