import tensorflow as tf
import matplotlib.pyplot as plt

def is_local_maximum(fft_vec):
    fft_prev = tf.roll(fft_vec, shift=1, axis=1)  # Circular shift to the right (previous)
    fft_next = tf.roll(fft_vec, shift=-1, axis=1)  # Circular shift to the left (next)

    # Find where the current element is larger than both the previous and next elements
    larger_than_prev = fft_vec > fft_prev
    larger_than_next = fft_vec > fft_next

    # Logical AND to find the local maxima
    is_maximum = tf.logical_and(larger_than_prev, larger_than_next)

    # Convert boolean tensor to float (1.0 for maxima, 0.0 for non-maxima)
    return tf.cast(is_maximum, dtype=tf.float32)

#Currently defined as a function, though not nessessary,
#however it might easy swapping it out for a more complex model in the future
#@tf.function
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
    snr_gained = snr + processing_gain + 20
    snr_linear = tf.pow(tf.cast(10.0,dtype=tf.float64), snr_gained / 10.0)
    s = tf.sqrt(snr_linear * noise_power)

    s = tf.constant(2.075e-06,dtype=tf.float64)
    s_arr = tf.fill(fft_input.shape, s)
    s_arr = tf.cast(s_arr, dtype = fft_input.dtype)
    #Only grab maximas
    max_arr = is_local_maximum(fft_input)
    fft_masked = fft_input * max_arr
    fft_diff = tf.pow(fft_masked  - s_arr,2)

    if False:
        x_dat = tf.linspace(0,M-1,M)
        m = tf.argmin(fft_diff[0])
        plt.subplots(1,4,figsize =(5,25))
        plt.subplot(1,4,1)
        plt.plot(x_dat,fft_input[0].numpy(),'-o')
        plt.plot(x_dat,s_arr[0].numpy(),'--')
        plt.axvline(m,c='r',linestyle = '--')
        plt.legend(["fft","s_arr","desc"])
        plt.title("fft orig")

        plt.subplot(1,4,2)
        s = tf.cast(s, tf.float32)
        plt.stem(x_dat,max_arr[0]*s)
        plt.plot(x_dat,fft_input[0].numpy(),'--o',c='g')
        plt.title("mask'")
        plt.axvline(m, c='r', linestyle = '--')
        plt.legend(["mask","fft","decision"])
        
        plt.subplot(1,4,3)
        plt.plot(x_dat,fft_masked[0],'-o')
        plt.axvline(m,c='r',linestyle = '--')
        plt.title("Post mask")
        plt.legend(["fft post mask", "desicion"])

        plt.subplot(1,4,4)
        plt.plot(x_dat,fft_diff[0],'-o')
        plt.axvline(m,c='r',linestyle='--')
        plt.title("Error squared")
        plt.legend(["error", "desicion"])
        plt.show()
    
    output = tf.argmin(fft_diff, axis=1, output_type=tf.int32)

    if False:
        print(f"fft_input shape: {fft_input.shape}")
        print(f"is_maximum shape: {max_arr.shape}")
        print(f"fft_masked shape: {fft_masked.shape}")
        print(f"fft_diff shape: {fft_diff.shape}")
        print(f"output shape: {output.shape}")

    return output
