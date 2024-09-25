import tensorflow as tf


#Currently defined as a function, though not nessessary,
#however it might easy swapping it out for a more complex model in the future
@tf.function
def detect(fft_input, device):
    """Detects the symbol from the FFT output.

    Args:
        fft_input (tf.complex64): The FFT output
        device (tf.device): The currently running device

    Returns:
        tf.int32: Returns the most likely symbol
    """
    with device:
        output = tf.argmax(fft_input, axis=1)
        output = tf.cast(output, tf.int32)
        return output