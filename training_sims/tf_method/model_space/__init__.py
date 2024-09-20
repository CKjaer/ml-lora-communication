import tensorflow as tf


#Currently defined as a function, though not nessessary,
#however it might easy swapping it out for a more complex model in the future
def detect(fft_input, device):
    with device:
        output = tf.argmax(fft_input, axis=1)
        output = tf.cast(output, tf.int32)
        return output
