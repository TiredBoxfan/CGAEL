import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="CGAEL", name="ArgmaxLayer")
class ArgmaxLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ArgmaxLayer, self).__init__()

    def call(self, data, axis=-1):
        return tf.math.argmax(data, axis=axis)