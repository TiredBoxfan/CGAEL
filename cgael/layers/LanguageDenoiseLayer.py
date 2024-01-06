import tensorflow as tf

def denoise_language(data, do_columns=True):
    """
    Denoises language data when it is represented as a tensor of integers.
    
    Parameters
    ---
    do_columns : bool
        If false, only row-wise denoising will be done.
    """
    # Create a binary mask for the data where:
    # - 0 represents pad tokens.
    # - 1 represents any other token.
    # Pad tokens are always represented by 0 and all other tokens are positive so:
    mask = tf.math.sign(data)

    # Apply column-wise denoising.
    if do_columns:
        # Gather the starting values of each row of the mask.
        # This will highlight all rows starting with pad tokens.
        col = mask[..., 0]
        # Apply the cumulative product to the isolated column.
        # Since 1*1=1, 1*0=0, 0*0=0, the column is left with a sequence of 1's followed only by 0's.
        # This will allow us to remove all values after first empty word.
        col = tf.math.cumprod(col, axis=-1)
        # Multiply this isolated column back over the mask.
        mask = tf.multiply(mask, col[..., tf.newaxis])

    # Apply row-wise denoising.
    # Use cumulative product for the same reasoning as in column-wise denoising.
    mask = tf.math.cumprod(mask, axis=-1)

    # Multiply the (binary) mask back over the data.
    result = tf.math.multiply(data, mask)
    return result

@tf.keras.utils.register_keras_serializable(package="CGAEL", name="LanguageDenoiseLayer")
class LanguageDenoiseLayer(tf.keras.layers.Layer):
    def __init__(self, do_columns):
        self.do_columns = do_columns
        super(LanguageDenoiseLayer, self).__init__()

    def call(self, data):
        return denoise_language(data, do_columns=self.do_columns)