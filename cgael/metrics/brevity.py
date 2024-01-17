import tensorflow as tf

@tf.function
def simple_brevity(data, power=1):
    # Get the number of non-padded tokens for each entry of the batch.
    sums = tf.math.reduce_sum(tf.math.sign(data), axis=[-2, -1])
    # Calculate the score of each entry of the batch 'n' such that:
    # - if n == 0: maximum area of entry (worst possible score)
    # - else: n - 1 (for calibration purposes)
    # Remember: Golf rules; lower is better.
    scores = tf.where(
        condition = tf.math.equal(sums, 0),
        x = tf.constant(data.shape[-2] * data.shape[-1], dtype=sums.dtype),
        y = tf.math.subtract(sums, 1)
    )
    # Calculate the final loss by dividing sum of scores over maximum scores.
    total = tf.math.reduce_sum(scores)
    shape = tf.shape(data, out_type=sums.dtype)
    area = tf.math.reduce_prod(shape)
    loss = tf.math.divide(total, area)
    loss = tf.math.pow(loss, power)
    return loss

@tf.function
def power_brevity(data, word_length_power=2, word_count_power=2):
    @tf.function
    def helper(mask):
        # STEP 1: WORD LENGTH POWER
        # Get the lengths of each word.
        x = tf.math.reduce_sum(mask, axis=-1)
        # Divide by maximum length of words, placing the function on the range [0, 1].
        x = tf.math.divide(x, data.shape[-1])
        # Apply word_length_power.
        x = tf.math.pow(x, word_length_power)
        
        # STEP 2: WORD COUNT POWER
        # Get the sum of each word score.
        x = tf.math.reduce_sum(x, axis=-1)
        # Divide by maximum number of words, placing the function on the range [0, 1].
        x = tf.math.divide(x, data.shape[-2])
        # Apply word_count_power.
        x = tf.math.pow(x, word_count_power)
    
        return x
    
    # Get binary mask of data.
    mask = tf.sign(data)
    sums = tf.math.reduce_sum(mask, axis=[-2, -1])
    results = tf.where(
        condition = tf.math.equal(sums, 0),
        x = tf.constant(1, dtype=tf.float64),
        y = helper(mask)
    )
    return tf.reduce_prod(results)