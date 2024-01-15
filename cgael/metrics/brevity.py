import tensorflow as tf

def simple_brevity(data):
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
    return loss