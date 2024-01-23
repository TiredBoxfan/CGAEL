import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layer

import random

class LanguageDiscriminatorGenerator(keras.utils.Sequence):
    def __init__(self, tokens, real_words:list, encode_length:int, batch_size:int, batch_count:int, len_min:int=1, len_max:int=None):
        """
        Parameters
        ---
        tokens : cgael.LanguageTokenSet
            The language tokens to use for encoding and for the fake words.
        real_words : list(String) OR set(String)
            Words that you would like the discriminator to mark as real.
            These words should be representative of the style of the language; consider excluding outliers.
        encode_length : int
            How long a word should be encoded as. Will have the shape (encode_length,).
        batch_size : int
            How big a batch is.
        batch_count : int
            How many batches per generation.
        len_min : int
            The minimum length for a fake word.
            This value defaults to 1.
        len_max : int
            The maximum length for a fake word.
            This value defaults to the value of encode_length.
        """
        self.tokens = tokens
        self.word_list = list(real_words) # For random values (set is not subscriptable).
        self.word_set = set(real_words) # For a fast way to see if a word in real or not.
        self.encode_shape = (encode_length,)
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.len_min = len_min
        self.len_max = encode_length if len_max is None else len_max
        
    def gibberish(self, length=None):
        """
        Generates a random sequence of letters that may or may not match up to a real word.
        
        Parameters
        ---
        length : int
            If supplied, it will generate a "word" of exactly that length.
            Otherwise, it will generate a "word" between self.len_min and self.len_max in length.
        """
        length = random.randint(self.len_min, self.len_max) if length is None else length
        return ''.join(random.choices(self.tokens.alphabet_tokens, k=length))
    
    def nonsense(self, length=None):
        """
        Generates a random sequence of letters that is never a real word (as provided).
        
        Parameters
        ---
        length : int
            If supplied, it will generate a "word" of exactly that length.
            Otherwise, it will generate a "word" between self.len_min and self.len_max in length.
        """
        while True:
            x = self.gibberish(length=length)
            if x not in self.word_set:
                return x
            
    # Required.
    def __len__(self):
        return self.batch_count

    # Required.
    def __getitem__(self, index=0):
        #print(f"[LanguageDiscriminatorGenerator.__getitem__] Called with index={index}.")
        
        ls_x = []
        ls_y = []
        
        def append_text(text, value):
            ls_x.append(self.tokens.encode(text, shape=self.encode_shape))
            ls_y.append(value)
        
        for _ in range(0, self.batch_size):
            if random.random() < 0.5: # 50% chance of real value:
                append_text(random.choice(self.word_list), 1)
            else: # 50% chance of fake value:
                append_text(self.nonsense(), 0)
        
        return tf.stack(ls_x), tf.stack(ls_y)
    
class LanguageDiscriminatorModel():
    def __init__(self, word_length, compile=True):
        self.word_length = word_length
        
        self.model = self._build_model()
        
        if compile:
            self.model.compile(
                loss = self._model_loss(),
                optimizer = self._model_optimizer(),
                metrics = self._model_metrics()
            )
        
    def _build_model(self):
        x = y = layer.Input((self.word_length,))
        y = layer.Reshape((self.word_length, 1))(y)
        y = layer.Conv1D(self.word_length, 5, padding="same", activation="relu")(y)
        y = layer.Dense(1, activation="relu")(y)
        y = layer.Reshape((self.word_length,))(y)
        y = layer.Dense(self.word_length, activation="relu")(y)
        y = layer.Dense(1, activation="sigmoid")(y)
        return keras.Model(x, y)
        
    def _model_loss(self):
        return keras.losses.BinaryCrossentropy(from_logits=False)
    
    def _model_optimizer(self):
        return keras.optimizers.Adam(0.001)
    
    def _model_metrics(self):
        return ["accuracy"]
    
    def train(self, training_generator:LanguageDiscriminatorGenerator, epochs:int):
        return self.model.fit(training_generator, epochs=epochs)
    
    def calculate_loss(self, data):
        # Reshape tensor so that it is a list of all words.
        # This is okay because they will all be averaged individually.
        x = tf.reshape(data, (-1, tf.shape(data)[-1]))
        # Remove all words that start with 0 from the list,
        # but still keep the first words of each sentence.
        msk_first = tf.equal(tf.math.mod(tf.range(tf.shape(x)[-2]), tf.shape(data)[-2]), 0)
        msk_nonzero = tf.math.not_equal(x[:,0], 0)
        msk_join = tf.logical_or(msk_first, msk_nonzero)
        x = tf.boolean_mask(x, msk_join, axis=0)
        # Prevent errors.
        if tf.equal(tf.size(x), 0):
            return tf.constant(1.)
        # Evaluate remaining words with discriminator.
        x = self.model(x)
        # Calculate the mean and present value as loss.
        return tf.clip_by_value(1 - tf.reduce_mean(x), 0., 1.)
    
    def __call__(self, data):
        return self.model(data)
        #return self.calculate_loss(data)