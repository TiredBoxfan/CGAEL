import cgael
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layer
from cgael.swatches import *

class SimpleColorGenerator():
    def __init__(self, swatches:list, blur:float, fixed:bool, batch_size:int=None):
        """
        Parameters
        ---
        swatches : list(Swatch)
            The swatches to sample from.
        blur : float
            The blurriness value to sample with.
        fixed : bool
            If True, each batch will be consist of 1 sample per swatch.
            If False, each batch will consist of 'batch_size' random samples from random swatches.
        batch_size : int
            How big a batch size should be.
            Will be ignored if 'fixed' is set to True.
        """
        self.swatches = swatches
        self.blur = blur
        self.fixed = fixed
        self.batch_size = len(self.swatches) if batch_size is None else batch_size

        self.new_batch()

    def new_batch(self):
        """
        Generates a random new batch.
        """
        if self.fixed:
            x = [x for x in self.swatches] # deepcopy.
            random.shuffle(x)
        else:
            x = random.choices(self.swatches, k=self.batch_size)
        self.current_batch = np.array([sample_swatch(y, self.blur) for y in x])
        return self.current_batch

    def __call__(self):
        return self.current_batch
    

class SimpleColorModel():
    def __init__(self, token_set:cgael.LanguageTokenSet, word_count:int, word_length:int, color_count:int=1, color_channels:int=3, listener_embedding_size:int=4):
        self.token_set = token_set
        self.word_count = word_count
        self.word_length = word_length
        self.color_count = color_count
        self.color_channels = color_channels

        self.speaker = self._build_speaker()
        self.listener = self._build_listener(listener_embedding_size)
        self.model = self._build_model(self.speaker, self.listener)

    def _build_speaker(self):
        # (x, 2, 3)
        x = y = layer.Input((self.color_count, self.color_channels))
        # (x, 2, 3)
        y = layer.Flatten()(y)
        # (x, 6)
        y = layer.Dense(self.word_count * self.word_length * self.token_set.token_count, activation="sigmoid")(y)
        # (x, 2*3*5) -> (x, 30)
        y = layer.Reshape((self.word_count, self.word_length, self.token_set.token_count))(y)
        # (x, 2, 3, 5)

        y = cgael.layers.ArgmaxLayer()(y)
        # (x, 2, 3)
        #y = cgael.layers.LanguageDenoiseLayer(do_columns=True)(y)
        # (x, 2, 3)
        
        return keras.Model(x, y, name="speaker")
    
    def _build_listener(self, embedding_size):
        x = y = layer.Input((self.word_count, self.word_length))
        y = layer.Embedding(self.token_set.token_count, embedding_size, embeddings_initializer="random_normal")(y)
        y = layer.Flatten()(y)
        y = layer.Dense((self.color_count * self.color_channels), activation="sigmoid")(y)
        y = layer.Reshape((self.color_count, self.color_channels))(y)

        return keras.Model(x, y, name="listener")
    
    def _build_model(self, speaker, listener):
        x = y = layer.Input((self.color_count, self.color_channels))
        y = z = speaker(y)
        y = listener(y)
        
        return keras.Model(x, [z, y])
    
    #def _fitness(self, ga_instance, solution, sol_idx):
    #    pred = pygad.kerasga.predict(model=self.model, solution=solution, data=)