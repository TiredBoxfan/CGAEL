import cgael
from cgael.swatches import *

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layer

import pygad
import pygad.kerasga

import numpy as np

import random

class SimpleColorGenerator():
    def __init__(self, swatches:list, blur:float, count:int=1, batch_size:int=None, batch_lock:bool=False, allow_repeats:bool=True):
        """
        Parameters
        ---
        swatches : list(Swatch)
            The swatches to sample from.
        blur : float
            The blurriness value to sample with.
        count : int
            How many colors to generate in a batch entry.
            Will be ignored if 'batch_lock' is set to True.
        batch_size : int
            How big a batch size should be.
            Will be ignored if 'batch_lock' is set to True.
        batch_lock : bool
            If true, 'batch_size' and 'count' will be ignored.
            Instead, each batch will consist of one individual sample per swatch in 'swatches'.
        allow_repeats : bool
            If false, random.sample() will be used over random.choices().
            In this case, 'count' should not be greater than the length of 'swatches'.
        """
        self.swatches = swatches
        self.blur = blur
        self.count = count
        self.batch_size = len(self.swatches) if batch_size is None else batch_size
        self.batch_lock = batch_lock
        self.allow_repeats = allow_repeats

        self.new_batch()

    def new_batch(self):
        """
        Generates a random new batch.
        """
        # Get swatches to sample.
        if self.batch_lock:
            x = [[x] for x in self.swatches]
            random.shuffle(x)
        elif self.allow_repeats:
            x = [random.choices(self.swatches, k=self.count) for _ in range(self.batch_size)]
        else:
            x = [random.sample(self.swatches, k=self.batch_size) for _ in range(self.batch_size)]

        # Sample and return swatches.
        self.current_batch = np.array([[sample_swatch(y, self.blur) for y in z] for z in x])
        return self.current_batch

    def __call__(self):
        return self.current_batch
    

class SimpleColorModel():
    def __init__(self, token_set:cgael.LanguageTokenSet, word_count:int, word_length:int, color_count:int=1, color_channels:int=3, listener_embedding_size:int=4, loss=None):
        self.token_set = token_set
        self.word_count = word_count
        self.word_length = word_length
        self.color_count = color_count
        self.color_channels = color_channels

        self.loss = keras.losses.MeanAbsoluteError() if loss is None else loss

        self.speaker = self._build_speaker()
        self.listener = self._build_listener(listener_embedding_size)
        self.model = self._build_model(self.speaker, self.listener)

    def _build_speaker(self):
        x = y = layer.Input((self.color_count, self.color_channels))
        y = layer.Flatten()(y)
        y = layer.Dense(self.word_count * self.word_length * self.token_set.token_count, activation="sigmoid")(y)
        y = layer.Reshape((self.word_count, self.word_length, self.token_set.token_count))(y)
        
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
        y = speaker(y)
        y = cgael.layers.ArgmaxLayer()(y)
        y = z = cgael.layers.LanguageDenoiseLayer(do_columns=True)(y)
        y = listener(y)
        
        return keras.Model(x, [z, y])
    
    def _fitness(self, ga_inst, solution, sol_idx):
        pred = pygad.kerasga.predict(model=self.model, solution=solution, data=self.generator())
        loss_score = self.loss(self.generator(), pred[1]).numpy()
        return -loss_score
    
    def _generation_callback(self, ga_inst):
        print("Completed Generation:", ga_inst.generations_completed)
        print("Generating new batch...")
        self.generator.new_batch()
    
    def train(self, generator, generations:int, num_solutions:int, num_parents_mating:int, mutation_percent_genes:float=.1, random_mutation_range:float=1.0):
        # Store generator.
        self.generator = generator
        # Convert Keras model to KerasGA model.
        kga = pygad.kerasga.KerasGA(
            model = self.model,
            num_solutions = num_solutions
        )
        # Create and run GA Instance.
        ga_inst = pygad.GA(
            num_generations = generations,
            fitness_func = self._fitness,
            on_generation = self._generation_callback,
            initial_population=kga.population_weights,
            num_parents_mating = num_parents_mating,
            keep_parents = 0,
            keep_elitism = 0,
            mutation_percent_genes = mutation_percent_genes * 100,
            random_mutation_min_val= -abs(random_mutation_range),
            random_mutation_max_val = abs(random_mutation_range)
        )
        ga_inst.run()
        
        # Find best solution.
        print("Identifying best...")
        solution, solution_fitness, solution_idx = ga_inst.best_solution()
        
        # Set model's weights to best solution.
        solution_weights = pygad.kerasga.model_weights_as_matrix(self.model, solution)
        self.model.set_weights(solution_weights)
        
        # Return 'ga_inst' for further uses.
        return ga_inst

    def save_weights(self, npy_file):
        np.save(npy_file, self.model.get_weights())

    def load_weights(self, npy_file):
        weights = np.load(npy_file, allow_pickle=True)
        self.model.set_weights(weights)