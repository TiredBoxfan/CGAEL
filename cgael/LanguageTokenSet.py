import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layer

from cgael.helpers import ischar, list_len_force_to

class LanguageTokenSet():
    def __init__(self, alphabet_tokens:list, pad_token:str):
        # Ensure the pad_token is a single character.
        if not ischar(pad_token):
            raise ValueError(f"'pad_token' must be a single character string. Got {pad_token}.")
        self.pad_token = pad_token

        # Get the alphabet_tokens as both a list and a set.
        # The list will maintain the order of the tokens. (Later stored.)
        # The set will allow for fast error checking. (Discarded after.)
        if isinstance(alphabet_tokens, list):
            alph_list = [x for x in alphabet_tokens] # deepcopy
            alph_set = set(alphabet_tokens)
        elif isinstance(alphabet_tokens, set):
            alph_list = list(alphabet_tokens)
            alph_set = alphabet_tokens
        elif isinstance(alphabet_tokens, str):
            alph_list = [*alphabet_tokens]
            alph_set = set(alph_list)
        else:
            raise ValueError(f"'alphabet_tokens' should be provided as either a list, set or string. Got type '{type(alphabet_tokens)}'.")
        
        # Check for errors in alphabet.

        # List and set size should be the same. If different, this indicates a duplicate token.
        if len(alph_list) != len(alph_set):
            raise ValueError(f"{len(alph_list)-len(alph_set)} duplicate tokens detected in 'alphabet_tokens'.")
        # Pad token cannot be in alphabet.
        if self.pad_token in alph_set:
            raise ValueError(f"'pad_token' {self.pad_token} cannot be in alphabet.")
        # Check that each token is a single character.
        for x in alph_list:
            if not ischar(x):
                raise ValueError(f"Each entry of 'alphabet_tokens' must be a single character. Got '{x}'.")
        
        # Alphabet tokens validated. Store.
        self.alphabet_tokens = alph_list

        # Instantiate encoder & decoder.
        self._encoder = layer.StringLookup(vocabulary=self.alphabet_tokens, oov_token=self.pad_token, output_mode="int", invert=False)
        self._decoder = layer.StringLookup(vocabulary=self.alphabet_tokens, oov_token=self.pad_token, output_mode="int", invert=True)

    @property
    def token_count(self):
        """
        The number of tokens in the language, including the pad token.
        """
        return self.encoder.vocabulary_size()
    
    def encode(self, data:str, shape:tuple):
        """
        Encodes a Python string into a TensorFlow tensor.
        """
        # Different rank shapes need to be handled differently.
        if len(shape) == 1:
            # Just need to ensure that the resulting list is of the proper size.
            data = list_len_force_to([*data], shape[0], self.pad_token)
        elif len(shape) == 2:
            # Split the input by whitespace.
            data = [list(x) for x in data.split()]
            # Ensure each word is the proper length.
            data = [list_len_force_to(x, shape[1], self.pad_token) for x in data]
            # Ensure there are the proper number of "words".
            data = list_len_force_to(data, shape[0], [self.pad_token]*shape[1])
        else:
            raise ValueError(f"Unsupported shape rank={len(shape)}.")
        # Send to encoder.
        return self._encoder(data)
    
    def decode(self, data):
        """
        Encodes a TensorFlow tensor, NumPy array or Python array to a Python string.
        """
        # Send to decoder.
        data = self._decoder(data).numpy()
        # Different rank shapes need to be handled differently.
        if len(data.shape) == 1:
            data = b''.join(data).decode("utf-8").rstrip(self.pad_token)
        elif len(data.shape) == 2:
            # Join letters.
            data = [b''.join(x).decode("utf-8").rstrip(self.pad_token) for x in data]
            # Join words.
            data = ' '.join([x for x in data if len(x) > 0])
        else:
            raise ValueError(f"Unsupported shape rank={len(data.shape)}.")
        return data