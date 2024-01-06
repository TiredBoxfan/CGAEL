import random
from enum import Enum

class Swatch(Enum):
    WHITE = 1
    BLACK = 2
    RED = 3
    GREEN = 4
    BLUE = 5
    YELLOW = 6
    CYAN = 7
    MAGENTA = 8

def sample_swatch(swatch:Swatch, blurriness=0):
    """
    Samples a Swatch using the BGR colorspace.

    Parameters
    ---
    swatch : Swatch
        The swatch to sample from.
    blurriness : float
        A real number on the range [0, 1] which indicates how "blurry" a single value can be.
        If 0, only 0's and 1's will be returned.
        A higher blurriness value will allow values to be returned closer to 0.5.
    """
    def sample_low():
        return random.random() * blurriness * .5
    def sample_high():
        return 1-sample_low()
    match swatch:
        case Swatch.WHITE:
            return [sample_high(), sample_high(), sample_high()]
        case Swatch.BLACK:
            return [sample_low(), sample_low(), sample_low()]
        case Swatch.RED:
            return [sample_low(), sample_low(), sample_high()]
        case Swatch.GREEN:
            return [sample_low(), sample_high(), sample_low()]
        case Swatch.BLUE:
            return [sample_high(), sample_low(), sample_low()]
        case Swatch.YELLOW:
            return [sample_low(), sample_high(), sample_high()]
        case Swatch.CYAN:
            return [sample_high(), sample_high(), sample_low()]
        case Swatch.MAGENTA:
            return [sample_high(), sample_low(), sample_high()]
        case _:
            raise ValueError(f"Could not sample swatch '{swatch}'.")