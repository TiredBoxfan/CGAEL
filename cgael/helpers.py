# String Helper Functions
 
def ischar(token):
    """
    Returns true only if provided 'token' is a string with a length of exactly 1.
    """
    if not isinstance(token, str):
        return False
    return len(token) == 1

# List Helper Functions

def list_len_trim_to(data:list, length:int):
    """
    If a list is longer than the provided length, it will be returned trimmed to the provided length.
    If the list is shorter or equal in length, it will be returned unchanged.
    """
    return data[:min(len(data), length)]

def list_len_pad_to(data:list, length:int, pad):
    """
    If a list is shorter than the provided length, it will be returned padded to the provided length by the pad token.
    If the list is longer or equal in length, it will be returned unchanged.
    """
    return data + [pad] * (length - len(data))

def list_len_force_to(data:list, length:int, pad):
    """
    If a list is longer than the provided length, it will be returned trimmed to the provided length.
    If a list is shorter than the provided length, it will be returned padded to the provided length by the pad token.
    If the list is equal in length, it will be returned unchanged.
    """
    data = list_len_trim_to(data, length)
    return list_len_pad_to(data, length, pad)