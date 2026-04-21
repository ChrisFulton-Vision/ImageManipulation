import numpy as np

def convert_aftrVec_to_cvVec(vector):
    raise NotImplementedError("Not tested.")
    if vector.shape != (3,):
        raise ValueError(f"Wrong size vector. Expected (3,), got {vector.shape}")
    return np.array([vector[1], -vector[2], -vector[0]])