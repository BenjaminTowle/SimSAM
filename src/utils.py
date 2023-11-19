import numpy as np
import random
import torch

SEED = 42


def set_seed(seed: int = None):
    # Set random seed
    if seed is None:
        seed = SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RegistryMixin:

    subclasses = {}

    @classmethod
    def register_subclass(cls, model_type):
        def decorator(subclass):
            if cls not in cls.subclasses:
                cls.subclasses[cls] = {}
            cls.subclasses[cls][model_type] = subclass
            return subclass
        
        return decorator

    @classmethod
    def create(cls, subclass: str):
        if subclass not in cls.subclasses[cls]:
            raise ValueError(f"Bad subclass: {subclass}.")
        
        return cls.subclasses[cls][subclass]

    @classmethod
    def get_subclasses(cls):
        return cls.subclasses[cls]
