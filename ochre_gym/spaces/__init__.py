from typing import Tuple, List
import numpy as np


class SubSpaceType:
    """Enum for the action type of the subspace."""
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'


def register_equipment_subspace(list_of_dicts: List):
    """Converts a list of dictionaries to a dictionary of dictionaries.

    Args:
        list_of_dicts (list): A list of dictionaries

    Returns:
        subspace (Dict): A dictionary of dictionaries
    """
    return {d['name']: d for d in list_of_dicts}


class OchreEquipmentSubSpace:
    """Class for keeping track of attributes for 
    Ochre Equipment actions/obs subspaces.
    """
    def __init__(self, name: str, shape: Tuple, type: SubSpaceType, 
                 upper_bound: np.array = np.array([1.0]),
                 lower_bound: np.array = np.array([0.0]),
                 n: int = 2):
        """
        Args:
            name (str): Name of the subspace
            shape (Tuple): Shape of the subspace
            type (SubSpaceType): Type of the subspace
            upper_bound (np.array): Upper bound of the subspace
            lower_bound (np.array): Lower bound of the subspace
            n (int): Number of discrete values in the subspace
        """
        self.name = name
        self.shape = shape
        self.type = type
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.n = n


    def register(self):
        """Returns a dictionary of the attributes of the subspace."""
        if self.type == SubSpaceType.CONTINUOUS:
            return {
                'name': self.name,
                'type': self.type,
                'shape': self.shape,
                'upper_bound': self.upper_bound,
                'lower_bound': self.lower_bound
            }
        elif self.type == SubSpaceType.DISCRETE:
            return { 
                'name': self.name,
                'type': self.type,
                'shape': self.shape,
                'n': self.n
            }

