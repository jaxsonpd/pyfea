## @file pyfea.py
# @brief the main file for the pyfea module containing the high level functions
# @author Jack Duignan (JackpDuignan@gmail.com)
# @datemodified 27-07-2023

import numpy as np
import matplotlib.pyplot as plt

def local_bar(E: float, A: float, L: float) -> np.ndarray:
    """ Creates a local stiffness matrix for a bar element 
    ### Parameters:
    E : float
        The Young's modulus of the bar
    A : float
        The cross sectional area of the bar
    L : float
        The length of the bar

    ### Returns:
    out: ndarray
        A 2x2 numpy array representing the local stiffness matrix
    """    
    base_matrix = np.array([[1, -1], [-1, 1]])
    return (E * A)/(L) * base_matrix 

def global_bar(k: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """ Creates the stiffness matrix for a bar element in global coordinates
    ### Parameters:
    k : ndarray
        The local stiffness matrix for the bar element
    angle : float
        The angle of the bar element in degrees

    ### Returns:
    out: tuple[ndarray, ndarray]
        A tuple containing the global stiffness matrix and the transformation matrix
    """
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))
    T = np.array([[c, s, 0, 0], [0, 0, c, s]])
    Khat = T.T @ k @ T
    return Khat, T

if __name__ == "__main__":
    print("This is the main file for the pyfea module")