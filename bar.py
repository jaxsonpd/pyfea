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

def findForce(Khat: np.ndarray, A: np.ndarray, q: np.ndarray) -> np.ndarray:
    """ Find the local force vector for the bar element 
    ### Parameters:
    Khat : ndarray
        The global stiffness matrix for the bar element
    A : ndarray
        The assembly matrix for the bar element
    q : ndarray
        The global displacement vector

    ### Returns:
    out: ndarray
        The local force vector for the bar element
    """
    return Khat @ (A.T @ q)

def findDisplacement(lambdaMat: np.ndarray, A: np.ndarray, q: np.ndarray) -> np.ndarray:
    """ Find the local displacement vector for the bar element 
    ### Parameters:
    lambdaMat : ndarray
        The local force vector for the bar element
    A : ndarray
        The assembly matrix for the bar element
    q : ndarray
        The global displacement vector

    ### Returns:
    out: ndarray
        The local displacement vector for the bar element
    """
    return lambdaMat @ (A.T @ q)

def findStrain(d: np.ndarray, L: float) -> float:
    """ Find the strain in the bar element 
    ### Parameters:
    d : ndarray
        The local displacement vector for the bar element
    L : float
        The length of the bar element

    ### Returns:
    out: float
        The strain in the bar element
    """
    return (d[1]-d[0]) / L




if __name__ == "__main__":
    print("This is the file containg the bar element functions")