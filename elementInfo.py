## @file elementInfo.py
# @brief this file contains function to help interpreate beam elements
# @date 12-08-2023
# @author Jack Duignan (JackpDuignan@gmail.com)

import numpy as np
import matplotlib.pyplot as plt

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