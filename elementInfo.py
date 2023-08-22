## @file elementInfo.py
# @brief this file contains function to help interpreate beam elements
# @date 12-08-2023
# @author Jack Duignan (JackpDuignan@gmail.com)

import numpy as np
import matplotlib.pyplot as plt

def find_local(K: np.ndarray, Assem: np.ndarray, lambdaMat: np.ndarray, q: np.ndarray) -> [np.ndarray, np.ndarray]:
    """ Find the local force and displacement vectors for the frame element
    ### Parameters:
    K : ndarray
        The local stiffness matrix for the bar element
    Assem : ndarray
        The assembly matrix for the frame element
    lambdaMat : ndarray
        The lambda matrix for the frame element
    q : ndarray
        The global displacement vector
        
    ### Returns:
    out: [ndarray, ndarray]
        A tuple containing the local force and displacement vectors for the frame element
    """
    D_e = Assem.T @ q
    d_e = lambdaMat @ D_e
    f_e = K @ d_e

    return f_e, d_e

def find_global(Khat: np.ndarray, Assem: np.ndarray, q: np.ndarray) -> [np.ndarray, np.ndarray]:
    """ Find the global force and displacement vectors for the frame element 
    ### Parameters:
    Khat : ndarray
        The global stiffness matrix for the frame element
    Assem : ndarray
        The assembly matrix for the frame element
    q : ndarray
        The global displacement vector

    ### Returns:
    out: [ndarray, ndarray]
        A tuple containing the global force and displacement vectors for the frame element
    """
    D_e = Assem.T @ q
    F_e = Khat @ D_e

    return F_e, D_e