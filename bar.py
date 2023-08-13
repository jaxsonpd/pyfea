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

def assemble_bar(Khat: np.ndarray, A: np.ndarray, q: np.ndarray) -> np.ndarray:
    """ Assembles the bar element into the global stiffness matrix
    ### Parameters:
    Khat : ndarray
        The global stiffness matrix for the bar element
    A : ndarray
        The assembly matrix for the bar element
    q : ndarray
        The global displacement vector

    ### Returns:
    out: ndarray
        The assembled global stiffness matrix
    """
    return A @ Khat @ A.T

def plot_deflected_bar(node1XG: float, node1YG: float, node2XG: float, node2YG: float, d_e: np.ndarray, disp_scale: float = 100) -> None:
    """ Plot the deflected and undeformed shape of a bar element
    ### Parameters:
    node1XG : float
        The global X coordinate of node 1
    node1YG : float
        The global Y coordinate of node 1
    node2XG : float
        The global X coordinate of node 2
    node2YG : float
        The global Y coordinate of node 2
    d_e : ndarray
        The local displacement vector for the bar element
    disp_scale : float (100)
        The scale of the displacement vector
    
    ### Returns:
    out: None
    """
    L = np.sqrt((node2XG-node1XG)**2 + (node2YG-node1YG)**2)
    angle = np.rad2deg(np.arctan2(node2YG-node1YG, node2XG-node1XG))

    # Plot the undeformed shape
    plt.plot([node1XG, node2XG], [node1YG, node2YG], 'k-', label='Undeformed')

    # Plot the deflected shape
    plt.plot([node1XG, node2XG+float(d_e[0])*disp_scale], [node1YG, node2YG+float(d_e[1])*disp_scale], 'r-', label='Deflected')




if __name__ == "__main__":
    print("This is the file containg the bar element functions")