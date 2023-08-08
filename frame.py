## @file beam.py
# @brief this file contains the functions for the beam element
# @author Jack Duignan (JackpDuignan@gmail.com)
# @datemodified 04-07-2023


import numpy as np
import matplotlib.pyplot as plt

def local_frame(E: float, I:float, A:float, L:float) -> np.ndarray:
    """ Creates a local stiffness matrix for a frame element
    ### Parameters:
    E : float
        The Young's modulus of the frame
    I : float
        The second moment of area of the frame
    A : float
        The cross sectional area of the frame
    L : float 
        The length of the frame

    ### Returns:
    out: ndarray
        A 4x4 numpy array representing the local stiffness matrix
    """
    beta = A * L**2 / I
    base_matrix = np.array([[beta, 0, 0, -beta, 0, 0],
                            [0, 12, 6 * L, 0, -12, 6 * L],
                            [0, 6 * L, 4 * L**2, 0, -6 * L, 2 * L**2],
                            [-beta, 0, 0, beta, 0, 0],
                            [0, -12, -6 * L, 0, 12, -6 * L],
                            [0, 6 * L, 2 * L**2, 0, -6 * L, 4 * L**2]])
    return (E * I / L**3) * base_matrix


def global_frame(k: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """ Creates the stiffness matrix for a frame element in global coordinates
    ### Parameters:
    k : ndarray
        The local stiffness matrix for the frame element
    angle : float
        The angle of the frame element in degrees

    ### Returns:
    out: tuple[ndarray, ndarray]
        A tuple containing the global stiffness matrix and the transformation matrix
    """
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))

    lambdaMat = np.array([[c, s, 0],
                          [-s, c, 0],
                          [0, 0, 1]])
    LambdaMat = np.block([[lambdaMat, np.zeros((3, 3))],
                          [np.zeros((3, 3)), lambdaMat]])
    
    Khat = LambdaMat.T @ k @ LambdaMat

    return Khat, LambdaMat


def plot_deflected_frame(node1XG: float, node1YG: float, node2XG: float, node2YG: float, d_e: np.ndarray, N_points: float = 100, disp_scale: float = 100):
    """ Plot the fully deflected frame shape 
    ### Parameters:
    node1XG : float
        The x coordinate of the first node in the global coordinate system
    node1YG : float
        The y coordinate of the first node in the global coordinate system
    node2XG : float
        The x coordinate of the second node in the global coordinate system
    node2YG : float
        The y coordinate of the second node in the global coordinate system
    d_e : ndarray
        The displacement vector of the frame element (6x1)
    N_points : float (100)
        The number of points to plot along the frame element miniumum 20 recommended
    disp_scale : float (100)
        The scale of the displacements in the plot
        
    ### Returns:
    out: None
    """
    # Find information about the frame
    L = np.sqrt((node2XG - node1XG)**2 + (node2YG - node1YG)**2)
    angle = np.rad2deg(np.arctan2(node2YG - node1YG, node2XG - node1XG))
    
    # Create a local coordinate system
    x_e = np.linspace(0, L, N_points)

    # Find local shape functions
    axial_1 = (1 - x_e)/L
    axial_2 = x_e/L

    trans_1 = 1 - 3 * (x_e/L)**2 + 2 * (x_e/L)**3
    trans_2 = (x_e**3/L**2) - (2 * x_e**2)/L + x_e
    trans_3 = 3 * (x_e/L)**2 - 2 * (x_e/L)**3
    trans_4 = (x_e**3/L**2) - (x_e**2)/L

    # Find overall deflections in axial and transverse directions
    axial = axial_1 * d_e[0] + axial_2 * d_e[3]
    trans = trans_1 * d_e[1] + trans_2 * d_e[2] + trans_3 * d_e[4] + trans_4 * d_e[5]

    # Transform to global coordinates
    deflections_XG = axial * np.cos(np.deg2rad(angle)) - trans * np.sin(np.deg2rad(angle))
    deflections_YG = axial * np.sin(np.deg2rad(angle)) + trans * np.cos(np.deg2rad(angle))

    # Find undeflected base shape
    Undeflected_XG = np.linspace(node1XG, node2XG, N_points)
    Undeflected_YG = np.linspace(node1YG, node2YG, N_points)

    # Find the deflected shape
    Deflected_XG = Undeflected_XG + disp_scale * deflections_XG
    Deflected_YG = Undeflected_YG + disp_scale * deflections_YG

    # Plot the frame
    plt.plot(Undeflected_XG, Undeflected_YG, 'k--', label='Undeflected')
    plt.plot(Deflected_XG, Deflected_YG, 'g-', label='Deflected')