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


def assemble_frame(Khat: np.ndarray, A: np.ndarray) -> np.ndarray:
    """ Assembles the global stiffness matrix for a frame element
    ### Parameters:
    Khat : ndarray
        The global stiffness matrix for the frame element
    A : ndarray
        The assembly matrix for the frame element

    ### Returns:
    out: ndarray
        The global stiffness matrix for the frame element KG
    """
    return A @ Khat @ A.T

def plot_deflected_frame(node1XG: float, node1YG: float, node2XG: float, node2YG: float, d_e: np.ndarray, N_points: float = 100, disp_scale: float = 10):
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
    disp_scale : float (10)
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
    axial_1 = 1 - x_e/L
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

def find_UDL(L: float, wHat: float, lambdaMat: np.ndarray, Assem: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Creates a uniform distributed load vector for a frame element
    ### Parameters:
    L : float
        The length of the frame element
    wHat : float
        The magnitude of the uniform distributed load in the local coordinate system
    lambdaMat : ndarray
        The transformation matrix for the frame element
    Assem : ndarray
        The assembly matrix for the frame element
        
    ### Returns:
    out: ndarray
        The uniform distributed load vector for the frame element
    """

    # Find the local uniform distributed load vector
    f_e = np.array([[0],
                    [wHat * L / 2],
                    [wHat * L**2 / 12],
                    [0],
                    [wHat * L / 2],
                    [-wHat * L**2 / 12]])

    # Transform to global coordinates
    f_G = lambdaMat.T @ f_e

    # Assemble the load vector
    return Assem @ f_G, f_G, f_e

def find_LVL(L: float, wHat: float, lambdaMat: np.ndarray, Assem: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Creates a linearly varying load vector for a frame element
    ### Parameters:
    L : float
        The length of the frame element
    wHat : float
        The magnitude of the linearly varying load in the local coordinate system
    lambdaMat : ndarray
        The transformation matrix for the frame element
    Assem : ndarray
        The assembly matrix for the frame element
        
    ### Returns:
    out: [ndarray, ndarray, ndarray]
        The linearly varying load vector in the global and element coordinate systems
    """

    # Find the local linearly varying load vector
    f_e = np.array([[0],
                    [3 * wHat * L / 20],
                    [wHat * L**2 / 30],
                    [0],
                    [7 * wHat * L / 20],
                    [-wHat * L**2 / 20]])
    
    
    # Transform to global coordinates
    f_G = lambdaMat.T @ f_e

    # Assemble the load vector
    return Assem @ f_G, f_G, f_e

def find_point_load(L: float, wHat: float, lambdaMat: np.ndarray, Assem: np.ndarray, a: float = -1) -> np.ndarray:
    """ Creates a point load vector for a frame element
    ### Parameters:
    L : float
        The length of the frame element
    wHat : float
        The magnitude of the point load in the local coordinate system
    lambdaMat : ndarray
        The transformation matrix for the frame element
    Assem : ndarray
        The assembly matrix for the frame element
    a : float (-1)
        The distance from the first node to the point load if -1 then point load 
        is halfway along the frame element
        
    ### Returns:
    out: [ndarray, ndarray, ndarray]
        The linearly varying load vector in the global and element coordinate systems
    """

    if (a == -1):
        a = L / 2

    # Find the local point load vector
    f_e = wHat * np.array([[0],
                           [1-3*(a/L)**2+2*(a/L)**3],
                           [(a**3/L**2)-(2*a**2)/L+a],
                           [0],
                           [3*(a/L)**2-2*(a/L)**3],
                           [(a**3/L**2)-(a**2)/L]])

    # Transform to global coordinates
    f_G = lambdaMat.T @ f_e

    # Assemble the load vector
    return Assem @ f_G, f_G, f_e

def find_axial_UDL(L: float, pHat: float, lambdaMat: np.ndarray, Assem: np.ndarray, ) -> np.ndarray:
    """ Creates an axial uniform distributed load vector for a frame element
    ### Parameters:
    L : float
        The length of the frame element
    pHat : float
        The magnitude of the axial uniform distributed load in the local coordinate system
    lambdaMat : ndarray
        The transformation matrix for the frame element
    Assem : ndarray
        The assembly matrix for the frame element
        
    ### Returns:
    out: [ndarray, ndarray, ndarray]
        The linearly varying load vector in the global and element coordinate systems
    """

    # Find the local axial uniform distributed load vector
    f_e = pHat* np.array([[L/2],
                    [0],
                    [0],
                    [L/2],
                    [0],
                    [0]])

    # Transform to global coordinates
    f_G = lambdaMat.T @ f_e

    # Assemble the load vector
    return Assem @ f_G, f_G, f_e

def find_axial_point_load(L: float, pHat: float, lambdaMat: np.ndarray, Assem: np.ndarray, a: float = -1) -> np.ndarray:
    """ Creates an axial point load vector for a frame element
    ### Parameters:
    L : float
        The length of the frame element
    pHat : float
        The magnitude of the axial point load in the local coordinate system
    lambdaMat : ndarray
        The transformation matrix for the frame element
    Assem : ndarray
        The assembly matrix for the frame element
    a : float (-1)
        The distance from the first node to the axial point load if -1 then axial point load 
        is halfway along the frame element
        
    ### Returns:
    out: [ndarray, ndarray, ndarray]
        The linearly varying load vector in the global and element coordinate systems
    """

    if (a == -1):
        a = L / 2

    # Find the local axial point load vector
    f_e = pHat * np.array([[1-(a/L)],
                           [0],
                           [0],
                           [a/L],
                           [0],
                           [0]])

    # Transform to global coordinates
    f_G = lambdaMat.T @ f_e

    # Assemble the load vector
    return Assem @ f_G, f_G, f_e

def find_global_point_defelections(x_e: float, L: float, d_e: np.ndarray, angle: float) -> [float, float]:
    """ Find the deflections in global coordinates at a spesifice frame point
    ### Parameters:
    x_e : float
        The distance from the first node to the point of interest
    L : float
        The length of the frame element
    d_e : ndarray
        The displacement vector of the frame element (6x1)
    angle : float
        The angle of the frame element in degrees

    ### Returns:
    out: [float, float]
        The deflections in the X and Y directions in the global coordinate system
    """
    axial_1 = 1 - x_e/L
    axial_2 = x_e/L

    trans_1 = 1 - 3 * (x_e/L)**2 + 2 * (x_e/L)**3
    trans_2 = (x_e**3/L**2) - (2 * x_e**2)/L + x_e
    trans_3 = 3 * (x_e/L)**2 - 2 * (x_e/L)**3
    trans_4 = (x_e**3/L**2) - (x_e**2)/L

    # Find overall deflections in axial and transverse directions
    axial = axial_1 * d_e[0] + axial_2 * d_e[3]
    trans = trans_1 * d_e[1] + trans_2 * d_e[2] + trans_3 * d_e[4] + trans_4 * d_e[5]

    X_G = axial * np.cos(np.deg2rad(angle)) - trans * np.sin(np.deg2rad(angle))
    Y_G = axial * np.sin(np.deg2rad(angle)) + trans * np.cos(np.deg2rad(angle))

    return X_G, Y_G

def find_local_point_defelections(x_e: float, L: float, d_e: np.ndarray) -> [float, float]:
    """ Find the axial and transverse deflections in local coordinates at a spesifice frame point
    ### Parameters:
    x_e : float
        The distance from the first node to the point of interest
    L : float
        The length of the frame element
    d_e : ndarray
        The displacement vector of the frame element (6x1)
        
    ### Returns:
    out: [float, float]
        The deflections in the axial and transverse directions in the local coordinate system
    """
    axial_1 = 1 - x_e/L
    axial_2 = x_e/L

    trans_1 = 1 - 3 * (x_e/L)**2 + 2 * (x_e/L)**3
    trans_2 = (x_e**3/L**2) - (2 * x_e**2)/L + x_e
    trans_3 = 3 * (x_e/L)**2 - 2 * (x_e/L)**3
    trans_4 = (x_e**3/L**2) - (x_e**2)/L

    # Find overall deflections in axial and transverse directions
    axial = axial_1 * d_e[0] + axial_2 * d_e[3]
    trans = trans_1 * d_e[1] + trans_2 * d_e[2] + trans_3 * d_e[4] + trans_4 * d_e[5]

    return axial, trans

def find_Strain(d: np.ndarray, L: float) -> float:
    """ Find the strain in the frame element 
    ### Parameters:
    d : ndarray
        The local displacement vector for the frame element
    L : float
        The length of the frame element

    ### Returns:
    out: float
        The strain in the frame element
    """
    return (d[3]-d[0]) / L

def find_Stress(E: float, d: np.ndarray, L: float) -> float:
    """ Find the stress in the frame element 
    ### Parameters:
    E : float
        The modulus of elasticity of the frame element
    d : ndarray
        The local displacement vector for the frame element
    L : float
        The length of the frame element

    ### Returns:
    out: float
        The stress in the bar element
    """
    return E * (d[3]-d[0]) / L
