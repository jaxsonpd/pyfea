## @file module_test.py
# @brief a file to test the pyfea module
# @author Jack Duignan (JackpDuignan@gmail.com)

import pyfea
import unittest
import numpy as np

class Test_local_bar(unittest.TestCase):
    """ A class to test the local_bar function """
    def setUp(self) -> None:
        pass

    def test_unit_bar(self) -> None:
        """ A test to check the local stiffness matrix of a unit bar """
        E = 1
        A = 1
        L = 1

        expected = np.array([[1, -1], [-1, 1]])

        actual = pyfea.local_bar(E, A, L)

        np.testing.assert_array_equal(actual, expected)

    def test_simple_bar(self):
        """ A test to check the local stiffness matrix of a simple bar """
        E = 1
        A = 1
        L = 2

        expected = np.array([[ 0.5, -0.5],[-0.5,  0.5]])

        actual = pyfea.local_bar(E, A, L)

        np.testing.assert_array_equal(actual, expected)

    def test_actual_bar(self):
        """ A test to check the local stiffness matrix of an actual bar element """
        E = 200e9
        A = 0.1 **2 * np.pi / 4
        L = 10

        expected = np.array([[1.57079633e+08, -1.57079633e+08], [-1.57079633e+08, 1.57079633e+08]])

        actual = pyfea.local_bar(E, A, L)

        np.testing.assert_array_almost_equal(np.round(actual, -5), np.round(expected, -5))

class Test_global_bar(unittest.TestCase):
    """ A class to test the global_bar function """
    def setUp(self) -> None:
        pass

    def test_flat_unit_bar(self) -> None:
        """ A test to check the global stiffness matrix of a unit bar """
        k1 = np.array([[1, -1], [-1, 1]])
        angle = 0

        expected_Khat = np.array([[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]])
        expected_Lambda = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        actual_Khat, actual_lambda = pyfea.global_bar(k1, angle)

        np.testing.assert_array_equal(actual_Khat, expected_Khat)
        np.testing.assert_array_equal(actual_lambda, expected_Lambda)

    def test_actual_flat_bar(self) -> None:
        """ Find the global stiffness matrix of a flat bar of actual dimensions """
        k1 = np.array([[1.57079633e+08, -1.57079633e+08], [-1.57079633e+08, 1.57079633e+08]])
        angle = 0

        expected_Lambda = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        expected_khat = np.array([[ 1.571e+08,  0.000e+00, -1.571e+08,  0.000e+00],
                                [ 0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00],
                                [-1.571e+08,  0.000e+00,  1.571e+08,  0.000e+00],
                                [ 0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00]])

        actual_Khat, actual_lambda = pyfea.global_bar(k1, angle)

        np.testing.assert_array_equal(actual_lambda, expected_Lambda)
        np.testing.assert_array_equal(np.round(actual_Khat, -5), np.round(expected_khat, -5))


if __name__ == "__main__":
    unittest.main()