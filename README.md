# Python FEA
This is a python module contaning function to perform simple FEA anlysis of structures

# Design Philosophy
This projects loosely follows the TDD process. The test functions are located in method\_test.py.

# Currently supported methods
local\_bar - this function creates the stiffness matrix for a bar element in    local coordinates.

global\_bar - This function transforms the stiffness matrix and angle into
    global coordinates and also creates the transform matrix. 

# Installation
Currently this module is not available on pip. To import this module clone the repository to the working directory and add the following line to the top of your python file:
```from pyfea import pyfea```