This code calculates the electric potential in one or multiple cylindrical electrodes used in a penning trap. Figure_1 shows typical output of a three electrode trap.

The code leverages the mirror and cylindrical symmetry present in a penning trap to make the calulcation more efficient.

The governing equation solved is Laplace's equation in polar coordinates. Note the use of Sparse arrays and sparse linear solvers to minimize runtime.