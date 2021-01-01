*** RUN THE GUI.PY FILE TO START THE PROGRAM ***

This is a customisable sudoku solver which includes different sudoku boards to solve, based on their difficulty ratings

sudoku_solver.py (Solver #1) solves the sudoku by checking from top left to bottom right for empty spaces every time the solve function is called. Compared to the other solver, it better shows how backtracking works but its less optimised and thus slower.

sudoku_solver_2.py (Solver #2) solves the sudoku by checking the empty spaces from the number with the least possible options (ie. least possibilities of the different numbers that can be placed there) to the most. This optimises the solver by shaving off many of the branches in the backtracking 'tree'. The trade-off is that it doesn't illustrate how backtracking works as well as the other slower solver.