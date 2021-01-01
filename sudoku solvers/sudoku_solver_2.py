import numpy as np
import GUI

BLACK = (0, 0, 0)
RED = (255, 0, 0)

# This solves the sudoku by checking from the number with the least possible options to the most
# This optimises the solver by shaving off many of the branches in the backtracking 'tree'
# But, it doesn't illustrate how backtracking works as well as the other slower solver

def solve(board, start_time):
    find = find_min(board)
    
    if not find:
        return True
    else:
        (row, col), valid_values = find

    if GUI.check_for_quit():
        return False
    for num in valid_values:
        GUI.draw_timer(GUI.get_time(start_time))
        GUI.draw_tmp_num(row, col, num, BLACK)
        
        
        

        board[row, col] = num
            
        if solve(board, start_time):
            return True
            
        board[row, col] = 0

        GUI.draw_tmp_num(row, col, 0, RED)
    return False


def check_valid_num(board, check_n, row, col):
    for num in board[row, :]:
        if num == check_n:
            return False
    
    for num in board[:, col]:
        if num == check_n:
            return False

    #get the 9 boxes

    box_row = row // 3
    box_col = col // 3

    for num in np.nditer(board[box_row * 3 : box_row* 3 + 3, box_col * 3 : box_col * 3 + 3]):
        if num == check_n:
            return False
    
    return True


def find_min(board):
    zero_values = []
    for (row, col), num in np.ndenumerate(board):
        if num == 0:
            pos_values = set()
            for n in range(10):
            
                if check_valid_num(board, n, row, col):
                    pos_values.add(n)

            zero_values.append([(row, col), pos_values])
    if zero_values:
        return min(zero_values, key=lambda x: len(x[1]))
    
    else:
        return None

