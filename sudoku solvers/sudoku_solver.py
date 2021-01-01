import numpy as np
import GUI

RED = (255, 0, 0)
BLACK = (0, 0,0)

# This solves the sudoku by checking from top left to bottom right
# As compared to the other solver,  it better shows how backtracking works 
# But its slower

def solve(board, start_time):
    
    
    find = find_zero(board)
    if not find:
        return True
    else:
        row, col = find
    
    if GUI.check_for_quit():
        return False
    for num in range(1, 10):
        GUI.draw_timer(GUI.get_time(start_time))
        GUI.draw_tmp_num(row, col, num, BLACK)
        

        if check_valid_num(board, num, row, col):
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



def find_zero(board):
    for (row, col), num in np.ndenumerate(board):
        if num == 0:
            return row, col
    return None



