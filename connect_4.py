import pygame
from pygame.locals import *
from pygame import freetype
import numpy as np
import statistics
import time
import itertools
import random
import math
pygame.init()
width = 770
height = 760
gameDisplay = pygame.display.set_mode((width,height))
pygame.display.set_caption('Pong')
clock = pygame.time.Clock()
WHITE = (255, 255, 255)
BLUE = (46, 61, 215 )
BLACK = (0, 0, 0)
YELLOW = (255,255,0)
RED = (255, 0 ,0)
ORANGE = (226, 145, 64)
GAME_FONT = pygame.font.SysFont("Sans Serif", 45)
ROW_COUNT = 6
COLUMN_COUNT = 7
board = np.zeros((ROW_COUNT,COLUMN_COUNT), dtype = int)
x_coords = [[110*x, 110*(x+1)] for x in range(COLUMN_COUNT)] 

colour_dict = {
    0 : BLACK,
    1 : YELLOW,
    2 : RED
}

def draw_display():
    gameDisplay.fill(BLACK)
    pygame.draw.rect(gameDisplay, BLUE, pygame.Rect(0, 100, width, 660)) 
    for x in range(7):
        for y in range(6):
            pygame.draw.circle(gameDisplay, colour_dict[board[y][x]], (55+ x * 110, 155 + y*110), 44)

def game_win(player):
    global game
    if player == 'draw':
        text = GAME_FONT.render('Draw!', True, ORANGE)
        print('draw')
    else:
        text = GAME_FONT.render(player +" wins!", True, ORANGE)
        print(player +'wins')
    text_rect = text.get_rect(center=(width/2, 50))
    gameDisplay.blit(text, text_rect)
    print('neither')
    game = False

def game_reset():
    global board, game, player_iterator, player_turn
    board = np.zeros((6,7)).astype(int)
    game = True
    player_iterator = itertools.cycle([1,2])
    player_turn = next(player_iterator)

def is_valid_location(column, board):
    if board[0][column] == 0:
        return True
    else:
        return False

def add_board(column):
    global player_turn, board
    if is_valid_location(column, board):#check if turn went through
        for i, row in enumerate(board):
            if row[column] == 0:
                row[column] = player_turn
                draw_display()
                pygame.display.update()
            
                if i != 5: # is there a better way to do this
                    if board[i+1][column] == 0:
                        row[column] = 0
            else:
                break
            
            time.sleep(0.04)
        check_win(board, player_turn, False)
        player_turn = next(player_iterator)

def add_sim_board(sim_board, column, turn_number):
    for i, row in enumerate(sim_board):
        if row[column] == 0 :
            row[column] = turn_number
            if i != 5:
                if sim_board[i+1][column] ==0:
                    row[column] = 0
    
        else:
            break


def check_win(board, turn_number, sim): #or draw

    if len(get_valid_locations(board)) == 0 and not sim:
        game_win('draw')
        return 
    for r in range(ROW_COUNT):
        row_list = list(board[r])
        for c in range(COLUMN_COUNT-3):
            window = row_list[c:c+4]
            if window.count(turn_number) == 4:
                
                return confirm_win(turn_number, sim)

    #vertical
    for c in range(COLUMN_COUNT):
        col_list = [x for x in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_list[r:r+4]
            if window.count(turn_number) == 4:
                
                return confirm_win(turn_number, sim)

    #negative gradient diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(4)]
            if window.count(turn_number) == 4:
                
                return confirm_win(turn_number, sim)
    #positive slope diagonal
    for r in range(ROW_COUNT-3):
        for c in range(3, COLUMN_COUNT):
            window = [board[r+i][c-i] for i in range(4)]
            if window.count(turn_number) == 4:
                
                return confirm_win(turn_number, sim)

def confirm_win(turn_number, sim):
    if not sim:
        if turn_number == 1:
            game_win('Yellow')
        elif turn_number == 2:
            game_win('Red')
    elif sim:
        return True

    


#Minimax algorithm

def eval_window(window, turn_number):
    if turn_number == 1:
        opp_number = 2
    elif turn_number == 2:
        opp_number = 1
    score = 0

    if window.count(turn_number) == 4:            
        score += 100 
    
                
    elif window.count(turn_number) == 3 and window.count(0) == 1:
        score += 8

    elif window.count(turn_number) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opp_number) == 3 and window.count(0) == 1:
        score -= 5

    return score
    
def score_position(sim_board ,turn_number):
    score = 0
    # draw state
    if len(get_valid_locations(sim_board)) == 0:
        score = 0
        return score
    #horizontal
    center_list = [i for i in sim_board[:, COLUMN_COUNT//2]]
    center_count = center_list.count(turn_number)
    score += center_count * 3

    
    for r in range(ROW_COUNT):
        row_list = list(sim_board[r])
        for c in range(COLUMN_COUNT-3):
            window = row_list[c:c+4]
            score += eval_window(window, turn_number)

    #vertical
    for c in range(COLUMN_COUNT):
        col_list = [x for x in list(sim_board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_list[r:r+4]
            score += eval_window(window, turn_number)

    #negative gradient diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [sim_board[r+i][c+i] for i in range(4)]
            score += eval_window(window, turn_number)
    #positive slope diagonal
    for r in range(ROW_COUNT-3):
        for c in range(3, COLUMN_COUNT):
            window = [sim_board[r+i][c-i] for i in range(4)]
            score += eval_window(window, turn_number)
    return score

def is_terminal_node(sim_board):
    return check_win(sim_board, 1, True) or check_win(sim_board, 2, True) or len(get_valid_locations(sim_board)) == 0

def minimax(board, depth, alpha, beta, maximising_player):
    turn_number = player_turn
    if turn_number == 1:
        opp_number = 2
    elif turn_number == 2:
        opp_number = 1
    valid_locations = get_valid_locations(board)
    
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            
            if check_win(board, turn_number, True):
                return (None, 1000000)
            elif check_win(board, opp_number, True):
            
                return (None, -1000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, turn_number))

    if maximising_player:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            sim_board = board.copy()
            add_sim_board(sim_board, col, turn_number) 
            new_score = minimax(sim_board, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            sim_board = board.copy()
            add_sim_board(sim_board, col, opp_number) # check node???????????????
            new_score = minimax(sim_board, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(col,board):
            valid_locations.append(col)
    return valid_locations

crashed = False
game = True
player_iterator = itertools.cycle([1,2])
player_turn = next(player_iterator)

#customisable --> u can choose whether u want player 1/2 to be human or AI
player_AI = {
    1 : False,
    2 : True
}
#game loop
while not crashed:
    mos_x, mos_y = pygame.mouse.get_pos()
    if game:
        draw_display()
    for i, x in enumerate(x_coords):
        if mos_x >= x[0] and mos_x <= x[1] and game:
            pygame.draw.circle(gameDisplay, colour_dict[player_turn], (statistics.mean(x), 50), 44)
            manual_column = i
            break
    
    if player_AI[player_turn] and game:
        depth = 5 #customisable, depending on how far you want the AI to look 
        pick_col, minimax_score = minimax(board, depth, -math.inf, math.inf, True)
        
        add_board(pick_col)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        elif event.type == pygame.MOUSEBUTTONDOWN and game and not player_AI[player_turn]:
            add_board(manual_column)
        elif event.type == pygame.KEYDOWN:
            if event.key == K_SPACE:
                game_reset()
    
    pygame.display.update()
    clock.tick(60)