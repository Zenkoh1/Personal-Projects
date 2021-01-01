import numpy as np
import pygame
from pygame.locals import *
import time
from boards import board
import sudoku_solver_2 as sdk2
import sudoku_solver as sdk
from copy import deepcopy
pygame.init()
pygame.display.set_caption('Sudoku Solver')
SQUARE_SIZE = 80
SIDEBAR = 200
width = 80 * 9 + 5 * 4 + 2 * 6 + SIDEBAR
height = 80 * 9 + 5 * 4 + 2 * 6
#Colour constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (233, 217, 137)

#Dropdown colours
DROPDOWN_INACTIVE = (100, 80, 255)
DROPDOWN_ACTIVE = (100, 200, 255)
DROPDOWN_LIST_INACTIVE = (255, 100, 100)
DROPDOWN_LIST_ACTIVE = (255, 150, 150)

#Button colours
BUTTON_INACTIVE = (0, 153, 0)
BUTTON_ACTIVE = (0, 220, 0)

#Some math to get the coords of the squares
COORDS = [x * 80 + (x+1) * 2 + ((x + 3) // 3) * 3 + 40 for x in range(10)] 



class DropDown:

    def __init__(self, color_menu, color_option, x, y, w, h, font, main, options):
        self.color_menu = color_menu
        self.color_option = color_option
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.main = main
        self.options = options
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1
    
    def draw(self, surf):
        pygame.draw.rect(surf, self.color_menu[self.menu_active], self.rect, 0)
        msg = self.font.render(self.main, 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center = self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.options):
                rect = self.rect.copy()
                rect.y += (i+1) * self.rect.height
                pygame.draw.rect(surf, self.color_option[1 if i == self.active_option else 0], rect, 0)
                msg = self.font.render(text, 1, (0, 0, 0))
                surf.blit(msg, msg.get_rect(center = rect.center))

    def update(self, event_list, mos_pos):
        self.menu_active = self.rect.collidepoint(mos_pos)
        
        self.active_option = -1
        for i in range(len(self.options)):
            rect = self.rect.copy()
            rect.y += (i+1) * self.rect.height
            if rect.collidepoint(mos_pos):
                self.active_option = i
                break

        if not self.menu_active and self.active_option == -1:
            self.draw_menu = False

        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    self.draw_menu = not self.draw_menu
                elif self.draw_menu and self.active_option >= 0:
                    self.draw_menu = False
                    return self.active_option
        return -1

class Button:
    def __init__(self, color, x, y, w, h, font, text):
        self.color = color
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.text = text
        self.active = False
        
    def draw(self, surf):
        pygame.draw.rect(surf, self.color[self.active], self.rect, 0)
        msg = self.font.render(self.text, 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center = self.rect.center))

    def update(self, event_list, mos_pos):

        self.active = self.rect.collidepoint(mos_pos)
        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.active:
                    return True
    
def create_game_board(board, board_timers, difficulty):
    global final_time
    for x in range(10):
        if x % 3 == 0:
            line_width = 5
        else:
            line_width = 2 

        pygame.draw.rect(sudoku_display, BLACK, pygame.Rect(x * 82 + ((x + 2) // 3) * 3, 0, line_width, height)) 

        pygame.draw.rect(sudoku_display, BLACK, pygame.Rect(0 , x * 82 + ((x + 2) // 3) * 3, width, line_width)) 
    
    draw_initial_num(board)
    draw_side_bar()
    sudoku_boards.draw(sudoku_display)
    solvers.draw(sudoku_display)
    solve_button.draw(sudoku_display)
    reset_button.draw(sudoku_display)
    if difficulty:
        draw_timer(board_timers[difficulty])
    else:
        draw_timer("00:00:00")



def draw_number(num, row, column, colour):
    myfont = pygame.font.SysFont('Arial', 40)
    text = myfont.render(str(num), False, colour)
    text_rect = text.get_rect(center=(COORDS[column], COORDS[row]))
    sudoku_display.blit(text, text_rect)

def clear_number(row, column):
    rect = pygame.Rect(0, 0, 40, 40)
    rect.center = (COORDS[column], COORDS[row])
    pygame.draw.rect(sudoku_display, WHITE , rect)

def draw_initial_num(board):
    for (row, column), num in np.ndenumerate(board):
        if num:
            draw_number(num, row, column, BLACK)
    
        
def draw_side_bar():
    pygame.draw.rect(sudoku_display, YELLOW, pygame.Rect(width - SIDEBAR, 0, SIDEBAR, height)) 

def draw_tmp_num(row, col, num, colour):
    #try/except statement to catch error when it pygame closes before solver loop is done
    try:
        pygame.event.pump()
    except pygame.error:
        pass
    clear_number(row, col)
    draw_number(num, row, col, colour)
    pygame.display.update()

def draw_timer(time):
    rect = pygame.Rect(width - SIDEBAR, 0, SIDEBAR, 90)
    pygame.draw.rect(sudoku_display, BLACK, rect)
    font = pygame.font.SysFont(None, 50)
    num = font.render(time, 1, (255, 0, 0))
    sudoku_display.blit(num, (width - SIDEBAR + 28, 28))

def get_time(start_time):
    present_time = pygame.time.get_ticks()
    time_elapsed = present_time - start_time
    ms = int(time_elapsed % 1000) // 10
    sec = int(time_elapsed // 1000 % 60)
    minute = (time_elapsed // 1000) // 60
    return "{:02d}:{:02d}:{:02d}".format(minute, sec, ms)
    
def check_for_quit():
    event_list = pygame.event.get()
    for event in event_list:
        if event.type == pygame.QUIT:
            pygame.display.quit()
            pygame.quit()
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == K_SPACE:
                print('lol')
                return True
    
    
sudoku_display = pygame.display.set_mode((width,height)) # so other files can access it

if __name__ == "__main__":
    # store copies so board can be resetted
    stored_board = deepcopy(board)
    board_timers = dict.fromkeys(board.keys(),"00:00:00")
    crashed = False
    sudoku_display.fill(WHITE)
    selected_board = np.zeros((9,9)) 

    #no options chosen yet
    chosen_solver = None
    difficulty = None
    
    sudoku_boards = DropDown(
        [DROPDOWN_INACTIVE, DROPDOWN_ACTIVE],
        [DROPDOWN_LIST_INACTIVE, DROPDOWN_LIST_ACTIVE],
        width - 180, 100, 150, 50, 
        pygame.font.SysFont(None, 30), 
        "Select Board", ["Easy", "Medium", "Hard", "Expert"])
    
    solvers = DropDown(
        [DROPDOWN_INACTIVE, DROPDOWN_ACTIVE],
        [DROPDOWN_LIST_INACTIVE, DROPDOWN_LIST_ACTIVE],
        width - 180, 400, 150, 50, 
        pygame.font.SysFont(None, 30), 
        "Select Solver", ["Solver #1", "Solver #2"])

    solve_button = Button(
        [BUTTON_INACTIVE, BUTTON_ACTIVE],
        width - 180, 600, 150, 50,
        pygame.font.SysFont(None, 30), 
        "Solve")
    
    reset_button = Button(
        [BUTTON_INACTIVE, BUTTON_ACTIVE],
        width - 180, 680, 150, 50,
        pygame.font.SysFont(None, 30), 
        "Reset Board")
    
    create_game_board(selected_board, board_timers, difficulty)

    while not crashed:
        event_list = pygame.event.get()
        mos_pos = pygame.mouse.get_pos()
        for event in event_list:
            if event.type == pygame.QUIT:
                crashed = True
            
        
        selected_board_option = sudoku_boards.update(event_list, mos_pos)
        selected_solver = solvers.update(event_list, mos_pos)
        if solve_button.update(event_list, mos_pos):
            if chosen_solver and np.any(selected_board) and not np.all(selected_board):
                start_time = pygame.time.get_ticks()
                #solve_button.text = 'Cancel'
                if chosen_solver == 'sdk':
                    sdk.solve(selected_board, start_time)
                elif chosen_solver == 'sdk2':
                    sdk2.solve(selected_board, start_time)
                final_time = get_time(start_time)
                board_timers[difficulty] = final_time
        if reset_button.update(event_list, mos_pos):
            sudoku_display.fill(WHITE)
            if difficulty:
                board_timers[difficulty] = "00:00:00"
                board[difficulty] = stored_board[difficulty].copy()
                selected_board = board[difficulty]
            else:
                selected_board = np.zeros((9, 9))


        if selected_board_option >= 0:
            sudoku_boards.main = sudoku_boards.options[selected_board_option]
            sudoku_display.fill(WHITE)
            if selected_board_option == 0:
                difficulty = 'Easy'

            elif selected_board_option == 1:
                difficulty = 'Medium'
                
            elif selected_board_option == 2:
                difficulty = 'Hard'
                
            elif selected_board_option == 3:
                difficulty = 'Expert'
            
            selected_board = board[difficulty]
                
            
        if selected_solver >= 0:
            solvers.main = solvers.options[selected_solver]
            sudoku_display.fill(WHITE)
            
            if selected_solver == 0:
                chosen_solver = 'sdk'
            elif selected_solver == 1:
                chosen_solver = 'sdk2'

        create_game_board(selected_board, board_timers, difficulty)

        
            
        pygame.display.update()
