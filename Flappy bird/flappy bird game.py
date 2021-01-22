import os
import random
from math import sin
from itertools import cycle
import pygame

pygame.init()

game_dir = os.path.dirname(os.path.abspath(__file__))

width = 448
height = 700
ZOOM_SCALE = 683/512
GRAVITY = 0.45
FLOOR_Y = height - 127
game_font_dir = os.path.join(game_dir, 'assets/04B_19.ttf')
GAME_FONT = pygame.font.Font(game_font_dir, 35)
game_display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Flappy Bird')
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BEIGE = (222, 216, 149)
BLACK = (0, 0, 0)
BROWN = (84, 56, 71)
ORANGE = (252,120,88)

# Sound settings
sound_list = cycle((True, False))
sound_on = next(sound_list)

# Set icon
icon_dir = os.path.join(game_dir,'assets/icon.ico')
icon = pygame.image.load(icon_dir)
pygame.display.set_icon(icon)

# Other needed images

bg_dir = os.path.join(game_dir, 'assets/sprites/background-day.png')
bg_surface = pygame.image.load(bg_dir).convert()
bg_surface = pygame.transform.scale(bg_surface, (width, height))

floor_dir = os.path.join(game_dir, 'assets/sprites/base.png')
floor_surface = pygame.image.load(floor_dir).convert()
floor_surface = pygame.transform.rotozoom(floor_surface, 0, ZOOM_SCALE)


bird_downflap_dir = os.path.join(game_dir, 'assets/sprites/bluebird-downflap.png')
bird_downflap_surf = pygame.image.load(bird_downflap_dir).convert_alpha()
bird_downflap_surf = pygame.transform.rotozoom(bird_downflap_surf, 0, ZOOM_SCALE)

bird_midflap_dir = os.path.join(game_dir, 'assets/sprites/bluebird-midflap.png')  
bird_midflap_surf = pygame.image.load(bird_midflap_dir).convert_alpha()
bird_midflap_surf = pygame.transform.rotozoom(bird_midflap_surf, 0, ZOOM_SCALE)

bird_upflap_dir = os.path.join(game_dir, 'assets/sprites/bluebird-upflap.png')  
bird_upflap_surf = pygame.image.load(bird_upflap_dir).convert_alpha()
bird_upflap_surf = pygame.transform.rotozoom(bird_upflap_surf, 0, ZOOM_SCALE)

BIRD_FLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRD_FLAP, 200)

pipe_dir = os.path.join(game_dir, 'assets/sprites/pipe-green.png')
pipe_surface = pygame.image.load(pipe_dir).convert_alpha()
pipe_surface = pygame.transform.rotozoom(pipe_surface, 0, ZOOM_SCALE)
flipped_pipe_surface = pygame.transform.flip(pipe_surface, False, True)
pipe_list = []

SPAWN_PIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWN_PIPE, 1700)
pipe_heights = [x for x in range(250, 500, 60)]


game_over_dir = os.path.join(game_dir, 'assets/sprites/message.png')
game_over_surf = pygame.image.load(game_over_dir).convert_alpha()
game_over_surf = pygame.transform.rotozoom(game_over_surf, 0, ZOOM_SCALE)
game_over_rect = game_over_surf.get_rect(center = (width/2, height/2))

# Sound stuff
flap_sound_dir = os.path.join(game_dir, 'assets/audio/wing.wav')
flap_sound = pygame.mixer.Sound(flap_sound_dir)

death_sound_dir = os.path.join(game_dir, 'assets/audio/hit.wav')
death_sound = pygame.mixer.Sound(death_sound_dir)

score_sound_dir = os.path.join(game_dir, 'assets/audio/point.wav')
score_sound = pygame.mixer.Sound(score_sound_dir)

class Bird:
    def __init__(self, x, y):
        self.frames = cycle((bird_downflap_surf, bird_midflap_surf, bird_upflap_surf))
        self.surface = next(self.frames)
        self.start_y = y # temp variable
        self.rect = self.surface.get_rect(center= (x, y))
        self.movement = 0

def create_pipe():
    rand_pipe_height = random.choice(pipe_heights)
    bottom_pipe = pipe_surface.get_rect(midtop = (width + 100, rand_pipe_height))
    top_pipe = pipe_surface.get_rect(midbottom = (width + 100, rand_pipe_height - 130))
    return [bottom_pipe, top_pipe, True] # last true is for whether it has been counted in the score

def move_pipes(pipes):
    for i ,(bottom_pipe, top_pipe, _) in enumerate(pipes):
        bottom_pipe.centerx -= 3
        top_pipe.centerx -= 3
        if bottom_pipe.right < 0:
            pipes.pop(i)

    return pipes

def draw_pipes(pipes):
    for bottom_pipe, top_pipe, _ in pipes:
        game_display.blit(pipe_surface, bottom_pipe)
        game_display.blit(flipped_pipe_surface, top_pipe)

def write(msg, colour, x, y):
    msg_surface = GAME_FONT.render(msg, True, colour)
    msg_rect = msg_surface.get_rect(center = (x, y))
    game_display.blit(msg_surface, msg_rect)

def draw_end_score():
    outline_rect = pygame.Rect(0, 0, 179, 244)
    outline_rect.center = (width/2, height/2)
    pygame.draw.rect(game_display, BROWN, outline_rect, border_radius= 10)

    scoreboard_rect = pygame.Rect(0, 0, 175, 240)
    scoreboard_rect.center = (width/2, height/2)
    pygame.draw.rect(game_display, BEIGE, scoreboard_rect, border_radius= 10)

    write('Score', ORANGE, width/2, height/2 - 70)
    write(str(score), ORANGE, width/2, height/2 - 30)
    write('Highscore', ORANGE, width/2, height/2  + 30)
    with open(os.path.join(game_dir, 'info/high_score.txt'), 'r') as f:
        high_score = f.readline()
        
        write(high_score, ORANGE, width/2, height/2 + 70)



def draw_display(pipes):
    game_display.blit(bg_surface, (0, 0))
    draw_pipes(pipes)
    game_display.blit(floor_surface, (floor_x_pos, FLOOR_Y))
    game_display.blit(floor_surface, (floor_x_pos + width, FLOOR_Y))
 
    for bird in birds:
        game_display.blit(rotate_bird(bird), bird.rect)
    
    if game_state == 'Playing':
        write(str(score), WHITE, 230, 70)
    elif game_state == 'Pre-game':
        game_display.blit(game_over_surf, game_over_rect)
    
    elif game_state == 'Dead':
        draw_end_score()

def check_collision(pipes):
    for bird in birds:
        for bottom_pipe, top_pipe, _ in pipes:
            

            if bird.rect.colliderect(bottom_pipe) or bird.rect.colliderect(top_pipe):
                
                return True

        if bird.rect.bottom >= height - 127:
            return True

    return False

def rotate_bird(bird):
    new_bird = pygame.transform.rotate(bird.surface, -bird.movement * 3)
    return new_bird

def bird_animation(bird):
    new_bird_surf = next(bird.frames)
    new_bird_rect = new_bird_surf.get_rect(center = (100, bird.rect.centery))
    return new_bird_surf, new_bird_rect


def reset_game():
    global game_state, bird_movement, score, float_animation_num
    print('yes')
    if game_state == 'Dead':
        game_state = 'Pre-game'
        pipe_list.clear()
        for bird in birds:
            bird.rect.center = (100, height/2 - 65)
            bird.movement = 0   
        score = 0
        float_animation_num = 0
    elif game_state == 'Pre-game':
        game_state = 'Playing'
        
    
    

def check_pipe_score():
    global score
 
    for i ,(bottom_pipe, _ , can_score) in enumerate(pipe_list):
        if 95 < bottom_pipe.centerx < 105 and can_score:
            score += 1
            pipe_list[i][2] = False
            if sound_on:
                score_sound.play()
        





def main():

    global game_state, float_animation_num, pipe_list, birds, floor_x_pos, score, sound_on
    crashed = False

    floor_x_pos = 0
    score = 0
    game_state = 'Pre-game'
    float_animation_num = 0

    # List for flexibility; in case I want to add more birds next time
    birds = [Bird(100, height/2 -65)]
    
    

    while not crashed:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            
            # Keyboard 
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game_state == 'Playing':
                        for bird in birds:
                            bird.movement = -7
                            if sound_on:
                                flap_sound.play()
                    else:
                        reset_game()

                elif event.key == pygame.K_m:
                    sound_on = next(sound_list)

            # Touchscreen support
            elif event.type == pygame.FINGERDOWN:
                if game_state == 'Playing':
                    for bird in birds:
                        bird.movement = -7
                        if sound_on:
                            flap_sound.play()
                else:
                    reset_game()
                
            elif event.type == SPAWN_PIPE and game_state == 'Playing':
                pipe_list.append(create_pipe())

            elif event.type == BIRD_FLAP and game_state != 'Dead':
                for bird in birds:
                    bird.surface, bird.rect = bird_animation(bird)
        
        
        
        
        # Game logic
        if game_state == 'Playing': 
            # Pipe
            pipe_list = move_pipes(pipe_list)

            
            if check_collision(pipe_list):
                game_state = 'Dead'
                with open(os.path.join(game_dir, 'info/high_score.txt'), 'r+') as f:
                    high_score = int(f.readline())
                    if score > high_score:   
                        f.seek(0)
                        f.write(str(score))
                        f.truncate()

                if sound_on:
                    death_sound.play()
                

            check_pipe_score()

        
        
        # Animation Configs

        if game_state == 'Pre-game':
            for bird in birds:
                bird.rect.centery = bird.start_y + 10 * sin(float_animation_num) 
                float_animation_num += 0.05

        else:
            for bird in birds:
                if bird.rect.centery <= -50:
                    bird.rect.centery = -50

                if bird.rect.bottom <= FLOOR_Y:
                    bird.movement += GRAVITY
                    bird.rect.centery += bird.movement

        
        if game_state != 'Dead':
            # Floor
            floor_x_pos -= 1
            if floor_x_pos == -width:
                floor_x_pos = 0

            

        
        
        draw_display(pipe_list)
        pygame.display.update()
        clock.tick(60)

if __name__ == '__main__':
    main()