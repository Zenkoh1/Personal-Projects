import pygame
from pygame.locals import *
from pygame import freetype
from math import sin, cos, tan, pi
from numpy import random
from copy import deepcopy
pygame.init()
width = 800
height = 480
game_display = pygame.display.set_mode((width,height))
pygame.display.set_caption('Pong')
clock = pygame.time.Clock()
WHITE = (255, 255, 255)
GAME_FONT = pygame.freetype.SysFont("Sans Serif", 45)
#draw lines

p1_score = 0
p2_score = 0

def draw_display():
    game_display.fill((0,0,0))
    for x in range(int(height/15)):
        pygame.draw.rect(game_display, WHITE, pygame.Rect(width/2 - 10/2, 5 + 20*x, 4, 15)) 
    GAME_FONT.render_to(game_display, (width/2 - 50, 20), str(p1_score),  WHITE)
    GAME_FONT.render_to(game_display, (width/2 + 22 , 20), str(p2_score), WHITE)

#The paddles
class Player:
    def __init__(self, xPos, side, up_key, down_key):
        self.xPos = xPos
        self.yPos = height/2 - 107/2
        self.width = 14
        self.height = 107
        self.middle = self.yPos + self.height/2
        if side == 'left':
            self.x_contact_point = self.xPos + self.width
        elif side == 'right':
            self.x_contact_point = self.xPos
        self.y_contact_point = [self.yPos, self.yPos + self.height]
        self.speed = 8
        self.AI = False
        self.up_key = up_key
        self.down_key = down_key

    def draw(self):
        pygame.draw.rect(game_display, WHITE, pygame.Rect(self.xPos, self.yPos, self.width, self.height)) 
    
    def move_up(self):
        if self.yPos >= 0:
            self.yPos -= self.speed
        self.y_contact_point = [self.yPos, self.yPos + self.height]
        self.middle = self.yPos + self.height/2

    def move_down(self):
        if self.yPos <= height-self.height:
            self.yPos += self.speed
        self.y_contact_point = [self.yPos, self.yPos + self.height]
        self.middle = self.yPos + self.height/2


    def AI_control(self):
        if (self.xPos > width/2 and ball.xSpeed > 0) or (self.xPos < width/2 and ball.xSpeed <0):
            if self.middle > ai_contact + self.speed:
                self.move_up()
            elif self.middle < ai_contact - self.speed:
                self.move_down()
        
    def paddle_bounce(self):
        if ball.x_contact_point < self.x_contact_point and self.xPos <width/2:
            bounce = 'p1'
        elif ball.x_contact_point >self.x_contact_point and self.xPos > width/2:
            bounce = 'p2'
        else:
            bounce = None
        if bounce and ball.y_contact_point < self.y_contact_point[1] and ball.y_contact_point > self.y_contact_point[0]:
            bounce_value = ball.y_contact_point - self.middle
            normalised_bounce_value = bounce_value /(self.height/2)

            # affecting speed?
            ball.speed = ball_speed + abs(normalised_bounce_value*2)

            #direction
            bounce_angle = normalised_bounce_value*(5*pi/12) # change the angle here
            ball.xSpeed = ball.speed*cos(bounce_angle)
            if bounce == 'p2': # cos p1 doesnt need a negative value
                ball.xSpeed = - ball.xSpeed
            ball.ySpeed = ball.speed*sin(bounce_angle)
            return True

        

class Ball:
    def __init__(self, angle, speed):
        self.xPos = width/2 - 10/2
        self.yPos = height/2 - 10/2
        self.length = 10
        self.speed = speed
        self.xSpeed = speed*sin(angle)
        self.ySpeed = speed*cos(angle)
        self.y_contact_point = self.yPos + self.length/2
        self.x_contact_point = self.xPos if self.xPos <width/2 else self.xPos + self.length
    
    def draw(self):
        pygame.draw.rect(game_display, WHITE, pygame.Rect(self.xPos, self.yPos, self.length, self.length))

    def move(self):
        self.xPos += self.xSpeed
        self.yPos += self.ySpeed 
        self.y_contact_point = self.yPos + self.length/2
        self.x_contact_point = self.xPos if self.xPos <width/2 else self.xPos + self.length

    def bounce_back(self): #for wall
        if self.yPos <= self.ySpeed and self.ySpeed <= 0:
            self.ySpeed = -self.ySpeed
        elif self.yPos >= height - self.ySpeed - self.length and self.ySpeed >= 0:
            self.ySpeed = -self.ySpeed
    
    def check_lose(self):
        if self.xPos <= 0:
            return 'p1'
        elif self.xPos + self.length >= width:
            return 'p2'
      


def uniform_two(a1, a2, b1, b2):
    # Calc weight for each range
    delta_a = a2 - a1
    delta_b = b2 - b1
    global player_lost
    if random.rand() < delta_a / (delta_a + delta_b):
        player_lost = 'p1'
        return random.uniform(a1, a2)
        
    else:
        player_lost = 'p2'
        return random.uniform(b1, b2)
        

def physics_sim(paddle_player):
    clone = deepcopy(ball)

    clone.move()
    while (clone.x_contact_point > p1.x_contact_point and paddle_player == 'p2') or (clone.x_contact_point < p2.x_contact_point and paddle_player == 'p1'):
        clone.bounce_back()
        clone.move()
    return clone.y_contact_point
    





    

p1 = Player(30, 'left', K_w, K_s)
p1.draw()

p2 = Player(756, 'right', K_UP, K_DOWN)
p2.draw()

players = [p1, p2]

#customisable stuff 
p1.AI = False
p2.AI = True
ball_speed = 10
p1.speed = 8
p2.speed = 8 # paddle speeds


ball = Ball(uniform_two(pi/4, 3*pi/4, 5*pi/4, 7*pi/4), ball_speed)
ball.draw()
crashed = False
game = True
ai_contact = physics_sim(player_lost)

# game loop
while not crashed:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        elif event.type == pygame.KEYDOWN:
            if event.key == K_SPACE and not game:
                game = True
                ball = Ball(angle, ball_speed)
                ai_contact = physics_sim(player_lost)

    
    
    draw_display()
    p1.draw()
    p2.draw()
        
    ball.draw()
    pressed = pygame.key.get_pressed()

    for player in players:
        if not player.AI:
            if pressed[player.up_key]:
                player.move_up()
            elif pressed[player.down_key]:
                player.move_down()
        elif player.AI:
            player.AI_control()
    

    if game:
        ball.bounce_back()
        if ball.x_contact_point >= p1.x_contact_point - ball.speed and ball.x_contact_point <= p2.x_contact_point + ball.speed: # check if its in the playing field

            if p1.paddle_bounce():
                ai_contact = physics_sim('p1')
            elif p2.paddle_bounce():
                ai_contact = physics_sim('p2')
                
           
        ball.move()
        if ball.check_lose() == 'p1':
            p2_score += 1
            angle = random.uniform(pi/4, 3*pi/4)
            player_lost = 'p1'
            game = False
        elif ball.check_lose() == 'p2':
            p1_score += 1
            angle = random.uniform(5*pi/4, 7*pi/4)
            player_lost = 'p2'
            game = False
    

    pygame.display.update()
    clock.tick(60)

