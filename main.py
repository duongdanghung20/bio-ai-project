import pygame
import neat
import sys
import random
import os
import math
from pygame.math import Vector2


WIN_WIDTH = 900
WIN_HEIGHT = 900

BIN_WIDTH = 30
BIN_HEIGHT = 30

NO_BINS_X = int(WIN_WIDTH / BIN_WIDTH)
NO_BINS_Y = int(WIN_HEIGHT / BIN_HEIGHT)

pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 30, True)

pygame.init()

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
win_rect = win.get_rect()
clock = pygame.time.Clock()

APPLE_IMG_NAME = 'apple.png'
SNAKE_HEAD_IMG_NAME = 'head.png'
SNAKE_BODY_IMG_NAME = 'body.png'
SNAKE_BODY_TURN_IMG_NAME = 'body_turn.png'
SNAKE_TAIL_IMG_NAME = 'tail.png'

IMGS_DIR = './imgs'
APPLE_IMG = pygame.image.load(os.path.join(IMGS_DIR, APPLE_IMG_NAME)).convert_alpha()
SNAKE_HEAD_IMG = pygame.image.load(os.path.join(IMGS_DIR, SNAKE_HEAD_IMG_NAME)).convert_alpha()
SNAKE_BODY_IMG = pygame.image.load(os.path.join(IMGS_DIR, SNAKE_BODY_IMG_NAME)).convert_alpha()
SNAKE_BODY_TURN_IMG = pygame.image.load(os.path.join(IMGS_DIR, SNAKE_BODY_TURN_IMG_NAME)).convert_alpha()
SNAKE_TAIL_IMG = pygame.image.load(os.path.join(IMGS_DIR, SNAKE_TAIL_IMG_NAME)).convert_alpha()

VEC_UP = Vector2(0, -1)
VEC_DOWN = Vector2(0, 1)
VEC_LEFT = Vector2(-1, 0)
VEC_RIGHT = Vector2(1, 0)

class Apple:
    IMG = pygame.transform.scale(APPLE_IMG, (BIN_WIDTH, BIN_HEIGHT))
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pos = Vector2(self.x, self.y)


    def draw(self, win):
        win.blit(self.IMG, (self.pos.x * BIN_WIDTH, self.pos.y * BIN_HEIGHT))


class Snake:
    HEAD_IMG = pygame.transform.scale(SNAKE_HEAD_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    BODY_IMG = pygame.transform.scale(SNAKE_BODY_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    BODY_TURN_IMG = pygame.transform.scale(SNAKE_BODY_TURN_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    TAIL_IMG = pygame.transform.scale(SNAKE_TAIL_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    def __init__(self):
        self.body = [Vector2(7, 10), Vector2(6, 10), Vector2(5, 10)]
        self.direction = Vector2(1, 0)
        self.new_block = False
        self.head_img = self.HEAD_IMG
        self.tail_img = self.TAIL_IMG
        self.mid_imgs = []
        self.moved = False

    def draw(self, win):
        self.update_head()
        self.update_body()
        self.update_tail()
        win.blit(self.head_img, (self.body[0].x * BIN_WIDTH, self.body[0].y * BIN_HEIGHT))
        for i, block in enumerate(self.body[1:-1]):
            win.blit(self.mid_imgs[i], (block.x * BIN_WIDTH, block.y * BIN_HEIGHT))
        win.blit(self.tail_img, (self.body[-1].x * BIN_WIDTH, self.body[-1].y * BIN_HEIGHT))

    def update_head(self):
        relative_direction = self.body[0] - self.body[1]
        if relative_direction == VEC_RIGHT:
            self.head_img = self.HEAD_IMG
        elif relative_direction == VEC_LEFT:
            self.head_img = pygame.transform.flip(self.HEAD_IMG, flip_x=True, flip_y=True).convert_alpha()
        elif relative_direction == VEC_DOWN:
            self.head_img = pygame.transform.rotate(self.HEAD_IMG, angle=-90).convert_alpha()
        elif relative_direction == VEC_UP:
            self.head_img = pygame.transform.rotate(self.HEAD_IMG, angle=90).convert_alpha()

    def update_body(self):
        self.mid_imgs = []
        for i in range(1, len(self.body) - 1):
            relative_direction_fw = self.body[i - 1] - self.body[i]
            relative_direction_bw = self.body[i] - self.body[i + 1]

            if (relative_direction_fw.y == 0 and relative_direction_bw.y == 0):
                body_img = self.BODY_IMG
            elif (relative_direction_fw.x == 0 and relative_direction_bw.x == 0):
                body_img = pygame.transform.rotate(self.BODY_IMG, angle=90).convert_alpha()
            elif (relative_direction_bw == VEC_UP and relative_direction_fw == VEC_RIGHT) or (relative_direction_bw == VEC_LEFT and relative_direction_fw == VEC_DOWN):
                body_img = pygame.transform.flip(self.BODY_TURN_IMG, flip_x=True, flip_y=True).convert_alpha()
            elif (relative_direction_bw == VEC_UP and relative_direction_fw == VEC_LEFT) or (relative_direction_bw == VEC_RIGHT and relative_direction_fw == VEC_DOWN):
                body_img = pygame.transform.rotate(self.BODY_TURN_IMG, angle=90).convert_alpha()
            elif (relative_direction_bw == VEC_DOWN and relative_direction_fw == VEC_RIGHT) or (relative_direction_bw == VEC_LEFT and relative_direction_fw == VEC_UP):
                body_img = pygame.transform.rotate(self.BODY_TURN_IMG, angle=-90).convert_alpha()
            elif (relative_direction_bw == VEC_DOWN and relative_direction_fw == VEC_LEFT) or (relative_direction_bw == VEC_RIGHT and relative_direction_fw == VEC_UP):
                body_img = self.BODY_TURN_IMG
            
            self.mid_imgs.append(body_img)

    def update_tail(self):
        relative_direction = self.body[-2] - self.body[-1]
        if relative_direction == VEC_RIGHT:
            self.tail_img = self.TAIL_IMG
        elif relative_direction == VEC_LEFT:
            self.tail_img = pygame.transform.flip(self.TAIL_IMG, flip_x=True, flip_y=True).convert_alpha()
        elif relative_direction == VEC_UP:
            self.tail_img = pygame.transform.rotate(self.TAIL_IMG, angle=90).convert_alpha()
        elif relative_direction == VEC_DOWN:
            self.tail_img = pygame.transform.rotate(self.TAIL_IMG, angle=-90).convert_alpha()


    def move(self):
        if not self.new_block:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy
            self.moved = True
        else:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy
            self.new_block = False
            self.moved = True

    def turn_right(self):
        self.direction = Vector2(1, 0)
        self.moved = False

    def turn_left(self):
        self.direction = Vector2(-1, 0)
        self.moved = False

    def turn_up(self):
        self.direction = Vector2(0, -1)
        self.moved = False

    def turn_down(self):
        self.direction = Vector2(0, 1)
        self.moved = False

    def add_block(self):
        self.new_block = True

class Main:
    def __init__(self):
        self.snake = Snake()
        self.apple = Apple(random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1))
        self.score = 0

    def update(self):
        self.snake.move()
        self.check_collision()
        self.check_fail()

    def draw(self, win):
        self.draw_bg(win)
        self.draw_score(win)
        self.snake.draw(win)
        self.apple.draw(win)

    def draw_bg(self, win):
        # Draw the background
        color_dark_green = True
        last_dark = True
        for i in range(NO_BINS_X):
            if last_dark:
                color_dark_green = False
                last_dark = False
            else:
                color_dark_green = True
                last_dark = True
            for j in range(NO_BINS_Y):
                bin = pygame.Rect(i * BIN_WIDTH, j * BIN_HEIGHT, BIN_WIDTH, BIN_HEIGHT)
                if color_dark_green:
                    pygame.draw.rect(win, (92, 181, 92), bin)
                    color_dark_green = False
                else:
                    pygame.draw.rect(win, (98, 209, 98), bin)
                    color_dark_green = True

    def draw_score(self, win):
        score_text = STAT_FONT.render("Score: " + str(self.score), 1, (255, 255, 255))
        win.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))

    def check_collision(self):
        if self.apple.pos == self.snake.body[0]:
            while True:
                new_apple_x , new_apple_y = (random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1))
                if (new_apple_x != self.apple.x or new_apple_y != self.apple.y) and (Vector2(new_apple_x, new_apple_y) not in self.snake.body):
                    break
            self.apple = Apple(new_apple_x, new_apple_y)
            self.snake.add_block()
            self.score += 1
            global GAME_SPEED
            GAME_SPEED -= 2
            pygame.time.set_timer(SCREEN_UPDATE, GAME_SPEED)

    def check_fail(self):
        self_hit = self.snake.body[0] in self.snake.body[1:]
        wall_hit = self.snake.body[0].x < 0 or self.snake.body[0].x >= NO_BINS_X or self.snake.body[0].y < 0 or self.snake.body[0].y >= NO_BINS_Y
        if self_hit or wall_hit:
            print("Score =", self.score)
            pygame.quit()
            sys.exit()

global GAME_SPEED
GAME_SPEED = 90

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, GAME_SPEED) # 150 in miliseconds: update the screen every milisecond

main_game = Main()

run = True
while run:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
            sys.exit()
        if event.type == SCREEN_UPDATE:
            main_game.update()

    keys = pygame.key.get_pressed()
    current_snake_direction = main_game.snake.direction
    if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and current_snake_direction != VEC_LEFT and main_game.snake.moved:
        main_game.snake.turn_right()
    if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and current_snake_direction != VEC_RIGHT and main_game.snake.moved:
        main_game.snake.turn_left()
    if (keys[pygame.K_UP] or keys[pygame.K_w]) and current_snake_direction != VEC_DOWN and main_game.snake.moved:
        main_game.snake.turn_up()
    if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and current_snake_direction != VEC_UP and main_game.snake.moved:
        main_game.snake.turn_down()

    main_game.draw(win)
    pygame.display.update()
    clock.tick(120) # Set Maximum FPS