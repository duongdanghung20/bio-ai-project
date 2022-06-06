import pygame
import os
import random
import numpy as np
import math
from pygame.math import Vector2

WIN_WIDTH = 900
WIN_HEIGHT = 900

BIN_WIDTH = 90
BIN_HEIGHT = 90

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

#apple
IMG = pygame.transform.scale(APPLE_IMG, (BIN_WIDTH, BIN_HEIGHT))
x = random.randint(0, NO_BINS_X - 1)
y = random.randint(0, NO_BINS_Y - 1)
pos = Vector2(x, y)
eaten = False
global apple_pos
apple_pos = pos

#snake
HEAD_IMG = pygame.transform.scale(SNAKE_HEAD_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
BODY_IMG = pygame.transform.scale(SNAKE_BODY_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
BODY_TURN_IMG = pygame.transform.scale(SNAKE_BODY_TURN_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
TAIL_IMG = pygame.transform.scale(SNAKE_TAIL_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
INIT_HEAD_X = round(NO_BINS_X / 5)
INIT_HEAD_Y = round(NO_BINS_Y / 2)
body = [Vector2(INIT_HEAD_X, INIT_HEAD_Y), Vector2(INIT_HEAD_X - 1, INIT_HEAD_Y), Vector2(INIT_HEAD_X - 2, INIT_HEAD_Y)]
head = body[0]
direction = Vector2(1, 0)
new_block = False
head_img = HEAD_IMG
tail_img = TAIL_IMG
mid_imgs = []
moved = False
count_turn = 0
total_turn = 0

def is_outside(vec):
    if vec[0] < 0 or vec[0] >= NO_BINS_X or vec[1] < 0 or vec[1] >= NO_BINS_Y:
        return True
    return False

def collision_with_boundaries(snake_start):
    wall_hit = is_outside(snake_start)
    if wall_hit:
        return True
    return False


def collision_with_self(snake_start, snake_position):
    self_hit = snake_start in snake_position[1:]
    if self_hit:
        return True
    return False

def blocked_directions(snake_position):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])

    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    is_front_blocked = is_direction_blocked(snake_position, current_direction_vector)
    is_left_blocked = is_direction_blocked(snake_position, left_direction_vector)
    is_right_blocked = is_direction_blocked(snake_position, right_direction_vector)

    return current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked

def is_direction_blocked(snake_position, current_direction_vector):
    next_step = snake_position[0] + current_direction_vector
    snake_start = snake_position[0]
    if collision_with_boundaries(next_step) == 1 or collision_with_self(next_step.tolist(), snake_position) == 1:
        return 1
    else:
        return 0

def generate_random_direction(snake_position, angle_with_apple):
    direction = 0
    if angle_with_apple > 0:
        direction = 1
    elif angle_with_apple < 0:
        direction = -1
    else:
        direction = 0

    return direction_vector(snake_position, angle_with_apple, direction)


def direction_vector(snake_position, angle_with_apple, direction):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    new_direction = current_direction_vector

    if direction == -1:
        new_direction = left_direction_vector
    if direction == 1:
        new_direction = right_direction_vector

    button_direction = generate_button_direction(new_direction)

    return direction, button_direction


def generate_button_direction(new_direction):
    button_direction = 0
    if new_direction.tolist() == [10, 0]:
        button_direction = 1
    elif new_direction.tolist() == [-10, 0]:
        button_direction = 0
    elif new_direction.tolist() == [0, 10]:
        button_direction = 2
    else:
        button_direction = 3

    return button_direction


def angle_with_apple(snake_position, apple_position):
    apple_direction_vector = np.array(apple_position) - np.array(snake_position[0])
    snake_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])

    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10

    apple_direction_vector_normalized = apple_direction_vector / norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector / norm_of_snake_direction_vector
    angle = math.atan2(
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[1],
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[0]) / math.pi
    return angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized

def starting_positions():
    snake_start = [round(NO_BINS_X / 5), round(NO_BINS_Y / 2)]
    snake_position = [[round(NO_BINS_X / 5), 100], [round(NO_BINS_X / 5)-10, round(NO_BINS_Y / 2)], [round(NO_BINS_X / 5)-20, round(NO_BINS_Y / 2)]]
    apple_position = [random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1)]
    score = 0

    return snake_start, snake_position, apple_position, score

def display_apple(apple_position, display):
    win.blit(IMG, (pos.x * BIN_WIDTH, pos.y * BIN_HEIGHT))

def display_snake(snake_position, display):
    update_head()
    update_body()
    update_tail()
    win.blit(head_img, (body[0].x * BIN_WIDTH, body[0].y * BIN_HEIGHT))
    for i, block in enumerate(body[1:-1]):
        win.blit(mid_imgs[i], (block.x * BIN_WIDTH, block.y * BIN_HEIGHT))
    win.blit(tail_img, (body[-1].x * BIN_WIDTH, body[-1].y * BIN_HEIGHT))

def update_head():
    relative_direction = body[0] - body[1]
    if relative_direction == VEC_RIGHT:
        head_img = HEAD_IMG
    elif relative_direction == VEC_LEFT:
        head_img = pygame.transform.flip(HEAD_IMG, flip_x=True, flip_y=True).convert_alpha()
    elif relative_direction == VEC_DOWN:
        head_img = pygame.transform.rotate(HEAD_IMG, angle=-90).convert_alpha()
    elif relative_direction == VEC_UP:
        head_img = pygame.transform.rotate(HEAD_IMG, angle=90).convert_alpha()

def update_body():
    mid_imgs = []
    for i in range(1, len(body) - 1):
        relative_direction_fw = body[i - 1] - body[i]
        relative_direction_bw = body[i] - body[i + 1]

        if (relative_direction_fw.y == 0 and relative_direction_bw.y == 0):
            body_img = BODY_IMG
        elif (x == 0 and relative_direction_bw.x == 0):
            body_img = pygame.transform.rotate(BODY_IMG, angle=90).convert_alpha()
        elif (relative_direction_bw == VEC_UP and relative_direction_fw == VEC_RIGHT) or (relative_direction_bw == VEC_LEFT and relative_direction_fw == VEC_DOWN):
            body_img = pygame.transform.flip(BODY_TURN_IMG, flip_x=True, flip_y=True).convert_alpha()
        elif (relative_direction_bw == VEC_UP and relative_direction_fw == VEC_LEFT) or (relative_direction_bw == VEC_RIGHT and relative_direction_fw == VEC_DOWN):
            body_img = pygame.transform.rotate(BODY_TURN_IMG, angle=90).convert_alpha()
        elif (relative_direction_bw == VEC_DOWN and relative_direction_fw == VEC_RIGHT) or (relative_direction_bw == VEC_LEFT and relative_direction_fw == VEC_UP):
            body_img = pygame.transform.rotate(BODY_TURN_IMG, angle=-90).convert_alpha()
        elif (relative_direction_bw == VEC_DOWN and relative_direction_fw == VEC_LEFT) or (relative_direction_bw == VEC_RIGHT and relative_direction_fw == VEC_UP):
            body_img = BODY_TURN_IMG
        
        self.mid_imgs.append(body_img)

def update_tail():
    relative_direction = body[-2] - body[-1]
    if relative_direction == VEC_RIGHT:
        tail_img = TAIL_IMG
    elif relative_direction == VEC_LEFT:
        tail_img = pygame.transform.flip(TAIL_IMG, flip_x=True, flip_y=True).convert_alpha()
    elif relative_direction == VEC_UP:
        tail_img = pygame.transform.rotate(TAIL_IMG, angle=90).convert_alpha()
    elif relative_direction == VEC_DOWN:
        tail_img = pygame.transform.rotate(TAIL_IMG, angle=-90).convert_alpha()


class Snake:
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
        if self.direction != VEC_LEFT and self.moved:
            self.direction = VEC_RIGHT
            self.moved = False
            self.count_turn += 1
            self.total_turn += 1

    def turn_left(self):
        if self.direction != VEC_RIGHT and self.moved:
            self.direction = VEC_LEFT
            self.moved = False
            self.count_turn += 1
            self.total_turn += 1

    def turn_up(self): 
        if self.direction != VEC_DOWN and self.moved:
            self.direction = VEC_UP
            self.moved = False
            self.count_turn += 1
            self.total_turn += 1

    def turn_down(self):
        if self.direction != VEC_UP and self.moved:
            self.direction = VEC_DOWN
            self.moved = False
            self.count_turn += 1
            self.total_turn += 1
    
    def ai_turn_left(self):
        if self.moved:
            self.direction = self.direction.rotate(-90)
            self.moved = False
            self.count_turn += 1
            self.total_turn += 1

    def ai_turn_right(self):
        if self.moved:
            self.direction = self.direction.rotate(90)
            self.moved = False
            self.count_turn += 1
            self.total_turn += 1

    def add_block(self):
        self.new_block = True
        self.count_turn = 0

    def sensor(self):
        global apple_pos
        self.tail_direction = self.body[-2] - self.body[-1]
        vec_list = [VEC_UP, VEC_DOWN, VEC_LEFT, VEC_RIGHT]
        direction_onehot = [self.direction == vec for vec in vec_list]
        direction_onehot = list(map(int, direction_onehot))
        tail_direction_onehot = [self.tail_direction == vec for vec in vec_list]
        tail_direction_onehot = list(map(int, tail_direction_onehot))
        sense = []
        sense_dir = [VEC_UP, VEC_UP + VEC_RIGHT, VEC_RIGHT, VEC_RIGHT + VEC_DOWN, VEC_DOWN, VEC_DOWN + VEC_LEFT, VEC_LEFT, VEC_LEFT + VEC_UP]
        for dir in sense_dir:
            temp = Vector2(self.body[0].x, self.body[0].y)
            dist_to_wall = 0
            has_apple = 0
            has_body = 0
            while not is_outside(temp):
                temp += dir
                dist_to_wall += 1
                if temp in self.body:
                    has_body = 1
                if temp == apple_pos:
                    has_apple = 1
            sense += [dist_to_wall, has_apple, has_body]
        return tuple(sense + direction_onehot + tail_direction_onehot)

def draw_bg(win):
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

def draw_window(win, snake, apples, score):
    draw_bg(win)

    for apple in apples:
        apple.draw(win)

    score_text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))

    snake.draw(win)
    pygame.display.update()

def play_game(snake_start, snake_position, apple_position, button_direction, score, display, clock):
    crashed = False
    while crashed is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        display.fill((255, 255, 255))

        display_apple(apple_position, display)
        display_snake(snake_position, display)

        snake_position, apple_position, score = generate_snake(snake_start, snake_position, apple_position,
                                                               button_direction, score)
        pygame.display.set_caption("SCORE: " + str(score))
        pygame.display.update()
        clock.tick(50000)

        return snake_position, apple_position, score

