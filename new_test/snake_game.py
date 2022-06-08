import random
import numpy as np
import time
import math
import pygame

def display_snake(snake_position, display):
    for position in snake_position:
        pygame.draw.rect(display, (255, 0, 0), pygame.Rect(position[0], position[1], 10, 10))

def display_apple(apple_position, display):
    pygame.draw.rect(display, (0, 255, 0), pygame.Rect(apple_position[0], apple_position[1], 10, 10))

def starting_position():
    snake_start = [100, 100]
    snake_position = [[100, 100], [90, 100], [80, 100]]
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score = 0

    return snake_start, snake_position, apple_position, score

def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position) - np.array(snake_position[0]))

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50)]
    score += 1
    return apple_position, score

def generate_snake(snake_start, snake_position, apple_position, button_direction, score):
    if button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10

    if snake_start == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0, list(snake_start))

    else:
        snake_position.insert(0, list(snake_start))
        snake_position.pop()
    
    return 
