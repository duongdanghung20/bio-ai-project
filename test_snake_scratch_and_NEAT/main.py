import pygame
import neat
import sys
import random
import os
import pickle
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from visualize import *
from genetic_algorithm import Population, crossover, mutate
from neural_network import feedforward_nn
from pygame.math import Vector2


NUMGEN = 30
POP_SIZE = 100
GENOME_LENGTH = 32 * 20 + 20 + 20 * 12 + 12 + 12 * 4 + 4
NUM_WEIGHTS = 32 * 20 + 20 * 12 + 12 * 4
MU = 10                     # Number of pairs parents chosen for crossover
weights_range = (-10, 10)
biases_range = (-10, 10)
weight_mut_prob = 0.8       # Prob that 1 weight gene will be mutated
bias_mut_prob = 0.7         # Prob that 1 bias gene will be mutated
weight_mut_rate = 0.5       # Proportion of the population will be weight-mutated
bias_mut_rate = 0.5         # Proportion of the population will be bias-mutated
cx_prob = 0.5               # Prob that 2 genes of 2 genomes will be swapped
cx_rate = 0.8               # Proportion of the population will be mated
weight_mut_std = 1.0
bias_mut_std = 1.0


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
VEC_DOWN = -VEC_UP
VEC_LEFT = Vector2(-1, 0)
VEC_RIGHT = -VEC_LEFT

def is_outside(vec):
    return vec.x < 0 or vec.x >= NO_BINS_X or vec.y < 0 or vec.y >= NO_BINS_Y

global apple_pos
global best_fitness
best_fitness = -math.inf
global max_fitness
max_fitness = 10000
global best_ge
best_ge = None

global best_gen_fitnesses
global avg_gen_fitnesses
global worst_gen_fitnesses
best_gen_fitnesses = []
avg_gen_fitnesses = []
worst_gen_fitnesses = []

global std_gen_fitnesses
std_gen_fitnesses = []


class Apple:
    IMG = pygame.transform.scale(APPLE_IMG, (BIN_WIDTH, BIN_HEIGHT))
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pos = Vector2(self.x, self.y)
        self.eaten = False

        global apple_pos
        apple_pos = self.pos


    def draw(self, win):
        win.blit(self.IMG, (self.pos.x * BIN_WIDTH, self.pos.y * BIN_HEIGHT))


class Snake:
    HEAD_IMG = pygame.transform.scale(SNAKE_HEAD_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    BODY_IMG = pygame.transform.scale(SNAKE_BODY_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    BODY_TURN_IMG = pygame.transform.scale(SNAKE_BODY_TURN_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    TAIL_IMG = pygame.transform.scale(SNAKE_TAIL_IMG, (BIN_WIDTH, BIN_HEIGHT)).convert_alpha()
    INIT_HEAD_X = round(NO_BINS_X / 5)
    INIT_HEAD_Y = round(NO_BINS_Y / 2)
    def __init__(self):
        self.body = [Vector2(self.INIT_HEAD_X, self.INIT_HEAD_Y), Vector2(self.INIT_HEAD_X - 1, self.INIT_HEAD_Y), Vector2(self.INIT_HEAD_X - 2, self.INIT_HEAD_Y)]
        self.head = self.body[0]
        self.direction = Vector2(1, 0)
        self.new_block = False
        self.head_img = self.HEAD_IMG
        self.tail_img = self.TAIL_IMG
        self.mid_imgs = []
        self.moved = False
        self.count_turn = 0
        self.total_turn = 0

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

    # def sensor(self):
    #     left_sendir = self.direction.rotate(-90)
    #     right_sendir = self.direction.rotate(90)
    #     fwleft_sendir = self.direction + left_sendir
    #     fwright_sendir = self.direction + right_sendir
    #     dirs = [left_sendir, fwleft_sendir, self.direction, fwright_sendir, right_sendir]
    #     dists_to_food = []
    #     dists_to_obj = []
    #     for dir in dirs:
    #         dist_to_food = round(math.sqrt(NO_BINS_X ** 2 + NO_BINS_Y ** 2))
    #         dist_to_obj = 0
    #         temp = Vector2(self.head.x, self.head.y)
    #         counter = 0
    #         met_body = False
    #         while not is_outside(temp):
    #             counter += 1
    #             temp += dir
    #             if temp == apple_pos:
    #                 dist_to_food = counter
    #             if temp in self.body and not met_body:
    #                 dist_to_obj = counter
    #                 met_body = True
    #         if not met_body:
    #             dist_to_obj = counter
    #         dists_to_food.append(dist_to_food)
    #         dists_to_obj.append(dist_to_obj)
    #     return dists_to_food + dists_to_obj

    def collision_with_self(self, head):
        return head in self.body[1:]

    def collision_with_wall(self, head):
        return is_outside(head)

    def angle_with_apple(self):
        global apple_pos
        apple_direction = apple_pos - self.direction
        normalized_apple_direction = apple_direction.normalize()
        normalized_self_direction = self.direction.normalize()
        angle = math.radians(normalized_self_direction.angle_to(normalized_apple_direction))
        return angle, self.direction, normalized_apple_direction, normalized_self_direction

    def is_direction_blocked(self, direction_vector):
        next_step = Vector2(self.body[0].x, self.body[0].y) + direction_vector
        return self.collision_with_self(next_step) or self.collision_with_wall(next_step)

    def blocked_directions(self):
        left_dir_vec = Vector2(self.direction.y, -self.direction.x)
        right_dir_vec = -left_dir_vec
        is_front_blocked = self.is_direction_blocked(self.direction)
        is_left_blocked = self.is_direction_blocked(left_dir_vec)
        is_right_blocked = self.is_direction_blocked(right_dir_vec)
        return self.direction, is_front_blocked, is_left_blocked, is_right_blocked

    def generate_direction(new_direction):
        dir_choice = 0
        if new_direction == VEC_RIGHT:
            dir_choice = 1
        elif new_direction == VEC_LEFT:
            dir_choice = 0
        elif new_direction == VEC_DOWN:
            dir_choice = 2
        elif new_direction == VEC_UP:
            dir_choice = 3
        return dir_choice   


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

def fail(snake):
    self_hit = snake.body[0] in snake.body[1:]
    wall_hit = is_outside(snake.body[0])
    if self_hit or wall_hit:
        return True
    return False

def ai_fail(snake):
    if fail(snake) or snake.count_turn > 20:
        return True
    return False


def eval_genomes(genomes, config):
    for _, g in genomes:
        # net = neat.nn.RecurrentNetwork.create(g, config)
        net = neat.nn.FeedForwardNetwork.create(g, config)
        snake = Snake()
        g.fitness = 0

        init_apple_x = random.randint(0, NO_BINS_X - 1)
        init_apple_y = random.randint(0, NO_BINS_Y - 1)
        apples = [Apple(init_apple_x, init_apple_y)]

        score = 0
        run = True

        while run:
            # clock.tick(15)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    sys.exit()

            global apple_pos
            input = (snake.sensor())
            output = net.activate(input)
            index = output.index(max(output))
            if index == 0:
                snake.turn_right()
            if index == 1:
                snake.turn_left()
            if index == 2:
                snake.turn_up()
            if index == 3:
                snake.turn_down()

            prev_dist = snake.body[0].distance_to(apple_pos)
            snake.move()
            current_dist = snake.body[0].distance_to(apple_pos)
            if current_dist < prev_dist:
                g.fitness += 15 
            

            apples_rmv = []
            add_apple = False
            for apple in apples:
                if not apple.eaten and apple.pos == snake.body[0]:
                    apple.eaten = True
                    add_apple = True
                    apples_rmv.append(apple)

            if add_apple:
                score += 1
                g.fitness += 100 * score
                while True:
                    new_apple_x , new_apple_y = (random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1))
                    if (new_apple_x != apple.x or new_apple_y != apple.y) and (Vector2(new_apple_x, new_apple_y) not in snake.body):
                        apples.append(Apple(new_apple_x, new_apple_y))
                        break
                snake.add_block()

            for apple in apples_rmv:
                apples.remove(apple)

            # g.fitness = snake.total_turn + (2 ** score + (score ** 2.1) * 500) - ((score ** 1.2) * (0.25 * snake.total_turn) ** 1.3)

            if fail(snake):
                global best_fitness
                global best_ge
                g.fitness -= 900
                run = False
                if g.fitness > best_fitness:
                    best_fitness = g.fitness
                    best_ge = g

            if snake.count_turn > 20:
                g.fitness -= 99999
                run = False
            
            # draw_window(win, snake, apples, score)

def main():
    global best_fitness
    global best_ge
    population = Population(POP_SIZE, GENOME_LENGTH, NUM_WEIGHTS, weights_range, biases_range, NUMGEN)
    for i in range(NUMGEN):
        print("Generation {}".format(i))
        best_gen_fitness = -math.inf

        # Crossover
        for j in range(round(cx_rate * POP_SIZE)):
            # TODO: which type of crossover is this?
            rand_ind_1 = random.randrange(0, MU)
            rand_ind_2 = random.randrange(0, MU)
            offspring_1, offspring_2 = crossover(population.pop[rand_ind_1], population.pop[rand_ind_2], cx_prob)
            population.pop.extend([offspring_1, offspring_2])
            population.pop_size += 2

        # Mutation
        # TODO: which type of mutation is this?
        w_mutated_ind = []
        b_mutated_ind = []
        for i in range(round(weight_mut_rate * bias_mut_rate) * POP_SIZE):
            rand_ind = random.randrange(0, POP_SIZE)
            while rand_ind in w_mutated_ind or rand_ind in b_mutated_ind:
                rand_ind = random.randrange(0, POP_SIZE)
            offspring = mutate(population.pop[rand_ind], weight_mut_prob, bias_mut_prob, weight_mut_std, bias_mut_std)
            w_mutated_ind.append(rand_ind)
            b_mutated_ind.append(rand_ind)
            population.pop.append(offspring)
            population.pop_size += 1
        for i in range(round(weight_mut_rate) * POP_SIZE):
            rand_ind = random.randrange(0, POP_SIZE)
            while rand_ind in w_mutated_ind:
                rand_ind = random.randrange(0, POP_SIZE)
            offspring = mutate(population.pop[rand_ind], weight_mut_prob, 0, weight_mut_std, bias_mut_std)
            w_mutated_ind.append(rand_ind)
            population.pop.append(offspring)
            population.pop_size += 1
        for i in range(round(bias_mut_rate) * POP_SIZE):
            rand_ind = random.randrange(0, POP_SIZE)
            while rand_ind in b_mutated_ind:
                rand_ind = random.randrange(0, POP_SIZE)
            offspring = mutate(population.pop[rand_ind], 0, bias_mut_prob, weight_mut_std, bias_mut_std)
            b_mutated_ind.append(rand_ind)
            population.pop.append(offspring)
            population.pop_size += 1

        # Evaluate fitness
        for ind in population.pop:
            print("Running individual", population.pop.index(ind))
            ffnn = feedforward_nn(ind.genome)
            snake = Snake()
            ind.fitness = 0

            init_apple_x = random.randint(0, NO_BINS_X - 1)
            init_apple_y = random.randint(0, NO_BINS_Y - 1)
            apples = [Apple(init_apple_x, init_apple_y)]

            score = 0
            run = True
            while run:
                clock.tick(200)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        pygame.quit()
                        sys.exit()
        
                global apple_pos
                input = (snake.sensor())
                ffnn.activate(input)
                output = ffnn.output()
                index = output.index(max(output))
                if index == 0:
                    snake.turn_right()
                if index == 1:
                    snake.turn_left()
                if index == 2:
                    snake.turn_up()
                if index == 3:
                    snake.turn_down()

                prev_dist = snake.body[0].distance_to(apple_pos)
                snake.move()
                current_dist = snake.body[0].distance_to(apple_pos)
                if current_dist < prev_dist:
                    ind.fitness += 15

                apples_rmv = []
                add_apple = False
                for apple in apples:
                    if not apple.eaten and apple.pos == snake.body[0]:
                        apple.eaten = True
                        add_apple = True
                        apples_rmv.append(apple)

                if add_apple:
                    score += 1
                    ind.fitness += 100 * score
                    while True:
                        new_apple_x , new_apple_y = (random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1))
                        if (new_apple_x != apple.x or new_apple_y != apple.y) and (Vector2(new_apple_x, new_apple_y) not in snake.body):
                            apples.append(Apple(new_apple_x, new_apple_y))
                            break
                    snake.add_block()

                for apple in apples_rmv:
                    apples.remove(apple)

                # ind.fitness = snake.total_turn + (2 ** score + (score ** 2.1) * 500) - ((score ** 1.2) * (0.25 * snake.total_turn) ** 1.3)

                if ind.fitness > max_fitness:
                    print("Successfully passed with fitness =", ind.fitness)
                    run = False
                    best_fitness


                if fail(snake):
                    ind.fitness -= 900
                    run = False
                    if ind.fitness > best_fitness:
                        best_fitness = ind.fitness
                        best_ge = ind.genome

                    if ind.fitness > best_gen_fitness:
                        best_gen_fitness = ind.fitness

                if snake.count_turn > 20:
                    ind.fitness -= 900
                    run = False
                    if ind.fitness > best_fitness:
                        best_fitness = ind.fitness
                        best_ge = ind.genome
                    if ind.fitness > best_gen_fitness:
                        best_gen_fitness = ind.fitness
                
                # draw_window(win, snake, apples, score)
        print(f"Best generation fitness: {best_gen_fitness}")

        # Selection
        # TODO: What type of selection is this?
        fitnesses = [ind.fitness for ind in population.pop]
        population.pop = list(zip(fitnesses, population.pop))
        
        # Print results & save data to lists
        generation_fitness = [fit for fit, _ in sorted(population.pop, key=lambda x: x[0], reverse=True)]
        print("Generation fitness:", generation_fitness)
        global best_gen_fitnesses
        global avg_gen_fitnesses
        global worst_gen_fitnesses
        best_gen_fitnesses.append(best_gen_fitness)
        avg_gen_fitnesses.append(np.mean(generation_fitness))
        worst_gen_fitnesses.append(generation_fitness[-1])

        global std_gen_fitnesses
        std_gen_fitnesses.append(np.std(generation_fitness))

        population.pop = [ind for _, ind in sorted(population.pop, key=lambda x: x[0], reverse=True)]
        population.pop = population.pop[:POP_SIZE]
        population.current_gen += 1

    print("Best fitness overall:", best_fitness)
    with open("best_snake_ge_2.p", "wb") as best_g:
        pickle.dump(best_ge, best_g)

    plt.plot(best_gen_fitnesses)
    plt.plot(avg_gen_fitnesses)
    plt.plot(worst_gen_fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness throughout generations")
    plt.legend(["Best fitness", "Average fitness", "Worst fitness"])
    plt.savefig("fitness_trend.png")
    plt.show()

    plt.plot(std_gen_fitnesses)
    plt.title("Standard deviation of fitness throughout generations")
    plt.xlabel("Generation")
    plt.ylabel("Standard deviation")
    plt.savefig("fitness_std.png")
    plt.show()


def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, NUMGEN)
    with open("best_snake_ge.p", "wb") as best_g:
        pickle.dump(winner, best_g)
        # global best_ge
        # pickle.dump(best_ge, best_g)

    draw_net(config, winner, True)
    plot_stats(stats, ylog=False, view=True)
    plot_species(stats, view=True)

def load_n_test(g, config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # net = neat.nn.RecurrentNetwork.create(g, config)
    net = neat.nn.FeedForwardNetwork.create(g, config)
    snake = Snake()

    init_apple_x = random.randint(0, NO_BINS_X - 1)
    init_apple_y = random.randint(0, NO_BINS_Y - 1)
    apples = [Apple(init_apple_x, init_apple_y)]

    score = 0
    run = True

    while run:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

        global apple_pos
        input = (snake.sensor())
        output = net.activate(input)
        index = output.index(max(output))
        if index == 0:
            snake.ai_turn_right()
        if index == 1:
            snake.ai_turn_left()
        if index == 2:
            pass
        
        snake.move()

        apples_rmv = []
        add_apple = False
        for apple in apples:
            if not apple.eaten and apple.pos == snake.body[0]:
                apple.eaten = True
                add_apple = True
                apples_rmv.append(apple)

        if add_apple:
            score += 1
            while True:
                new_apple_x , new_apple_y = (random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1))
                if (new_apple_x != apple.x or new_apple_y != apple.y) and (Vector2(new_apple_x, new_apple_y) not in snake.body):
                    apples.append(Apple(new_apple_x, new_apple_y))
                    break
            snake.add_block()

        for apple in apples_rmv:
            apples.remove(apple)

        if fail(snake) and snake.count_turn > 20:
            run = False
        
        draw_window(win, snake, apples, score)

def load_n_test_v2(g):
    ffnn = feedforward_nn(g)
    snake = Snake()

    init_apple_x = random.randint(0, NO_BINS_X - 1)
    init_apple_y = random.randint(0, NO_BINS_Y - 1)
    apples = [Apple(init_apple_x, init_apple_y)]

    score = 0
    run = True
    while run:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

        global apple_pos
        input = (snake.sensor())
        ffnn.activate(input)
        output = ffnn.output()
        index = output.index(max(output))
        if index == 0:
            snake.turn_right()
        if index == 1:
            snake.turn_left()
        if index == 2:
            snake.turn_up()
        if index == 3:
            snake.turn_down()

        # prev_dist = snake.body[0].distance_to(apple_pos)
        snake.move()
        # current_dist = snake.body[0].distance_to(apple_pos)
        # if current_dist < prev_dist:
        #     ind.fitness += 15

        apples_rmv = []
        add_apple = False
        for apple in apples:
            if not apple.eaten and apple.pos == snake.body[0]:
                apple.eaten = True
                add_apple = True
                apples_rmv.append(apple)

        if add_apple:
            score += 1
            # ind.fitness += 100 * score
            while True:
                new_apple_x , new_apple_y = (random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1))
                if (new_apple_x != apple.x or new_apple_y != apple.y) and (Vector2(new_apple_x, new_apple_y) not in snake.body):
                    apples.append(Apple(new_apple_x, new_apple_y))
                    break
            snake.add_block()

        for apple in apples_rmv:
            apples.remove(apple)

        draw_window(win, snake, apples, score)

        if fail(snake) or snake.count_turn > 20:
            run = False

        

def play():
    score = 0
    snake = Snake()
    init_apple_x = random.randint(0, NO_BINS_X - 1)
    init_apple_y = random.randint(0, NO_BINS_Y - 1)
    apples = [Apple(init_apple_x, init_apple_y)]

    run = True

    while run:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        snake.move()

        keys = pygame.key.get_pressed()
        current_snake_direction = snake.direction
        if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and current_snake_direction != VEC_LEFT and snake.moved:
            snake.turn_right()            
        if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and current_snake_direction != VEC_RIGHT and snake.moved:
            snake.turn_left()
        if (keys[pygame.K_UP] or keys[pygame.K_w]) and current_snake_direction != VEC_DOWN and snake.moved:
            snake.turn_up()
        if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and current_snake_direction != VEC_UP and snake.moved:
            snake.turn_down()

        apples_rmv = []
        add_apple = False
        for apple in apples:
            if not apple.eaten and apple.pos == snake.body[0]:
                apple.eaten = True
                add_apple = True
                apples_rmv.append(apple)

        if add_apple:
            score += 1
            while True:
                new_apple_x , new_apple_y = (random.randint(0, NO_BINS_X - 1), random.randint(0, NO_BINS_Y - 1))
                if (new_apple_x != apple.x or new_apple_y != apple.y) and (Vector2(new_apple_x, new_apple_y) not in snake.body):
                    apples.append(Apple(new_apple_x, new_apple_y))
                    break
            snake.add_block()

        for apple in apples_rmv:
            apples.remove(apple)

        if ai_fail(snake):
            print(f"Score = {score}")

            run = False

        draw_window(win, snake, apples, score)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    config_files = ['config-feedforward.txt', 'config-recurrent.txt']
    local_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Enable test mode on the given genome")
    parser.add_argument("--play", help="Enable real-player mode", action="store_true")
    args = parser.parse_args()

    config_path = os.path.join(local_dir, config_files[0])

    if args.test:
        with open(args.test, "rb") as pickled_g:
            g = pickle.load(pickled_g)
        load_n_test(g, config_path=config_path)
    elif args.play:
        play()
    else:
        run(config_path)

    # main()
    
    # with open('best_snake_ge_2.p', 'rb') as pickled_g:
    #     g = pickle.load(pickled_g)
    # load_n_test_v2(g)
