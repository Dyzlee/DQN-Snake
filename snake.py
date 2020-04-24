import pygame
import random
import seaborn as sns
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from DQN import DQNAgent

pygame.init()

white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

dis_width = 400
dis_height = 400

dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game by Edureka.concat(DQN)')
score_font = pygame.font.SysFont("calibri", 12)

clock = pygame.time.Clock()

snake_block = 40  # The fields size is always 400 / snake_block ; 10 => 40x40 ; 20 => 20x20 ; 40 => 10x10
snake_speed = 0  # 0=instant speed 5=take your time 5>faster

def your_score(hs, score):
    value = score_font.render("Score: " + str(score) + " Highscore: " + str(hs), True, yellow)
    dis.blit(value, [0, 0])

def random_food(snake_List):
    while True:
        food = [random.randint(0, dis_width / snake_block - 1) * snake_block, random.randint(0, dis_height / snake_block - 1) * snake_block]  # Minus 1 because if the coord is 400, its out of bounds
        if food not in snake_List:
            return food

def get_highscore(highscore, score):
    if score >= highscore:
        highscore = score
    return highscore

def draw_snake(sBlock, sList):
    for x in sList:
        pygame.draw.rect(dis, black, [x[0], x[1], sBlock, sBlock])

def gameLoop():
    game_over = False
    games = 0
    games_to_play = 300
    highscore = 0
    score_plot = []
    games_plot = []
    agent = DQNAgent()

    while games < games_to_play:  # Play a total of x games
        xpos = dis_width / 2  # X Spawn point coordinate
        ypos = dis_height / 2  # Y Spawn point coordinate

        xdir = 0
        ydir = 0

        snake_List = []
        length_of_snake = 1

        food = random_food(snake_List)
        xfood = food[0]
        yfood = food[1]

        while not game_over:
            agent.epsilon = 80 - games
            # agent.epsilon = 0

            state_old = agent.get_state(xpos, ypos, xdir, ydir, snake_block, xfood, yfood, dis_width, dis_height, snake_List)

            if randint(0, 200) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
            # Do Action -------------------------------------------------------------
            if np.array_equal(final_move, [1, 0, 0]):
                pass
            elif np.array_equal(final_move, [0, 1, 0]) and ydir is 0:  # right - going horizontal
                ydir = snake_block
                xdir = 0
            elif np.array_equal(final_move, [0, 1, 0]) and xdir is 0:  # right - going vertical
                xdir = snake_block
                ydir = 0
            elif np.array_equal(final_move, [0, 0, 1]) and ydir is 0:  # left - going horizontal
                ydir = -snake_block
                xdir = 0
            elif np.array_equal(final_move, [0, 0, 1]) and xdir is 0:  # left - going vertical
                xdir = -snake_block
                ydir = 0
            # Did Action -------------------------------------------------------------
            # Update Frame after Action ----------------------------------------------
            eaten = False
            # If collide with border
            if xpos == dis_width - snake_block and xdir > 0 or xpos == 0 and xdir < 0 or ypos == dis_height - snake_block and ydir > 0 or ypos == 0 and ydir < 0:
                game_over = True
            xpos += xdir
            ypos += ydir
            dis.fill(blue)
            pygame.draw.rect(dis, green, [xfood, yfood, snake_block, snake_block])
            snake_Head = [xpos, ypos]
            snake_List.append(snake_Head)
            if len(snake_List) > length_of_snake:
                del snake_List[0]

            for x in snake_List[:-1]:
                if x == snake_Head:
                    game_over = True

            draw_snake(snake_block, snake_List)
            highscore = get_highscore(highscore, length_of_snake - 1)
            your_score(highscore, length_of_snake - 1)

            if xpos == xfood and ypos == yfood:
                eaten = True
                food = random_food(snake_List)
                xfood = food[0]
                yfood = food[1]
                length_of_snake += 1
            # Updated Frame after Action ---------------------------------------------

            state_new = agent.get_state(xpos, ypos, xdir, ydir, snake_block, xfood, yfood, dis_width, dis_height, snake_List)
            reward = agent.set_reward(game_over, eaten)

            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

            agent.remember(state_old, final_move, reward, state_new, game_over)

            pygame.display.update()
            clock.tick(snake_speed)

        print("Game:", games + 1, "Score:", length_of_snake - 1, "Highscore:", highscore)
        agent.replay_new(agent.memory)
        games += 1
        score_plot.append(length_of_snake - 1)
        games_plot.append(games)
        game_over = False

    agent.model.save_weights('weights10x10V3.hdf5')
    pygame.quit()
    # Plot stats of game:
    sns.set(color_codes=True)
    ax = sns.regplot(x=games_plot, y=score_plot)
    ax.set(xlabel='games', ylabel='score')
    plt.show()
    quit()

gameLoop()
