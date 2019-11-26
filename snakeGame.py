import random
import pygame
import numpy as np
import pynput
from pynput.keyboard import Key, Controller
import time
from DQN import DQNAgent

RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

keyboard = Controller()

class cube(object):
    rows = 20
    w = 500

    def __init__(self, start, dirnx=1, dirny=0, color=RED):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(surface, self.color, (i * dis + 1, j * dis + 1, dis - 2, dis - 2))
        if eyes:
            centre = dis // 2
            radius = 3
            circleMiddle = (i * dis + centre - radius, j * dis + 8)
            circleMiddle2 = (i * dis + dis - radius * 2, j * dis + 8)
            pygame.draw.circle(surface, BLACK, circleMiddle, radius)
            pygame.draw.circle(surface, BLACK, circleMiddle2, radius)

class snake(object):
    body = []
    turns = {}

    def __init__(self, color, pos):
        self.color = color
        self.head = cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.numOfGame = 0
        self.isDead = False

    def move(self, move_to_do):

        for event in pygame.event.get():  # Allows the spectator to close the game without it crashing
            if event.type == pygame.QUIT:
                pygame.quit()

            #keys = pygame.key.get_pressed()
            #for key in keys:
        if move_to_do is 0:  # Move Left (0) pygame specific: keys[pygame.K_LEFT]
        #if keys[pygame.K_LEFT]:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif move_to_do is 1:  # Move Up (1)
        #elif keys[pygame.K_UP]:
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif move_to_do is 2:  # Move Right (2)
        #elif keys[pygame.K_RIGHT]:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        elif move_to_do is 3:  # Move Down (3)
        #elif keys[pygame.K_DOWN]:
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:  # Make a turn
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:  # Just go forward
                c.move(c.dirnx, c.dirny)

            if c.dirnx == -1 and c.pos[0] < 0:  # Here's how to die :)
                self.die()
            elif c.dirnx == 1 and c.pos[0] > c.rows - 1:
                self.die()
            elif c.dirny == 1 and c.pos[1] > c.rows - 1:
                self.die()
            elif c.dirny == -1 and c.pos[1] < 0:
                self.die()

    def reset(self, pos):
        print('Game: ', self.numOfGame, 'Score: ', len(self.body) - 1)
        self.head = cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        self.isDead = False
        self.numOfGame += 1

    def die(self):
        self.isDead = True  # Don't immediately reset, for the AI to get the fact that it died

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(cube((tail.pos[0] - 1, tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(cube((tail.pos[0] + 1, tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(cube((tail.pos[0], tail.pos[1] - 1)))
        elif dx == 0 and dy == -1:
            self.body.append(cube((tail.pos[0], tail.pos[1] + 1)))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

def redrawWindow(surface, hs):
    global rows, width, s, snack
    surface.fill(BLACK)
    s.draw(surface)
    snack.draw(surface)
    pygame.draw.line(surface, (255, 255, 255), (0, width), (width, width))  # Draw line to separate Game from Score

    font = pygame.font.Font(None, 30)
    score = font.render('Score: '+str(len(s.body)-1), True, WHITE)
    surface.blit(score, (60, 550))
    hscore = font.render('Highscore: '+str(hs), True, WHITE)
    surface.blit(hscore, (280, 550))

    pygame.display.update()

def randomSnack(rows, item):
    x = random.randrange(rows)
    y = random.randrange(rows)

    return x, y

def initialize_game(agent, appleWasEaten):
    state_init1 = agent.get_state(snack, s)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = 2
    s.move(action)
    state_init2 = agent.get_state(snack, s)
    reward1 = agent.set_reward(s.isDead, appleWasEaten)
    agent.remember(state_init1, action, reward1, state_init2, s.isDead)
    agent.replay_new(agent.memory)

def getHighscore(hs):  # Updates the highscore
    if len(s.body)-1 >= hs:
        hs = len(s.body)-1
    return hs

def main():
    global width, rows, s, snack
    width = 500
    rows = 20
    highscore = 0
    win = pygame.display.set_mode((width, width+100))  # Create window
    s = snake(RED, (10, 10))
    snack = cube(randomSnack(rows, s), color=GREEN)
    agent = DQNAgent()
    speed = 0  # 0 --> Superduperhyperfucking fast ; 10 --> Normal speed

    pygame.init()  # Init pygame
    clock = pygame.time.Clock()

    while s.numOfGame < 300:  # Only play 300 games
        clock.tick(speed)  # Delay for speed

        agent.epsilon = 80 - s.numOfGame  # Set epsilon to a high value at beginning that becomes less and less

        state_old = agent.get_state(snack, s)  # Get the state BEFORE the move is made

        if random.randint(0, 200) < agent.epsilon:
            # Explore
            finale_move = random.randint(0, 3)  # Random move 1: Left 2: Up 3: Right 4: Down
            # print('Explore')
        else:
            # Exploitation
            prediction = agent.model.predict(state_old.reshape((1, 11)))  # Get action for given state from NN
            finale_move = np.argmax(prediction[0])  # Predicted move
            # print('Predicted move:', finale_move)
            # print('Exploit')

        s.move(finale_move)  # Execute final_move
        state_new = agent.get_state(snack, s)  # Get the state AFTER the move is made

        appleWasEaten = False  # Bool for reward
        if s.body[0].pos == snack.pos:  # If snake eats an apple
            s.addCube()
            appleWasEaten = True  # If an apple was eaten set to true
            snack = cube(randomSnack(rows, s), color=GREEN)

        '''for x in range(len(s.body)):
            if s.body[x].pos in list(map(lambda z: z.pos, s.body[x + 1:])):  # If snake bites its own tail
                s.die()
                break'''

        reward = agent.set_reward(s.isDead, appleWasEaten)  # Set reward for the new state following the action
        #print(reward)

        agent.train_short_memory(state_old, finale_move, reward, state_new, s.isDead)  # Train short memory with new action and state

        agent.remember(state_old, finale_move, reward, state_new, s.isDead)  # Save the new data in long term memory

        if s.isDead:  # If the die() function was called, isDead is True and then the game is reset
            agent.replay_new(agent.memory)  # Fit the neural network
            s.reset((10, 10))
            initialize_game(agent, appleWasEaten)

        highscore = getHighscore(highscore)  # Set highscore
        redrawWindow(win, highscore)

    agent.model.save_weights('weights.hdf5')

main()
