from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
import random
import numpy as np
import pandas as pd
from operator import add

class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        # self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, snack, s):  # Draws the game field as a 2D Array for the NN
        '''field = np.zeros((rows, rows), dtype=int)  # Somehow pos is reversed so I reverse it back
        field[snack.pos[1], snack.pos[0]] = 1  # Snack is 1
        if not snakeIsDead:  # If the snake is dead, it's head would be out of the arrays bounds and give an error
            field[(s.body[0].pos[1], s.body[0].pos[0])] = 3  # Head is 3
            for part in range(1, len(s.body)):
                field[s.body[part].pos[1], s.body[part].pos[0]] = 2  # Body is 2

        if not snakeIsDead:
            head_pos = np.argwhere(field == 3)
            snack_pos = np.argwhere(field == 1)
            head_snack_dist = np.linalg.norm(head_pos - snack_pos)

            north_from_head_pos = np.array(head_pos)  # Redefine it as np array before setting north to it or bug
            north_from_head_pos[0][0] = 0
            head_north_dist = np.linalg.norm(head_pos - north_from_head_pos)  # Distance between head and north wall

            east_from_head_pos = np.array(head_pos)  # Redefine it as np array before setting north to it or bug
            east_from_head_pos[0][1] = 19
            head_east_dist = np.linalg.norm(head_pos - east_from_head_pos)  # Distance between head and east wall

            south_from_head_pos = np.array(head_pos)  # Redefine it as np array before setting north to it or bug
            south_from_head_pos[0][0] = 19
            head_south_dist = np.linalg.norm(head_pos - south_from_head_pos)  # Distance between head and south wall

            west_from_head_pos = np.array(head_pos)  # Redefine it as np array before setting north to it or bug
            west_from_head_pos[0][1] = 0
            head_west_dist = np.linalg.norm(head_pos - west_from_head_pos)  # Distance between head and west wall

            state = []
            state.extend((head_snack_dist, head_north_dist, head_east_dist, head_south_dist, head_west_dist))
            state.append(0)
            state = [i / 20 for i in state]  # TODO Maybe this is not necessary
            #print(state)
        else:
            state = state_old
            state[5] = 1'''

        # print("Head", (s.body[0].pos[1], s.body[0].pos[0]))
        # print("Snack", (snack.pos[1], snack.pos[0]))
        new_state = [
            (s.body[0].dirnx == -1 and s.body[0].pos[0] < 1) or (s.body[0].dirnx == 1 and s.body[0].pos[0] > 18)
            or (s.body[0].dirny == 1 and s.body[0].pos[1] > 18) or (s.body[0].dirny == -1 and s.body[0].pos[1] < 1),  # Danger Straight

            (s.body[0].dirnx == -1 and s.body[0].pos[1] < 1) or (s.body[0].dirnx == 1 and s.body[0].pos[1] > 18)
            or (s.body[0].dirny == 1 and s.body[0].pos[0] < 1) or (s.body[0].dirny == -1 and s.body[0].pos[0] > 18),  # Danger Right

            (s.body[0].dirnx == -1 and s.body[0].pos[1] > 18) or (s.body[0].dirnx == 1 and s.body[0].pos[1] < 1)
            or (s.body[0].dirny == 1 and s.body[0].pos[0] > 18) or (s.body[0].dirny == -1 and s.body[0].pos[0] < 1),  # Danger Left

            s.body[0].dirnx == -1,  # Move Left
            s.body[0].dirnx == 1,  # Move Right
            s.body[0].dirny == -1,  # Move Up
            s.body[0].dirny == 1,  # Move Down
            s.body[0].pos[0] > snack.pos[0],  # Snack Left
            s.body[0].pos[0] < snack.pos[0],  # Snack Right
            s.body[0].pos[1] > snack.pos[1],  # Snack Up
            s.body[0].pos[1] < snack.pos[1]  # Snack Down
        ]
        '''
        if s.body[0].dirnx == -1:  # and c.pos[0] < 0:
            print("dirnx == -1")  # Left
        elif s.body[0].dirnx == 1:  # and c.pos[0] > c.rows - 1:
            print("dirnx == 1")  # Right
        elif s.body[0].dirny == -1:  # and c.pos[1] < 0:
            print("dirny == -1")  # Up
        elif s.body[0].dirny == 1:  # and c.pos[1] > c.rows - 1:
            print("dirny == 1")  # Down

        if (s.body[0].dirnx == -1 and s.body[0].pos[0] < 1) or (s.body[0].dirnx == 1 and s.body[0].pos[0] > 18) \
                or (s.body[0].dirny == 1 and s.body[0].pos[1] > 18) or (s.body[0].dirny == -1 and s.body[0].pos[1] < 1):
            print("Danger straight")
        elif (s.body[0].dirnx == -1 and s.body[0].pos[1] < 1) or (s.body[0].dirnx == 1 and s.body[0].pos[1] > 18) \
                or (s.body[0].dirny == 1 and s.body[0].pos[0] < 1) or (s.body[0].dirny == -1 and s.body[0].pos[0] > 18):
            print("Danger right")
        elif (s.body[0].dirnx == -1 and s.body[0].pos[1] > 18) or (s.body[0].dirnx == 1 and s.body[0].pos[1] < 1) \
                or (s.body[0].dirny == 1 and s.body[0].pos[0] > 18) or (s.body[0].dirny == -1 and s.body[0].pos[0] < 1):
            print("Danger left")

        if s.body[0].pos[0] > snack.pos[0]:
            print("Snack Left")
        if s.body[0].pos[0] < snack.pos[0]:
            print("Snack Right")
        if s.body[0].pos[1] > snack.pos[1]:
            print("Snack Up")
        if s.body[0].pos[1] < snack.pos[1]:
            print("Snack Down")'''

        for i in range(len(new_state)):
            if new_state[i]:
                new_state[i] = 1
            else:
                new_state[i] = 0
        #print(Bstate)

        #print(state)
        #field = [i / 3 for i in field

        return np.asarray(new_state)
        # return field

    def set_reward(self, snakeIsDead, appleWasEaten):  # Sets the reward for the current state
        self.reward = 0
        if snakeIsDead:
            self.reward = -10
        if appleWasEaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=11))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=4, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
