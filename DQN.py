from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np

class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.learning_rate = 0.0005  # Represents the amount at which a weight is changed when the model is optimizing its parameters.
        self.model = self.network()
        #self.model = self.network("weights40x40V2.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, xpos, ypos, xdir, ydir, snake_block, xfood, yfood, dis_width, dis_height, snake_List):
        tail_infront = False
        tail_right = False
        tail_left = False

        try:
            temp_save = snake_List[-2]
            # Temporarily ignore a part of the sanke. 
            # This is made due to a bug that creates errors in the tail_infront, tail_right and tail_left variables.
            snake_List[-2] = [-66, -66]
            if [xpos - ydir, ypos + xdir] in snake_List:
                tail_right = True
            if [xpos + 2 * xdir, ypos + 2 * ydir] in snake_List:
                tail_infront = True
            if [xpos + ydir, ypos - xdir] in snake_List:
                tail_left = True
            snake_List[-2] = temp_save
        except IndexError:
            pass

        state = [
            (xdir < 0 and xpos == snake_block) or (xdir > 0 and xpos == dis_width - 2*snake_block)
            or (ydir > 0 and ypos == dis_height - 2*snake_block) or (ydir < 0 and ypos == snake_block) or tail_infront,
            # Danger Straight

            (xdir < 0 and ypos == 0) or (xdir > 0 and ypos == dis_height - snake_block)
            or (ydir > 0 and xpos == 0) or (ydir < 0 and xpos == dis_width - snake_block) or tail_right,
            # Danger Right

            (xdir < 0 and ypos == dis_height - snake_block) or (xdir > 0 and ypos == 0)
            or (ydir > 0 and xpos == dis_width - snake_block) or (ydir < 0 and xpos == 0) or tail_left,
            # Danger Left

            xdir == -snake_block,  # Move Left
            xdir == snake_block,  # Move Right
            ydir == -snake_block,  # Move Up
            ydir == snake_block,  # Move Down
            xpos > xfood,  # Snack Left
            xpos < xfood,  # Snack Right
            ypos > yfood,  # Snack Up
            ypos < yfood  # Snack Down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def set_reward(self, game_over, eaten):
        self.reward = 0
        if game_over:
            self.reward = -10
        if eaten:
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
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            # The minibatch is a sample of all the data collected through training. Since the memory grows after time, it's contraproductive
            # to use every sample of data from the memory as training sample. So the training data consists of max. 1000 samples.
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
    # TODO: Test difference between the 2 target defenitions in replay_new and trainshortmemory
    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
