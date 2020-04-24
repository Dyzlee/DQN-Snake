# DQN-Snake
My try on Deep-Q-Learning (Based on an Article by Mauro Comi https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a with a Snake game made by Edureka https://www.edureka.co/blog/snake-game-with-pygame/)

# How to use the pretrained weights
1. Choose a field size you want to watch the snake on (10x10, 20x20 or 40x40)
2. Adjust the ```snake_block``` variable to 10 for 40x40, 20 for 20x20 or 40 for 10x10 in the ```snake.py``` file at line 28
3. Also change the ```snake_speed``` variable to around 8 (the higher the faster) in the ```snake.py``` file at line 29
4. Remove the ```#``` of ```agent.epsilon = 0``` to set epsilon to 0 in the ```snake.py``` file at lines 74 and 75
5. Remove the ```#``` of ```self.model = self.network("weights40x40.hdf5")``` in the ```DQN.py``` file at line 15 and change the filename to the field size you want to watch (Ex: Change ```self.model = self.network("weights40x40.hdf5")``` to ```self.model = self.network("weights10x10.hdf5")``` for a 10x10 Snake game). You can of course use different weights for different field sizes, but the Snake will have different behaviours.
