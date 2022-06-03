# Import custom snake environment 
from env import SnakeEnv

# Import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import adam_v2

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = SnakeEnv()

# Untrained
episodes = 1
for episode in range(episodes):
    states = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done = env.step(action)
        score += reward

    print(f"Episode: {episode}, Score: {score}")
    env.render_gif()


## Trained
#states = env.observation_space.shape
#actions = env.action_space.n
#
#def build_model(states, actions):
#    model = Sequential()
#    model.add(Dense(30, activation='relu', input_shape=states))
#    model.add(Dense(30, activation='relu'))
#    model.add(Dense(actions, activation='linear'))
#    return model
#
#model = build_model(states, actions)
#
#def build_agent(model, actions):
#    polcy = BoltzmannQPolicy()
#    memory = SequentialMemory(limit=50000, window_length=1)
#    dqn = DQNAgent(model=model, memory=memory, policy=polcy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#    return dqn
#
#dqn = build_agent(model, actions)
#dqn.compile(adam_v2(lr=1e-3, metrics=["mae"]))
#dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)