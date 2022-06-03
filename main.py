# Import custom snake environment 
from env import SnakeEnv
from training import train_agent

# Import libraries
import numpy as np

# Define env and q_table
env = SnakeEnv()
q_table = np.zeros([30*20*3, env.action_space.n])

# Random action
episodes = 1
for episode in range(episodes):
    continue
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        state, reward, done = env.step(action)
        score += reward

    print(f"Episode: {episode}, Score: {score}")
    env.render_gif()


# Untrained Q-Learning
# TODO

# Train agent
q_table = train_agent(q_table, env)
np.savetxt("q_table.txt", q_table)

print(q_table)
# Trained Q-Learning
episodes = 1
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        score += reward

    print(f"Episode: {episode}, Score: {score}")
    env.render_gif()