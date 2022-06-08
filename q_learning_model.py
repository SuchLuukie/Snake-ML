# Import files and libs
from env import SnakeEnv
from tqdm import tqdm
import numpy as np
import random

class QLearningModel:
    def __init__(self):
        # Define env
        self.env = SnakeEnv()

        # Define the tables
        self.large_states_index = []
        self.small_states_index = []
        self.large_q_table = []
        self.small_q_table = []

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

        # Set training size
        self.training_size = 100

        # Train the model
        print("[!] Training Q Learning Model")
        self.train()

        print("[!] Evaluating Q Learning Model")
        self.episodes = 10
        self.evaluate()


    def train(self):
        for i in tqdm(range(self.training_size)):
            self.env.reset()
            reward, done = 0, False
            large_state, small_state = self.env.states()

            while not done:
                # Get table indexes
                large_state_index, small_state_index = self.get_states_index(large_state, small_state)

                # Explores action space
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()

                # Exploit previously learned value
                else:
                    #large_state_actions = self.large_q_table[large_state_index]
                    small_state_actions = self.small_q_table[small_state_index]
                    action = np.argmax(small_state_actions)

                    # If large state actions are all 0 then look at small_state actions
                    #if len([i for i in large_state_actions if i != 0]) == 0:
                    #    action = np.argmax(small_state_actions)
                    
                    # Otherwise just use the large state actions
                    #else:
                    #    action = np.argmax(large_state_actions)

                # Take the next step in the env
                next_large_state, next_small_state, reward, done = self.env.step(action)
                next_large_state_index, next_small_state_index = self.get_states_index(next_large_state, next_small_state)

                # Old Q table values
                #old_large_value = self.large_q_table[large_state_index][action]
                old_small_value = self.small_q_table[small_state_index][action]

                # Next Q table values
                #next_large_max = max(self.large_q_table[next_large_state_index])
                next_small_max = max(self.small_q_table[next_small_state_index])

                # New values
                #new_large_value = (1 - self.alpha) * old_large_value + self.alpha * (reward + self.gamma * next_large_max)
                new_small_value = (1 - self.alpha) * old_small_value + self.alpha * (reward + self.gamma * next_small_max)

                # Update new values
                #self.large_q_table[large_state_index][action] = new_large_value
                self.small_q_table[small_state_index][action] = new_small_value

                # Ready for next iteration
                #large_state = next_large_state
                small_state = next_small_state
    

    def evaluate(self):
        episode_dict = {}
        average = 0
        for episode in range(self.episodes):
            self.env = SnakeEnv()
            large_state, small_state = self.env.reset()
            score, done = 0, False

            while not done:
                # Get table indexes for the given states
                large_state_index, small_state_index = self.get_states_index(large_state, small_state)

                # Get the actions for the given states
                large_state_actions = self.large_q_table[large_state_index]
                small_state_actions = self.small_q_table[small_state_index]

                # If large state actions are all 0 then look at small_state actions
                if len([i for i in large_state_actions if i != 0]) == 0:
                    action = np.argmax(small_state_actions)
                
                # Otherwise just use the large state actions
                else:
                    action = np.argmax(large_state_actions)

                # Take the step in the env
                large_state, small_state, reward, done = self.env.step(action)

                # Apply reward
                score += reward

            # Add to the average
            average += score

            # Print episode score
            print(f"Episode: {episode}, Score: {score}")

            # Add to dictionary
            episode_dict.update({
                episode: {
                    "env": self.env,
                    "score": score
                }
            })
        
        # Calculate and print average score
        average = average / self.episodes
        print(f"Episodes average: {average}")

        # Get the best score
        best_episode = episode_dict[0]
        for episode in episode_dict:
            if episode_dict[episode]["score"] > best_episode["score"]:
                best_episode = episode_dict[episode]

        print("[!] Best episode score: {}".format(best_episode["score"]))
        best_episode["env"].render_gif()


    def get_states_index(self, large_state, small_state):
        large_state_index = [i for i in range(len(self.large_states_index)) if large_state == self.large_states_index[i]]
        small_state_index = [i for i in range(len(self.small_states_index)) if small_state == self.small_states_index[i]]


        if large_state_index == []:
            self.large_states_index.append(large_state)
            large_state_index = len(self.large_states_index) - 1
            self.large_q_table.append([0 for i in range(self.env.action_space.n)])
        else:
            large_state_index = large_state_index[0]


        if small_state_index == []:
            self.small_states_index.append(small_state)
            small_state_index = len(self.small_states_index) - 1
            self.small_q_table.append([0 for i in range(self.env.action_space.n)])
        else:
            small_state_index = small_state_index[0]


        return large_state_index, small_state_index

QLearningModel()