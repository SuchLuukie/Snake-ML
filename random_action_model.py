from env import SnakeEnv


class RandomActionsModel:
    def __init__(self):
        # Define the enviroment
        self.env = SnakeEnv()

        # Define amount of episodes
        self.episodes = 1

        self.evaluate()


    def evaluate(self):
        average = 0
        for episode in range(self.episodes):
            score, done, state = 0, False, self.env.reset()
            
            while not done:
                # Get a random action from the env
                action = self.env.action_space.sample()

                # Take the step in the env
                states, reward, done = self.env.step(action)

                # Apply reward
                score += reward

            # Add to the average
            average += score

            # Print episode score
            print(f"Episode: {episode}, Score: {score}")
            self.env.render_gif()
        
        # Calculate and print average score
        average = average / self.episodes
        print(f"Episodes average: {average}")

RandomActionsModel()