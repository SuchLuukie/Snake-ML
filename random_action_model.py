from env import SnakeEnv


class RandomActionsModel:
    def __init__(self):
        # Define the enviroment
        self.env = SnakeEnv()

        # Define amount of episodes
        self.eval_episodes = 10

        self.gif_file_name = "random_actions"
        self.evaluate(render_best=True)


    # Main function that evaluates the model
    def evaluate(self, render_best = False):
        # Variables to keep track of evaluation
        best_episode = None
        average = 0

        for episode in range(self.eval_episodes):
            _, _ = self.env.reset()
            score, done = 0, False
            
            while not done:
                # Get a random action from the env
                action = self.env.action_space.sample()

                # Take the step in the env
                _, _, reward, done = self.env.step(action)

                # Apply reward
                score += reward

            # Add to the average
            average += score

            # Check if it's the best episode
            if best_episode == None:
                best_episode = [self.env, score]

            else:
                if score > best_episode[1]:
                    best_episode = [self.env, score]

        
        # Calculate and print average score
        average = int(average / self.eval_episodes)
        print(f"[!] Episodes average: {average}")

        # If render is True it will save the gif to the file name "TrainingSessionAmount_Score.gif"
        if render_best:
            print("[!] Best episode score: {}".format(best_episode[1]))
            best_episode[0].render_gif(self.gif_file_name + "_" + str(best_episode[1]) + ".gif")

RandomActionsModel()