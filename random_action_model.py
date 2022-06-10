from env import SnakeEnv


class RandomActionsModel:
    def __init__(self):
        # Define the enviroment
        self.env = SnakeEnv()

        # Define amount of episodes
        self.episodes = 10

        self.evaluate()


    def evaluate(self):
        episode_dict = {}
        average = 0
        for episode in range(self.episodes):
            score, done, _ = 0, False, self.env.reset()
            
            while not done:
                # Get a random action from the env
                action = self.env.action_space.sample()

                # Take the step in the env
                _, reward, done = self.env.step(action)

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
        average = int(average / self.episodes)
        print(f"Episodes average: {average}")

        # Get the best score
        best_episode = episode_dict[0]
        for episode in episode_dict:
            if episode_dict[episode]["score"] > best_episode["score"]:
                best_episode = episode_dict[episode]

        print("\n[!] Best episode score: {}".format(best_episode["score"]))

        # Save the best score gif
        best_episode["env"].render_gif()

RandomActionsModel()