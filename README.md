# Snake ML.
 Applying machine learning (Q Learning) to the widely known Snake game


# Planning.
The plan currently is to have 1 q learning table that will take this state as input:
Snake direction, food direction and distance to dangers in the 5 directions (Left, Diagonal left/up, ahead, diagonal right/up and right)

During training it will get the action values from the q learning table and grab the highest valued action to then use that action in the env.

The new q learning value will be calculated and adjusted into the table.
It will repeat this process N amount of times and I will document my findings

# Tasks of the model.
* Quantify states <br/>
   First we quantify the states into an index that the tables will use.
   This is done by creating a seperate table that will contain all previous seen states and uses said index for the Q learning tables.

* Calculate reward <br/>
   Secondly we want to calculate the reward that the snake will receive for their action.
   The base reward is extrapolated by calculating the distance of the snake head to the food.
   An additional bonus reward will be added or removed from the base reward when the snake consumes food, dies, or wins the game.

* Training <br/>
   Next we train both Q learning tables simultaneously in the enviroment by allowing it to play with random actions a certain amount of time. We then record the state and reward and put it into our Q learning tables

* Evaluation <br/>
   Lastly I evaluate the models and record my findings.

# Findings and learning points.
* Reward/Penalty Loop <br/>
   One of the first things I noticed when training the Q Learning table was the snake being stuck in a loop,
   It basically made the same move (For instance left) for the entirety of the game. I assume this was because of the food distance reward/penalty. It received a +1 reward for moving closer to food and -1 when moving away. This made the snake continuously seek that +1 -1 reward. 
   
   - To fix this I multiply the reward by -2 instead of -1, so it's a higher penalty when moving away than the reward when moving closer.

* Simple mistakes.
   In later games when the model has been training a few thousand times, I notice the model still making simple mistakes like turning into itself even though it knows there's danger there.
   I suspect this is because of if that specific state had not been seen then it has no clue on what is a good or bad move in that situation.
   
   - To fix this I'm going to be adding a second smaller q table that will take the directions of the normal state and check if there are immediate dangers 1 cell ahead/around itself (Will call this ID (Immediate Danger)).
   The ID q learning value will be divided by half to prevent it from overpowering the main q learning table. The main and ID q learning values will be added together and then the highest value action will be picked.

   - Note: This turned out to be a great succes, with this second table it's making a lot better and consistent choices.
   The average scores are exceeding 5 times the amount with 1 table, and that also in 1/5 the amount of training games.
   (Games do take a lot longer since the agent stays alive longer)

* Upper limits of the model.
   I was curious on the limitation of what the model was so I ran multiple env's with the max game step set to an infinite amount and the upper limit that I saw was around 6500, at that point it cornered itself (Might come back and train the model further to see any more improvement)

   - Note: The high score of 6500 exceeded my expectations and I'm thoroughly satisfied with my first Q Learning model.

# Showcase
These gifs that will be shown will be the best game of 10 played with an interval of 500 training games.

* Version two, uses the main and ID Q learning table:
<p float="left">
   <img src="/gifs/new_version_gifs/game_1.gif" width="360" height="240"/>
   <img src="/gifs/new_version_gifs/game_2.gif" width="360" height="240"/>
   <img src="/gifs/new_version_gifs/game_3.gif" width="360" height="240"/>
   <img src="/gifs/new_version_gifs/game_4.gif" width="360" height="240"/>
   <img src="/gifs/new_version_gifs/game_5.gif" width="360" height="240"/>
   <img src="/gifs/new_version_gifs/game_6.gif" width="360" height="240"/>
</p>

* Version one, only uses the main Q learning table:
<p float="left">
   <img src="/gifs/old_version_gifs/game_1.gif" width="360" height="240"/>
   <img src="/gifs/old_version_gifs/game_2.gif" width="360" height="240"/>
   <img src="/gifs/old_version_gifs/game_3.gif" width="360" height="240"/>
   <img src="/gifs/old_version_gifs/game_4.gif" width="360" height="240"/>
   <img src="/gifs/old_version_gifs/game_5.gif" width="360" height="240"/>
</p>

* Random Actions
<img src="/gifs/random_actions.gif" width="360" height="240"/>
