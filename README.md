# Snake ML.
 Applying machine learning (Q Learning) to the widely known Snake game


# Planning.
The plan currently is to have 1 q learning table that will take this state as input:
Snake direction, food direction and distance to dangers in the 5 directions (Left, Diagonal left/up, ahead, diagonal right/up and right)

During training it will get the action values from the q learning table and grab the highest valued action to then use that action in the env.

The new q learning value will be calculated and adjusted into the table.
It will repeat this process N amount of times and I will document my findings

# Tasks.
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
   
   - To fix this I multiply the reward by -2 so it's a higher penalty when moving away than the reward when moving closer.

* Simple mistakes
   In later games when the model has been training a few thousand times, I notice the model still making simple mistakes like turning into itself even though it knows there's danger there.
   I suspect this is because of if that specific state had not been seen then it has no clue on what is a good or bad move in that situation.
   
   - To fix this I'm going to be adding a second smaller q table that will take the directions of the normal state and  check if there are immediate dangers 1 cell ahead/around itself
   (Will call this ID (Immediate Danger))

# Showcase
These gifs that will be shown will be the best game of 50 played.
* Random Actions
<img src="/gifs/random_actions.gif" width="450" height="300"/>
