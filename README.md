# Snake ML
 Applying machine learning (Deep Reinforcement Learning) to the widely known Snake game

Plan currently is to have 2 Q Learning tables.
A large state table and a small one.

Large state would take the entire snake board and snake direction.
Small would take list of variables that holds information about position, danger, food location and snake direction.

In training when the large state is unknown in the table we look at the smaller state table to stimulate more positive actions.

- Tasks
* Quantify states
First we quantify the states into an index that the tables will use.
This is done by creating a seperate table that will contain all previous seen states and uses said index for the Q learning tables.

* Calculate reward
Secondly we want to calculate the reward that the snake will receive for their action.
The base reward is extrapolated by calculating the distance of the snake head to the food.
An additional bonus reward will be added or removed from the base reward when the snake consumes food, dies, or wins the game.

* Training
Next we train both Q learning tables simultaneously in the enviroment by allowing it to play with random actions a certain amount of time. We then record the state and reward and put it into our Q learning tables

* Evaluation
Lastly we evaluate the models and recording my findings.