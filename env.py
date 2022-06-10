# Import libraries
import imageio
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from PIL import Image, ImageDraw
from random import choice, randint

# Define custom snake environment
class SnakeEnv(Env):
    def __init__(self):
        # Define variables
        # Board dimensions
        self.board_width = 30
        self.board_height = 20

        # Board integers meaning:
        # 0: Air
        # 1: Snake
        # 2: Food

        # Reward table
        self.base_reward = 0
        self.food_reward = 100
        self.food_distance_reward = 5
        self.food_consume_game_step_remove = -20
        self.game_over_reward = -1000
        self.game_win_reward = 1000

        # Actions: From snake direction go left, up or right
        self.action_space = Discrete(3)
        self.actions = [
            [-1, 0], # Up
            [0, 1],  # Right
            [1, 0],  # Down
            [0, -1]  # Left
        ]
        self.diagonals = [
            [-1, 1], # Top right
            [1, 1],  # Bottom right
            [1, -1], # Bottom left
            [-1, -1]  # Top left
        ]

        # Observastion space
        self.observation_space = Box(low=0, high=2, shape=(self.board_height, self.board_width))

        # Game max length (Amount of steps)
        self.max_game_length = 150


    # Take a step in the env with the given action
    def step(self, action):
        # Update the gif
        self.gif.append(self.render())

        # ------ Snake direction and new pos ------
        # The direction of the snake (Ordered by the currents snake direction)
        directional_actions = [self.actions[self.snake_direction-1], self.actions[self.snake_direction], self.actions[self.snake_direction-3]]
        direction = directional_actions[action]

        # Set new snake direction
        self.snake_direction = [i for i in range(len(self.actions)) if self.actions[i] == direction][0]
        
        # The new position of the snake head
        new_pos = [self.snake[0][i] + direction[i] for i in range(len(direction))]


        # ------ Reward ------
        # Will fiddle with base reward
        reward = self.base_reward

        # Calculate food distance
        dy, dx = [new_pos[0] - self.food_location[0], new_pos[1] - self.food_location[1]]
        
        # Get the absolute distance (Positive)
        new_food_distance = abs(dy) + abs(dx)

        # Check if new food distance is is more or less than food distance.
        # Apply appriopriate reward/penalty
        if new_food_distance > self.food_distance:
            # Penalty
            reward += self.food_distance_reward * -2

        else:
            # Reward
            reward += self.food_distance_reward

        # Set new food distance as the food distance
        self.food_distance = new_food_distance


        y, x = new_pos
        # Check if move is inside the board
        if y < 0 or y >= self.board_height or x < 0 or x >= self.board_width:
            # Illegal
            done = True
            reward += self.game_over_reward
            return self.states() + [reward, done]


        # Check if the snake hits itself
        elif self.board[y][x] == 1:
            # Game over
            done = True
            reward += self.game_over_reward
            return self.states() + [reward, done]


        # Check if snake is eating an apple
        elif self.board[y][x] == 2:
            # If snake has eaten an apple, don't remove tail. (So it's length increases)
            done = False
            reward += self.food_reward
            food_spawn_success = self.spawn_food()
            self.game_step += self.food_consume_game_step_remove
        
        else:
            # Food spawn success is true if it didn't need to spawn food
            food_spawn_success = True

            # Else remove the tail
            tail = self.snake[-1]
            self.snake.pop()
            self.board[tail[0]][tail[1]] = 0

        # Insert the new head
        self.snake.insert(0, new_pos)
        
        # Check if the game is over by the food spawn succes
        # (If False it can't spawn food because snake takes up the whole board)
        if not food_spawn_success:
            done = True
            reward += self.game_win_reward
            return self.states() + [reward, done]

        # Check if game_step is equal to max_game_step
        if self.game_step == self.max_game_length:
            done = True
            return self.states() + [reward, done]

        # Update snake onto board
        for pos in self.snake:
            y, x = pos
            self.board[y][x] = 1

        # If it hasnt returned yet then the game is ready for next step
        # Finally increase game_step
        self.game_step += 1
        return self.states() + [reward, False]


    # Reset the env
    def reset(self):
        # Create the board and snake with a random starting position
        self.board = [[0 for j in range(self.board_width)] for i in range(self.board_height)]
        self.snake = [[10, 15]]#[[randint(3, self.board_height-4), randint(3, self.board_width-4)]] # Atleast 3 cells off any sides

        # Create the gamestep
        self.game_step = 0

        # Spawn the snake
        self.board[self.snake[0][0]][self.snake[0][1]] = 1

        # Spawn the food
        self.spawn_food()

        # Set random snake direction
        self.snake_direction = 0 #randint(0, 4)

        # Calculate food distance
        dy, dx = [self.snake[0][0] - self.food_location[0], self.snake[0][1] - self.food_location[1]]
        
        # Get the absolute distance (Positive)
        self.food_distance = abs(dy) + abs(dx)

        # Reset gif
        self.gif = []

        return self.states()


    # Render the current image of the board right now
    def render(self):
        color_dictionary = {
            1: (75, 150, 2),
            2: (139, 0, 0),
            3: (124, 252, 0)
        }
        cell_size = 100

        # Create image of the correct dimension and prepare for draw
        image = Image.new("RGB", (len(self.board[0])*cell_size, len(self.board)*cell_size), (17,17,17))
        draw = ImageDraw.Draw(image)

        # Draw the snake
        for idx, bit in enumerate(self.snake):
            color = 1
            if idx == 0:
                color = 3

            y, x = bit
            shape = [x*cell_size, y*cell_size, x*cell_size+cell_size, y*cell_size+cell_size]
            draw.rectangle(shape, fill=color_dictionary[color])

        # Draw the food
        y, x = self.food_location
        shape = [x*cell_size, y*cell_size, x*cell_size+cell_size, y*cell_size+cell_size]
        draw.rectangle(shape, fill=color_dictionary[2])

        return image


    # Uses all the images in self.gifs and creates an animated gif
    def render_gif(self, location = "game.gif"):
        print("[!] Saving gif")
        imageio.mimsave(location, self.gif)
        print("[!] Gif saved")


    # Spawn food on the board
    def spawn_food(self):
        # Get all open cells
        open_cells = [[y, x] for x in range(self.board_width) for y in range(self.board_height) if self.board[y][x] == 0]
        
        # If there are no open cells return False
        if len(open_cells) == 0:
            return False

        # If there are open cells, pick random one and make it food
        else:
            y, x = choice(open_cells)
            self.food_location = [y, x]
            self.board[y][x] = 2
            return True


    # States that the models will use
    def states(self):
        state = [self.snake_direction] + self.food_direction() + self.dangers()
        id_state = self.imminent_danger()
        return [state, id_state]


    # An array of the distance to each dangerous cell, used by the states
    def dangers(self):
        # The directions that will be checked
        # We get the cardinal directions by using our direction as index (direction-1, direction, direction-3)
        # We get the diagonals by using our direction as index (direction, direction-1)
        directions = [
            self.actions[self.snake_direction-1],
            self.diagonals[self.snake_direction-1],
            self.actions[self.snake_direction], 
            self.diagonals[self.snake_direction],
            self.actions[self.snake_direction-3]
        ]

        dangers = []
        for direction in directions:
            old_pos = self.snake[0]
            found_danger = False
            danger_distance = 0

            while not found_danger:
                new_pos = [old_pos[0] + direction[0], old_pos[1] + direction[1]]
                y, x = new_pos
                
                # Check if new pos is inside the board
                if y < 0 or y >= self.board_height or x < 0 or x >= self.board_width:
                    # Not in the board, so this is the danger
                    dangers.append(danger_distance)
                    found_danger = True

                # Check if the new pos is occupied by the snake
                elif new_pos in self.snake:
                    # If it is occupied by the snake then it's the danger
                    dangers.append(danger_distance)
                    found_danger = True

                old_pos = new_pos
                danger_distance += 1

        return dangers


    # An array of booleans that will be True if the cell directly around the snake is dangerous. Used by states
    def imminent_danger(self):
        # The directions that will be checked
        # We get the cardinal directions by using our direction as index (direction-1, direction, direction-3)
        # We get the diagonals by using our direction as index (direction, direction-1)
        directions = [
            self.actions[self.snake_direction-1],
            self.diagonals[self.snake_direction-1],
            self.actions[self.snake_direction], 
            self.diagonals[self.snake_direction],
            self.actions[self.snake_direction-3]
        ]

        # Origin will be where the danger will be check + the direction offset
        origin = self.snake[0]

        # If it is dangerous that it will be True, if not False
        dangers = []

        for direction in directions:
            location = [origin[0] + direction[0], origin[1] + direction[1]]
            y, x = location
                
            # Check if the location is inside the board
            if y < 0 or y >= self.board_height or x < 0 or x >= self.board_width:
                # Not in the board, so this is the danger
                dangers.append(True)

            # If the location is occupied by the snake then it's also a danger
            elif location in self.snake:
                dangers.append(True)

            # Otherwise it's not a danger
            else:
                dangers.append(False)

        return dangers
                

    # The direction of food, used by states
    def food_direction(self):
        y_distance = (self.snake[0][0] - self.food_location[0]) * -1
        x_distance = (self.snake[0][1] - self.food_location[1]) * -1

        if y_distance > 0:
            y = 1
        
        elif y_distance < 0:
            y = -1
        
        else:
            y = 0

        if x_distance > 0:
            x = 1

        elif x_distance < 0:
            x = -1
        
        else:
            x = 0

        return [y, x]