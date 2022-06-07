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
        self.base_reward = 1
        self.food_reward = 30
        self.food_distance_penalty = -1
        self.game_over_reward = -10000
        self.game_win_reward = 10000

        # Actions: From snake direction go left, up or right
        self.action_space = Discrete(3)
        self.actions = {
            0: [-1, 0], # Up
            1: [0, 1],  # Right
            2: [1, 0],  # Down
            3: [0, -1]  # Left
        }

        # Observastion space
        self.observation_space = Box(low=0, high=2, shape=(self.board_height, self.board_width))

        # Game max length (Amount of steps)
        self.max_game_length = 500


    def step(self, action):
        # Update the gif
        self.gif.append(self.render())


        # ------ Snake direction and new pos ------
        # The direction of the snake
        directional_actions = [self.actions[i] for i in self.actions]
        del directional_actions[self.snake_direction -2]

        direction = directional_actions[action]

        # Set new snake direction
        self.snake_direction = [i for i in self.actions if self.actions[i] == direction][0]
        
        # The new position of the snake head
        new_pos = [self.snake[0][i] + direction[i] for i in range(len(direction))]


        # ------ Reward ------
        # Will fiddle with base reward
        reward = self.base_reward

        # Calculate food distance
        dy, dx = [new_pos[0] - self.food_location[0], new_pos[1] - self.food_location[1]]
        
        # Get the absolute distance (Positive)
        distance = abs(dy) + abs(dx)

        # Calculate and apply penalty
        penalty = distance * self.food_distance_penalty
        reward += penalty


        y, x = new_pos
        # Check if move is inside the board
        if y < 0 or y >= self.board_height or x < 0 or x >= self.board_width:
            # Illegal
            done = True
            reward += self.game_over_reward
            return self.states(), reward, done


        # Check if the snake hits itself
        elif self.board[y][x] == 1:
            # Game over
            done = True
            reward += self.game_over_reward
            return self.states(), reward, done


        # Check if snake is eating an apple
        elif self.board[y][x] == 2:
            # If snake has eaten an apple, don't remove tail. (So it's length increases)
            done = False
            reward += self.food_reward
            food_spawn_success = self.spawn_food()
        
        else:
            # Food spawn success is true if it didn't need to spawn food
            food_spawn_success = True

            # Else remove the tail
            tail = self.snake[-1]
            self.snake.pop()
            self.board[tail[0]][tail[1]] = 0

        # Check if the game is over by the food spawn succes
        # (If False it can't spawn food because snake takes up the whole board)
        if not food_spawn_success:
            done = True
            reward += self.game_win_reward
            return self.states(), reward, done

        # Check if game_step is equal to max_game_step
        elif self.game_step == self.max_game_length:
            done = True
            return self.states(), reward, done

        # Insert the new head
        self.snake.insert(0, new_pos)

        # Update snake onto board
        for pos in self.snake:
            y, x = pos
            self.board[y][x] = 1

        # If it hasnt returned yet then the game is ready for next step
        # Finally increase game_step
        self.game_step += 1
        return self.states(), reward, False


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

        # Reset gif
        self.gif = []

        return self.states()


    def render(self):
        color_dictionary = {
            1: (124,252,0),
            2: (139, 0, 0),
            3: (75, 150, 2)
        }
        cell_size = 100

        # Create image of the correct dimension and prepare for draw
        image = Image.new("RGB", (len(self.board[0])*cell_size, len(self.board)*cell_size), (17,17,17))
        draw = ImageDraw.Draw(image)

        # Draw the food and snake
        for y, row in enumerate(self.board):
            for x, color in enumerate(row):
                if color != 0:
                    # Check if it's the head of the snake
                    if [y, x] == self.snake[0]:
                        color = 3

                    shape = [x*cell_size, y*cell_size, x*cell_size+cell_size, y*cell_size+cell_size]
                    draw.rectangle(shape, fill=color_dictionary[color])

        return image

    def render_gif(self, location = "game.gif"):
        print("[!] Saving gif")
        imageio.mimsave(location, self.gif)
        print("[!] Gif saved")


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


    def states(self):
        large_state = [self.snake_direction, self.board]
        small_state = [self.snake_direction] + self.dangers() + self.food_direction()
        return large_state, small_state

    def dangers(self):
        dangers = []

        # The direction of the snake
        directional_actions = [self.actions[i] for i in self.actions]
        del directional_actions[self.snake_direction -2]

        # Snake head from where the danger is calculated
        snake_head = self.snake[0]

        for direction in directional_actions:
            old_pos = snake_head
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
                elif self.board[y][x] == 1:
                    # If it is occupied by the snake then it's the danger
                    dangers.append(danger_distance)
                    found_danger = True
                
                old_pos = new_pos
                danger_distance += 1

        return dangers

    def food_direction(self):
        return [[(self.snake[0][0] - self.food_location[0]) * -1, (self.snake[0][1] - self.food_location[1]) * -1]]