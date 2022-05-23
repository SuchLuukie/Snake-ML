# Import libraries 
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

        # Actions: UP, RIGHT, DOWN, LEFT
        self.action_space = Discrete(4)
        self.actions = {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1]
        }

        # Observastion space
        self.observation_space = Box(low=0, high=2, shape=(self.board_height, self.board_width))

        # Game max length (Amount of steps)
        self.max_game_length = 500


    def step(self, action):
        # Reward for staying alive is -1
        # (To stimulate getting to food as fast as possible)
        reward = -1

        # The direction of the snake
        direction = self.actions[action]
        
        # The new position of the snake head
        new_pos = [self.snake[0][i] + direction[i] for i in range(len(direction))]

        y, x = new_pos
        # Check if move is inside the board
        if y < 0 or y >= self.board_height or x < 0 or x >= self.board_width:
            # Illegal
            done = True
            reward = float("-inf")
            return self.board, reward, done


        # Check if the snake hits itself
        elif self.board[y][x] == 1:
            # Game over
            done = True
            reward = float("-inf")
            return self.board, reward, done


        # Check if snake is eating an apple
        elif self.board[y][x] == 2:
            # If snake has eaten an apple, don't remove tail. (So it's length increases)
            done = False
            reward += 30
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
            reward = float("inf")
            return self.board, reward, done

        # Check if game_step is equal to max_game_step
        elif self.game_step == self.max_game_length:
            done = True
            return self.board, reward, done

        # Insert the new head
        self.snake.insert(0, new_pos)

        # Update snake onto board
        for pos in self.snake:
            y, x = pos
            self.board[y][x] = 1

        # If it hasnt returned yet then the game is ready for next step
        # Finally increase game_step
        self.game_step += 1
        return self.board, reward, False

    def reset(self):
        # Create the board and snake with a random starting position
        self.board = [[0 for j in range(self.board_width)] for i in range(self.board_height)]
        self.snake = [[randint(0, self.board_height-1), randint(0, self.board_width-1)]]

        # Create the gamestep
        self.game_step = 0

        # Spawn the snake
        self.board[self.snake[0][0]][self.snake[0][1]] = 1

        # Spawn the food
        self.spawn_food()

        return self.board


    def render(self):
        color_dictionary = {
            1: (124,252,0),
            2: (139, 0, 0)
        }
        cell_size = 100

        # Create image of the correct dimension and prepare for draw
        image = Image.new("RGB", (len(self.board[0])*cell_size, len(self.board)*cell_size), (17,17,17))
        draw = ImageDraw.Draw(image)

        # Draw the food and snake
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell != 0:
                    shape = [x*cell_size, y*cell_size, x*cell_size+cell_size, y*cell_size+cell_size]
                    draw.rectangle(shape, fill=color_dictionary[cell])

        return image

    def spawn_food(self):
        # Get all open cells
        open_cells = [[y, x] for x in range(self.board_width) for y in range(self.board_height) if self.board[y][x] == 0]
        
        # If there are no open cells return False
        if len(open_cells) == 0:
            return False

        # If there are open cells, pick random one and make it food
        else:
            y, x = choice(open_cells)
            self.board[y][x] = 2
            return True