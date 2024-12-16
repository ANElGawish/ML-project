import pygame
import time
from Player import Chaser,Victim

from Grid import Grid
import random


class Game:
    def __init__(self, episodes=100, duration=300, grid_size=30, cell_size=30):
        self.episodes = episodes  # Number of episodes for tracking learning
        self.duration = duration  # Duration of each episode in seconds
        self.grid = Grid(grid_size=grid_size, cell_size=cell_size)
        self.screen = pygame.display.set_mode((self.grid.screen_size, self.grid.screen_size))
        pygame.display.set_caption("Chaser and Victim")
        self.clock = pygame.time.Clock()

        # Initialize Chaser and Victim dynamically within grid bounds
        self.chaser = Chaser(x=random.randint(1, self.grid.grid_size - 2), y=random.randint(1, self.grid.grid_size - 2))
        self.victim = Victim(x=random.randint(1, self.grid.grid_size - 2), y=random.randint(1, self.grid.grid_size - 2))

    def run(self):
        for episode in range(self.episodes):
            print(f"Starting Episode {episode + 1}")

            # Reset positions dynamically at the start of each episode
            self.chaser.position = [random.randint(1, self.grid.grid_size - 2),
                                    random.randint(1, self.grid.grid_size - 2)]
            self.victim.position = [random.randint(1, self.grid.grid_size - 2),
                                    random.randint(1, self.grid.grid_size - 2)]

            # Reset radar and other attributes
            self.chaser.radar_limit = 3
            self.chaser.radar_history = []

            start_time = time.time()
            steps = 0

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.chaser.save_q_table()
                        self.victim.save_q_table()
                        pygame.quit()
                        exit()
                """ keys = pygame.key.get_pressed()  # Get all pressed keys
               self.victim.move_with_keys(keys, self.grid.grid_size)  # Control victim with keys"
               self.victim.move_randomly(self.grid.grid_size)"""
                self.victim.decide_move(self.chaser.position, self.grid.grid_size)

                elapsed_time = time.time() - start_time
                if elapsed_time >= self.duration:
                    print(f"Episode {episode + 1}: Time's up! Game Over.")
                    break

                self.chaser.decide_move(self.victim.position, self.grid.grid_size)
                steps += 1

                if self.chaser.position == self.victim.position:
                    print(f"Episode {episode + 1}: Chaser caught the victim in {steps} steps!")
                    break

                self.render(self.chaser, self.victim)

            print(f"Episode {episode + 1} complete. Epsilon: {self.chaser.epsilon:.2f}")
            self.chaser.save_q_table()
            self.victim.save_q_table()

    """def render(self, chaser, victim):
        try:
            self.screen.fill((0, 0, 0))  # Clear the screen
            # Draw the grid
            self.grid.draw_grid(self.screen)

            # Highlight the chaser's radar area
            for x in range(-1, 2):
                for y in range(-1, 2):
                    radar_x, radar_y = chaser.position[0] + x, chaser.position[1] + y
                    if 0 <= radar_x < self.grid.grid_size and 0 <= radar_y < self.grid.grid_size:
                        self.grid.highlight_cell(self.screen, (radar_x, radar_y), self.grid.LIGHT_BLUE)
            for x in range(-4, 4):
                for y in range(-4, 4):
                    radar_x, radar_y = victim.position[0] + x, victim.position[1] + y
                    if 0 <= radar_x < self.grid.grid_size and 0 <= radar_y < self.grid.grid_size:
                        self.grid.highlight_cell(self.screen, (radar_x, radar_y), self.grid.LIGHT_RED)

            # Draw the players

            chaser.draw_player(self.screen, self.grid.BLUE, self.grid.cell_size)
            victim.draw_player(self.screen, self.grid.RED, self.grid.cell_size)

            pygame.display.flip()  # Update the display
            self.clock.tick(30)  # Limit to 30 FPS
        except AttributeError as e:
            print(f"AttributeError: {e}")
        except Exception as e:
            print(f"Unexpected error in render: {e}")"""
    def render(self, chaser, victim):
        """
        Render the game grid, players, and their visibility ranges.
        """
        try:
            self.screen.fill((0, 0, 0))  # Clear the screen

            # Draw the grid
            self.grid.draw_grid(self.screen)

            # Highlight the chaser's visibility area (radius = 5)
            for dx in range(-5, 6):  # Iterate within visibility radius
                for dy in range(-5, 6):
                    if abs(dx) + abs(dy) <= 5:  # Manhattan distance
                        visible_x = chaser.position[0] + dx
                        visible_y = chaser.position[1] + dy
                        if 0 <= visible_x < self.grid.grid_size and 0 <= visible_y < self.grid.grid_size:
                            self.grid.highlight_cell(self.screen, (visible_x, visible_y), self.grid.LIGHT_BLUE)

            # Highlight the victim's visibility area (radius = 12)
            for dx in range(-12, 13):  # Iterate within visibility radius
                for dy in range(-12, 13):
                    if abs(dx) + abs(dy) <= 12:  # Manhattan distance
                        visible_x = victim.position[0] + dx
                        visible_y = victim.position[1] + dy
                        if 0 <= visible_x < self.grid.grid_size and 0 <= visible_y < self.grid.grid_size:
                            self.grid.highlight_cell(self.screen, (visible_x, visible_y), self.grid.LIGHT_RED)

            # Draw the players
            chaser.draw_player(self.screen, self.grid.BLUE, self.grid.cell_size)
            victim.draw_player(self.screen, self.grid.RED, self.grid.cell_size)

            pygame.display.flip()  # Update the display
            self.clock.tick(30)  # Limit to 30 FPS
        except AttributeError as e:
            print(f"AttributeError in render: {e}")
        except Exception as e:
            print(f"Unexpected error in render: {e}")


if __name__ == "__main__":
    pygame.init()

    # Adjust these parameters as needed
    game = Game(episodes=200, duration=300, grid_size=30, cell_size=30)
    game.run()

    pygame.quit()
