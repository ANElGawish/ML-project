import pygame


class Grid:
    DEFAULT_CELL_SIZE = 30
    DEFAULT_GRID_SIZE = 30
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GRAY = (200, 200, 200)
    LIGHT_BLUE = (173, 216, 230)
    LIGHT_RED = (255, 182, 193)

    def __init__(self, grid_size=DEFAULT_GRID_SIZE, cell_size=DEFAULT_CELL_SIZE):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_size = cell_size * grid_size

    def draw_grid(self, screen):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = self.GRAY if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1 else self.WHITE
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, self.BLACK, rect, 1)

    def highlight_cell(self, screen, position, color):
        """Highlight a specific cell on the grid."""
        x, y = position
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, color, rect)
