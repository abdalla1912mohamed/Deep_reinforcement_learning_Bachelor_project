

import pygame
import numpy as np

class GridViewer(object):
    
    def __init__(self, screen_width, screen_height, grid_lines, grid_columns):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_lines = grid_lines
        self.grid_columns = grid_columns

        self.started = False

    def start(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("PathFinding")
    
        self.font = pygame.font.SysFont("Arial", size=16)
        self.screen = pygame.display.set_mode((self.screen_width + 5, self.screen_height + 5), 0, 32)
        self.surface = pygame.Surface(self.screen.get_size())
        self.surface = self.surface.convert()
        self.surface.fill((255, 255, 255))

        self.tile_w = (self.screen_width + 5) / self.grid_lines
        self.tile_h = (self.screen_height + 5) / self.grid_columns

        self.started = True

    def stop(self):
        try:
            pygame.display.quit()
            pygame.quit()
        except:
            pass

    def draw(self, grid):
        """grid = a numpy array representing a grid of int value"""

        if not self.started:
            self.start()

        self.surface.fill((0, 0, 0))
        
        for (i, j), value in np.ndenumerate(grid):
            x, y = j, i # matrix has transposed positions

            quad = self.screen_quad_position(x, y)
            color = get_color(value)

            pygame.draw.rect(self.surface, color, quad)

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def screen_quad_position(self, x, y):
        return x * self.tile_w, y * self.tile_h, self.tile_w + 1, self.tile_h + 1


COLORS = [0xFFFFFF, 0x000000, 0x00FF00, 0xFF0000, 0x0000FF, 0x333333]

def get_color(value):
    if value in range(-1, 5):
        return COLORS[value]
    return 0xFFFF00
    

