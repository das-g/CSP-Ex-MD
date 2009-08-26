from __future__ import division
from numpy import array
from numpy import arange

class Cells:
    def get_cell_at_position(self, position):
        return (position // self.cell_size) % self.cells_per_edge
    
    def __init__(self, min_cell_size, min_coord, max_coord):
        edge_length = max_coord - min_coord
        self.cells_per_edge = edge_length // min_cell_size
        self.cell_size = edge_length / self.cells_per_edge
