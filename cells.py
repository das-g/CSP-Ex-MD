from numpy import array
from numpy import arange

class Cells:
    def cell(self, position):
        return position // self.cell_size
    
    def __init__(self, min_cell_size, min_coord, max_coord):
        edge_length = max_coord - min_coord
        cells_per_edge = edge_length // min_cell_size
        self.cell_size = edge_length / cells_per_edge
