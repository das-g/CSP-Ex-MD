from __future__ import division
from numpy import array
from numpy import arange
import itertools

class Cells:
    def get_cell_at_position(self, position):
        return (position // self.cell_size) % self.cells_per_edge
    
    def __init__(self, min_cell_size, min_coord, max_coord):
        edge_length = max_coord - min_coord
        self.cells_per_edge = int(edge_length // min_cell_size)
        self.cell_size = edge_length / self.cells_per_edge
        
        index_range = range(self.cells_per_edge)
        self.neighbour_indices = [cell+array([-1,0,1])
                for cell in index_range]
        self.neighbour_indices[0][0] = index_range[-1]
        self.neighbour_indices[-1][-1] = index_range[0]
    
    def get_neighbours_of_cell(self, cell):
        return array(list(itertools.product(
                *[self.neighbour_indices[int(component)]
                for component in cell])))
    
    def get_neighbours_at_position(self, position):
        return self.get_neighbours_of_cell(
                self.get_cell_at_position(position))
