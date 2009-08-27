from __future__ import division
from numpy import array
from numpy import arange
import itertools

class Cells:
    def get_cell_at_position(self, position):
        return tuple((position // self.cell_size) %
                     self.cells_per_edge)
    
    def __init__(self, min_cell_size, min_coord, max_coord):
        edge_length = max_coord - min_coord
        self.cells_per_edge = int(edge_length // min_cell_size)
        self.cell_size = edge_length / self.cells_per_edge
        
        index_range = range(self.cells_per_edge)
        self.neighbour_indices = [cell+array([-1,0,1])
                                  for cell in index_range]
        self.neighbour_indices[0][0] = index_range[-1]
        self.neighbour_indices[-1][-1] = index_range[0]
    
    def get_neighbouring_cells_of_cell(self, cell):
        return list(itertools.product(
                *[self.neighbour_indices[int(component)]
                  for component in cell]))
    
    def get_neighbouring_cells_at_position(self, position):
        return self.get_neighbouring_cells_of_cell(
                self.get_cell_at_position(position))
    
    def distribute_positions(self, positions):
        self.positions = {}
        for pos in positions:
            cell = self.get_cell_at_position(pos)
            try:
                self.positions[cell] = self.positions[cell]
            except KeyError:
                self.positions[cell] = []
            self.positions[cell].append(pos)
    
    def get_near_positions(self, position):
        near_positions = []
        for cell in self.get_neighbouring_cells_at_position(position):
            try:
                near_positions += self.positions[cell]
            except KeyError:
                ## cell contains no positions, so nothing to append
                pass
            except AttributeError:
                print "You might have to run `distribute_positions(positions)' first."
                raise
        return near_positions
