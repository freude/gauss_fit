import numpy as np
import myconsts


class CoordSys(object):
    """
    This class implements the coordinate grid with given number of crystall cells and points within each cell.
    """

    def __init__(self, num_cells, arrays_sizes, units):

        self.num_cells = num_cells
        self.arrays_sizes = arrays_sizes
        self.origin_cells = 1

        if units == 'au':
            self.lattice_const = myconsts.lc/myconsts.ab
        else:
            self.lattice_const = myconsts.lc

        self.num_cells = num_cells
        self.arrays_sizes = arrays_sizes
        self.coord_sizes = self.num_cells*self.arrays_sizes

        dist=self.num_cells*self.lattice_const
        self.coord_stps = dist/self.coord_sizes
        self.origin_inds = self.arrays_sizes*(self.origin_cells-1)+1

        self.coord_limits = [0, 0]

        self.coord_limits[0] = -self.origin_inds*self.coord_stps
        self.coord_limits[1] = -self.origin_inds*self.coord_stps+self.num_cells*self.lattice_const

    def set_origin_cells(self, origin_cells):

        self.origin_cells = origin_cells
        self.origin_inds=self.arrays_sizes*(self.origin_cells-1)+1
        self.coord_limits[0] = -self.origin_inds*self.coord_stps+self.coord_stps
        self.coord_limits[1] = -self.origin_inds*self.coord_stps+self.num_cells*self.lattice_const

    def x(self):
        return np.linspace(self.coord_limits[0], self.coord_limits[1], self.coord_sizes, endpoint=True)

    def x_ep(self):
        return np.linspace(self.coord_limits[0], self.coord_limits[1], self.coord_sizes+1, endpoint=False)
