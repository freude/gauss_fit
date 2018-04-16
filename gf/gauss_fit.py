"""
    The class is for fitting a grid function by a set of gaussian functions
"""
from __future__ import print_function, division
import os
import sys
import logging
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import generate_binary_structure
from abstract_gauss_fit import AbstractGFit
from aux_fun import mat2table

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GFit(AbstractGFit):
    __slots__ = ('_cube',
                 '_constraints',
                 '_save',
                 '_num_fu',
                 '_id',
                 '_init_mode',
                 '_gf',
                 '_is_fitted',
                 '_basis_set')

    def __init__(self, gf=None, **kw):

        # path to save fitting parameters
        save_dir = kw.get('psave', os.getcwd())
        self._id = kw.get('id', 0)  # id of the wave function
        self._save = os.path.join(save_dir, str(self._id), '.npy')
        self._cube = kw.get('cube', np.array([[-3, -3, -3], [3, 3, 3]]))
        # type of basis functions (s, px, py, pz)
        self._basis_set = kw.get('basis_set', [10, 0, 0, 0])

        self._init_mode = 0  # choice of the init method
        self._constraints = 0

        if isinstance(gf, list):
            self._gf = gf
            self._is_fitted = True

        elif isinstance(gf, GFit):

            self._gf = gf.gf
            self._is_fitted = gf.is_fitted
            self._id = gf.id
            self._save, self._basis_set, self._init_mode = gf.get_params()
            self._cube = gf.cube
            self._constraints = gf.constraints

        else:
            self._gf = []
            self._is_fitted = False

        self._num_fu = np.sum(self._basis_set)

    def get_params(self):
        return self._save, self._basis_set, self._init_mode

    @property
    def gf(self):
        return self._gf

    @property
    def id(self):
        return self._id

    @property
    def is_fitted(self):
        return self.is_fitted

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, bounds):
        self._constraints = bounds

    @property
    def cube(self):
        return self._cube

    @cube.setter
    def cube(self, cube_coords):
        self._cube = cube_coords

    def set_init_conditions(self, **kw):

        self._init_mode = kw.get('method', None)

        if self._init_mode == 'nuclei':

            nuclei_coords = kw.get('nuclei_coords', None)
            widths = kw.get('widths', None)
            amps = kw.get('amps', None)

            if nuclei_coords is None:
                raise ValueError("Please, specify the coords")
            if widths is None:
                raise ValueError("Please, specify the widths")
            if amps is None:
                raise ValueError("Please, specify the amps")

            nuclei = np.shape(nuclei_coords)

            if np.size(widths) == 1:
                widths = [widths for _ in xrange(nuclei[0])]
            if np.size(amps) == 1:
                amps = [amps for _ in xrange(nuclei[0])]

            self._distr_gaussians_over_nuclei(nuclei_coords, widths, amps)

    # ------------------------------------------------------------------------------------------------

    def _distr_gaussians_over_nuclei(self, coords, width, amp, num_fu=None, set_bounds=True):
        """
        Distribute N Gaussian functions, N=self._num_fu, over M coordinates of nuclei, M=coords.shape[0]

        :param coords:
        :param width:
        :param amp:
        :return:
        """

        if set_bounds:
            self._constraints = (np.zeros(self._num_fu * 5), np.zeros(self._num_fu * 5))

        if num_fu is None:
            num_fu = self._num_fu

        self._gf = [0.0 for _ in xrange(num_fu * 5)]

        j1 = 0
        aaaa = 1
        for j in range(num_fu):
            if j > aaaa * coords.shape[0] - 1:
                j1 = 0
                aaaa += 1
            self._gf[j * 5] = coords[j1, 0]
            self._gf[j * 5 + 1] = coords[j1, 1]
            self._gf[j * 5 + 2] = coords[j1, 2]
            self._gf[j * 5 + 3] = width[j1]
            self._gf[j * 5 + 4] = amp[j1]

            if set_bounds:
                self._constraints[0][j * 5] = max(coords[j1, 0] - 3.5, self._cube[0][0])
                self._constraints[0][j * 5 + 1] = max(coords[j1, 1] - 3.5, self._cube[0][1])
                self._constraints[0][j * 5 + 2] = max(coords[j1, 2] - 3.5, self._cube[0][2])
                self._constraints[0][j * 5 + 3] = 0.01
                self._constraints[0][j * 5 + 4] = -5.0

                self._constraints[1][j * 5] = min(coords[j1, 0] + 3.5, self._cube[1][0])
                self._constraints[1][j * 5 + 1] = min(coords[j1, 1] + 3.5, self._cube[1][1])
                self._constraints[1][j * 5 + 2] = min(coords[j1, 2] + 3.5, self._cube[1][2])
                self._constraints[1][j * 5 + 3] = 7.0
                self._constraints[1][j * 5 + 4] = 5.0

            j1 += 1

    def _init_gaussians_from_lists(self, coords, width, amp):
        """
        Initialize Gaussian functions from lists of coordinates, widths and amplitudes

        :param coords:
        :param width:
        :param amp:
        :return:
        """

        s = np.shape(coords)
        self._num_fu = s[0]
        self._gf = [0.0 for _ in xrange(self._num_fu)]

        for j in xrange(self._num_fu):
            self._gf[j * 5] = coords[j, 0]
            self._gf[j * 5 + 1] = coords[j, 1]
            self._gf[j * 5 + 2] = coords[j, 2]
            self._gf[j * 5 + 3] = width[j]
            self._gf[j * 5 + 4] = amp[j]

    def do_fit(self, data, x=None, y=None, z=None):
        """ The function processes the fitting procedure"""

        logging.info('---------------------------------------------------------------------')
        logging.info('The fitting is provided for the function ' + str(self._id))
        logging.info('The number of primitive Gaussians is {}'.format(self._num_fu))
        logging.info('---------------------------------------------------------------------')

        if True:

            if x is not None:
                data = mat2table(x, y, z, data)

            popt, pcov = curve_fit(self.modelfun, data[:, :3].T, np.squeeze(data[:, 3:]), p0=self._gf,
                                   bounds=self._constraints, ftol=0.0000001, xtol=0.0000001)
            self._gf = popt

        else:
            sys.exit("Wrong flag in do_fit")

    def modelfun(self, x, *par):
        """
        The model function represented by a sum of
        the Gaussian functions with variable positions, widths and
        amplitudes
        """

        g = np.zeros(len(x[0]))
        basis_set = np.cumsum(self._basis_set)

        j = 0

        flag = 0

        for j in range(len(par) // 5):
            x1 = par[j * 5]
            x2 = par[j * 5 + 1]
            x3 = par[j * 5 + 2]
            w = par[j * 5 + 3]
            a = par[j * 5 + 4]

            r1 = pow((x[0] - x1), 2) + pow((x[1] - x2), 2) + pow((x[2] - x3), 2)

            if flag >= basis_set[flag]:
                flag += 1

            if flag == 0:
                g = g + a * np.exp(-r1 / abs(w))

            if flag == 1:
                g = g + a * (x[0] - x1) * np.exp(-r1 / abs(w))

            if flag == 2:
                g = g + a * (x[1] - x2) * np.exp(-r1 / abs(w))

            if flag == 3:
                g = g + a * (x[2] - x3) * np.exp(-r1 / abs(w))

        return g


def detect_peaks(x, y, z, image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(3, 3)
    a = np.zeros((5, 5, 5))
    a[2, 2, 2] = 1
    a = binary_dilation(a, structure=neighborhood).astype(a.dtype)
    neighborhood = binary_dilation(a, structure=neighborhood).astype(bool)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image

    # #local_max is a mask that contains the peaks we are
    # #looking for, but also the background.
    # #In order to isolate the peaks we must remove the background from the mask.
    #
    # #we create the mask of the background
    # background = (image==0)
    #
    # #a little technicality: we must erode the background in order to
    # #successfully subtract it form local_max, otherwise a line will
    # #appear along the background border (artifact of the local maximum filter)
    # eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    #
    # #we obtain the final mask, containing only peaks,
    # #by removing the background from the local_max mask (xor operation)
    # detected_peaks = local_max ^ eroded_background

    x_peaks_max = x[local_max]
    y_peaks_max = y[local_max]
    z_peaks_max = z[local_max]
    peaks_max = image[local_max]

    if peaks_max.any():
        norm = np.max(peaks_max)
        x_peaks_max = x_peaks_max[peaks_max > 0.3 * norm]
        y_peaks_max = y_peaks_max[peaks_max > 0.3 * norm]
        z_peaks_max = z_peaks_max[peaks_max > 0.3 * norm]
        peaks_max = peaks_max[peaks_max > 0.3 * norm]

    local_max = minimum_filter(image, footprint=neighborhood) == image

    x_peaks_min = x[local_max]
    y_peaks_min = y[local_max]
    z_peaks_min = z[local_max]
    peaks_min = image[local_max]

    if peaks_min.any():
        norm = np.min(peaks_min)
        x_peaks_min = x_peaks_min[peaks_min < 0.3 * norm]
        y_peaks_min = y_peaks_min[peaks_min < 0.3 * norm]
        z_peaks_min = z_peaks_min[peaks_min < 0.3 * norm]
        peaks_min = peaks_min[peaks_min < 0.3 * norm]

    return peaks_min, np.vstack((x_peaks_min, y_peaks_min, z_peaks_min)), \
           peaks_max, np.vstack((x_peaks_max, y_peaks_max, z_peaks_max))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from read_env import read_env1
    from coordsys import CoordSys
    import silicon_params as si

    k0 = si.k0 * si.ab

    kk = k0 * np.array([[1, 0, 0],
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                        [0, 0, -1]])

    path = '/home/mk/p_dopant/p_bulk/p_dopant_data/'

    # data preparation
    num_cells = 80
    T = 1
    coorsys = CoordSys(num_cells, T, 'au')
    coorsys.set_origin_cells(num_cells / 2 + 1)
    x = coorsys.x()
    x = x[::2]

    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    s = X.shape
    bands = np.array([1, 2, 3, 4, 5, 6, 7]) - 1
    Nbands = len(bands)
    M1 = np.zeros((1, Nbands, s[0], s[1], s[2]))

    for jj1 in xrange(1):
        print(jj1)
        M1[jj1, :, :, :, :] = read_env1(X, Y, Z, bands, path, kk[2 * jj1], 0)

    wf = GFit(basis_set=[16])

    wf.set_init_conditions(method='nuclei', nuclei_coords=np.array([[0, 0, 0]]), widths=1, amps=1)
    wf.do_fit(np.squeeze(M1[0, 5, :, :, :]), x=X, y=Y, z=Z)
    ANS = wf.get_data_matrix(X, Y, Z)

    print('hi')
