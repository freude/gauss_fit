"""
    The class is for fitting a grid function by a set of gaussian functions
"""
import os
import sys
from scipy.optimize import curve_fit
# -----------------------------------------------------------
import matplotlib.pyplot as plt
from invdisttree import Invdisttree
# -----------------------------------------------------------
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from aux_fun import mat2table
import pickle


class GFit(object):

    __slots__ = ('_cube',
                 '_bounds',
                 '_save',
                 '_num_fu',
                 '_sn', '_qn',
                 '_model',
                 '_init_mode',
                 '_gf',
                 '_is_fitted',
                 '_nuclei_coords',
                 '_basis_set')

    def __new__(cls, *args, **kwargs):

        if args and isinstance(args[0], str):
            psave = args[0]
        else:
            psave = None

        if psave:
            with open(psave) as f:
                inst = pickle.load(f)
            if not isinstance(inst, cls):
                raise TypeError('Unpickled object is not of type {}'.format(cls))
        else:
            inst = super(GFit, cls).__new__(cls, *args, **kwargs)
        return inst

    def __init__(self, **kw):

        # path to save fitting parameters
        self._save = kw.get('psave', '/data/users/mklymenko/work_py/mb_project/')

        self._sn = kw.get('sn', 0)           # id of the system
        self._qn = kw.get('qn', 0)           # id of the wave function
        self._cube = kw.get('cube', np.array([[-3, -3, -3], [3, 3, 3]]))
        # type of basis functions (s, px, py, pz)
        self._basis_set = kw.get('basis_set', [12, 0, 0, 0])
        self._num_fu = np.sum(self._basis_set)

        self._model = 0                      # choice of the model function
        self._init_mode = 0                  # choice of the init method

        self._save = os.path.join(self._save, str(self._sn), '_', str(self._qn), '.npy')

        self._gf = 0
        self._is_fitted = False
        self._bounds = 0
        self._nuclei_coords = None

    def print_info(self):

        print('\n---------------------------------------------------------------------')
        print('The fitting is provided for the function')
        print(str(self._sn) + '_' + str(self._qn))
        print('The number of primitive Gaussians is {}'.format(self._num_fu))
        print('---------------------------------------------------------------------\n')

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        self._bounds = bounds

    @property
    def cube(self):
        return self._cube

    @cube.setter
    def cube(self, cube_coords):
        self._cube = cube_coords

    @property
    def nuclei_coords(self):
        return self._nuclei_coords

    @nuclei_coords.setter
    def nuclei_coords(self, nuclei_coords):
        self._nuclei_coords = nuclei_coords

    def set_init_conditions(self, **kw):

        self._init_mode = kw.get('method', None)

        if self._init_mode == 'nuclei':

            if np.size(self._nuclei_coords) == 0 or self._nuclei_coords is None:
                self._nuclei_coords = kw.get('coords', None)
                if self._nuclei_coords is None:
                    raise ValueError("Please, specify the coords")

            s = np.shape(self._nuclei_coords)
            widths = kw.get('widths', None)
            amps = kw.get('amps', None)

            if widths is None:
                raise ValueError("Please, specify the widths")
            if amps is None:
                raise ValueError("Please, specify the amps")
            if np.size(widths) == 1:
                widths = [widths for _ in xrange(s[0])]
            if np.size(amps) == 1:
                amps = [amps for _ in xrange(s[0])]

            self._distr_gaussians_over_nuclei(self._nuclei_coords, widths, amps)

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

            self._bounds = (np.zeros(self._num_fu*5), np.zeros(self._num_fu*5))

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

                self._bounds[0][j * 5] = max(coords[j1, 0]-3.5, self._cube[0][0])
                self._bounds[0][j * 5 + 1] = max(coords[j1, 1] - 3.5, self._cube[0][1])
                self._bounds[0][j * 5 + 2] = max(coords[j1, 2] - 3.5, self._cube[0][2])
                self._bounds[0][j * 5 + 3] = 0.01
                self._bounds[0][j * 5 + 4] = -5.0

                self._bounds[1][j * 5] = min(coords[j1, 0] + 3.5, self._cube[1][0])
                self._bounds[1][j * 5 + 1] = min(coords[j1, 1] + 3.5, self._cube[1][1])
                self._bounds[1][j * 5 + 2] = min(coords[j1, 2] + 3.5, self._cube[1][2])
                self._bounds[1][j * 5 + 3] = 7.0
                self._bounds[1][j * 5 + 4] = 5.0

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
        """ The function does the fitting procedure"""

        self.print_info()

        if True:

            if x is not None:
                data = mat2table(x, y, z, data)

            popt, pcov = curve_fit(self.modelfun, data[:, :3].T, np.squeeze(data[:, 3:]), p0=self._gf, bounds=self._bounds, ftol=0.0000001, xtol=0.0000001)
            self._gf = popt

        else:
            sys.exit("Wrong flag in do_fit")

    # @staticmethod
    # def modelfun(x, *par):
    #     """
    #     The model function represented by a sum of
    #     the Gaussian functions with variable positions, widths and
    #     amplitudes
    #     """
    #
    #     g = np.zeros(len(x[0]))
    #
    #     for j in range(len(par)//5):
    #         x1 = par[j*5]
    #         x2 = par[j*5+1]
    #         x3 = par[j*5+2]
    #         w = par[j*5+3]
    #         a = par[j*5+4]
    #         r1 = pow((x[0]-x1), 2)+pow((x[1]-x2), 2)+pow((x[2]-x3), 2)
    #         # if ((a > 1.1) or (a < -1.1)): a=0
    #         g = g+a*np.exp(-r1/abs(w))
    #
    #     return g

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
            x1 = par[j*5]
            x2 = par[j*5+1]
            x3 = par[j*5+2]
            w = par[j*5+3]
            a = par[j*5+4]

            r1 = pow((x[0] - x1), 2) + pow((x[1] - x2), 2) + pow((x[2] - x3), 2)

            if flag >= basis_set[flag]:
                flag += 1

            if flag == 0:
                g = g+a*np.exp(-r1/abs(w))

            if flag == 1:
                g = g+a*(x[0] - x1)*np.exp(-r1/abs(w))

            if flag == 2:
                g = g+a*(x[1] - x2)*np.exp(-r1/abs(w))

            if flag == 3:
                g = g+a*(x[2] - x3)*np.exp(-r1/abs(w))

        return g

#     # ------------------------------------------------------------------------------------------------
#
#     def get_value(self,x):
#         X, F = self.read_from_file(self._sn, self._qn, self._path)
#         invdisttree = Invdisttree(X.T, F, leafsize=10, stat=1)
#         return invdisttree(x, nnear=130, eps=0, p=1)
#
#     @staticmethod
#     def resample(data, xx, yy, zz):
#
#         # xx, yy, zz = np.mgrid[-6.5:6.5:160j, 4.0:8.9:160j, -7.5:7.5:160j]
#         xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
#         xx = np.vstack((xx, yy, zz))
#
#         invdisttree = Invdisttree(data[:, :3], data[:, 3:], leafsize=10, stat=1)
#         return xx, invdisttree(xx.T, nnear=130, eps=0, p=1)
#
#     @staticmethod
#     def detect_min_max(arr):
#         """That's a very nice function detecting all local minima and maxima
#         and computing their coordinates.
#         The method is based on derivatives.
#         """
#
#         x1, x2, y1, y2, z1, z2 = GFit.coords_of_cube(arr)
#         steps = 160
#         xi, yi, zi = np.mgrid[x1:x2:100j, y1:y2:100j, z1:z2:100j]
#         _, arr = GFit.resample(arr, xi, yi, zi)
#         arr = arr.reshape(xi.shape)
#
#         return detect_peaks(xi, yi, zi, arr)
#
#     @staticmethod
#     def coords_of_cube(data):
#
#         x_min = np.min(data[:, 0])
#         x_max = np.max(data[:, 0])
#         y_min = np.min(data[:, 1])
#         y_max = np.max(data[:, 1])
#         z_min = np.min(data[:, 2])
#         z_max = np.max(data[:, 2])
#
#         return x_min, x_max, y_min, y_max, z_min, z_max
#
#     def save(self):
#         """Save Gaussian functions coefficients to the file"""
#
#         if self._save != '0':
#             p = self._save+self._path[-3:-1]+'_'+str(self._qn)+'.dat'
#             np.save(self._gf, p)
#         else:
#             sys.exit("Wrong path to save")
#
#     def modelfun1(self, x, *par):
#         """
#         The model function represented by a sum of
#         the Gaussian functions with variable positions, widths and
#         amplitudes
#         """
#
#         g = np.zeros(len(x[0]))
#
#         for j in range(len(par)//5):
#             x1 = par[j*5]
#             x2 = par[j*5+1]
#             x3 = par[j*5+2]
#             w = par[j*5+3]
#             a = par[j*5+4]
#             r1 = pow((x[0]-x1), 2)+pow((x[1]-x2), 2)+pow((x[2]-x3), 2)
#             # if ((a > 1.1) or (a < -1.1)): a=0
#             g = g+a*np.exp(-r1/abs(w))
#
#         return g
#
# # -------------------------------------------------------------------------
# # --------------Tool for extracting values of the wave function------------
# # -------------------------------------------------------------------------
#
    def get_data(self, x):
        """
        Computes the value of the wave function in points stored in
        the vector x using fitting parameters and the model functions.
        """

        if (self._model == 0):
            g = self.modelfun(x, *self._gf)

        return g

    def get_data_matrix(self, *x):
        """
        Computes the value of the wave function in points stored in
        the vector x using fitting parameters and the model functions.
        """

        coords = [j.flatten() for j in x]
        XX = np.vstack(coords)
        g = self.get_data(XX)

        return g.reshape(x[0].shape)

    def show_gf(self, x):
        """Same as show_func(self,x) but returns
        decomposed primitive Gaussian functions
        """
        g = np.zeros((len(x[0]), self._num_fu), dtype=np.float64)
        for j in range(self._num_fu):
            x1 = self._gf[j*5]
            x2 = self._gf[j*5+1]
            x3 = self._gf[j*5+2]
            w = self._gf[j*5+3]
            a = self._gf[j*5+4]
            r1 = pow((x[0]-x1), 2)+pow((x[1]-x2), 2)+pow((x[2]-x3), 2)
            g[:, j] = a*np.exp(-r1/abs(w))

        return g

# # ----------------------------------------------------------------------
#
#     @staticmethod
#     def moments(data):
#         """Returns (height, x, y, width_x, width_y)
#          the gaussian parameters of a 2D distribution by calculating its
#         moments """
#
#         data = np.absolute(data)
#         total = data.sum()
#         X = np.indices(data.shape)
#         x = (X*data).sum()/total
#         width = np.sqrt((((X-x)**2)*data).sum()/data.sum())
#         m_max = data.max()
#         m_min = data.min()
#         if np.absolute(m_max) >= np.absolute(m_min):
#             height = m_max
#         else:
#             height = m_min
#         return height, x, width

#-----------------------------------------------------------------------


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

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image

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

    return peaks_min, np.vstack((x_peaks_min, y_peaks_min, z_peaks_min)),\
           peaks_max, np.vstack((x_peaks_max, y_peaks_max, z_peaks_max))
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------


if __name__ == "__main__":

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
    num_cells = 40
    T = 2
    coorsys = CoordSys(num_cells, T, 'au')
    coorsys.set_origin_cells(num_cells / 2 + 1)
    x = coorsys.x()
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    s = X.shape
    bands = np.array([1,2,3]) - 1
    Nbands = len(bands)
    M1 = np.zeros((1, Nbands, s[0], s[1], s[2]))

    for jj1 in xrange(1):
        print(jj1)
        M1[jj1, :, :, :, :] = read_env1(X, Y, Z, bands, path, kk[2 * jj1], 0)

    wf = GFit(sn=10,
              qn=1,
              num_fu=4)
    wf.set_init_conditions(method='nuclei', coords=np.array([[0, 0, 0]]), widths=1, amps=1)
    wf.do_fit(np.squeeze(M1[0, 2, :, :, :]), x=X, y=Y, z=Z)
    ANS = wf.get_data_matrix(X, Y, Z)
    print 'hi'
