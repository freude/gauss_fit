"""
The module contains abstract interfaces of the classes.
The interfaces are aimed to be schemas for further classes implementations.
Following these schemas will ensure compatibility of the code with the entire project.
"""
from abc import ABCMeta, abstractmethod
import os
import pickle
import numpy as np


class GFit(object):
    """
    The class serves to represent a wave function defined on a bunch of points as a sum of Gaussians
    using the non-linear regression technique
    """

    __metaclass__ = ABCMeta

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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def modelfun(self, x, *par):
        """
        The model function represented by a sum of
        the Gaussian functions with variable positions, widths and
        amplitudes
        """

        return 0

# -------------------------------------------------------------------------
# --------------Tool for extracting values of the wave function------------
# -------------------------------------------------------------------------

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
