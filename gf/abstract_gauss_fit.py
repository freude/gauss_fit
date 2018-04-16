"""
The module contains abstract interfaces of the classes.
The interfaces are aimed to be schemas for further classes implementations.
Following these schemas will ensure compatibility of the code with the entire project.
"""
from abc import ABCMeta, abstractmethod
import pickle
import numpy as np


class AbstractGFit(object):
    """
    The class serves to represent a wave function defined on a bunch of points as a sum of Gaussians
    using the non-linear regression technique
    """

    __metaclass__ = ABCMeta

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
            inst = super(AbstractGFit, cls).__new__(cls, *args, **kwargs)
        return inst

    def __init__(self):

        self._gf = []

    def print_info(self):

        print('\n---------------------------------------------------------------------')
        print('The fitting is provided for the function')
        print(str(self._sn) + '_' + str(self._qn))
        print('The number of primitive Gaussians is {}'.format(self._num_fu))
        print('---------------------------------------------------------------------\n')

    @abstractmethod
    def set_init_conditions(self, **kw):
        pass

    @abstractmethod
    def do_fit(self, data, x=None, y=None, z=None):
        """ The function does the fitting procedure"""
        pass

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

        return self.modelfun(x, *self._gf)

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

    def __add__(self, another_gf):
        return self.__class__(gf=self._gf + another_gf._gf)

    def __sub__(self, another_gf):

        another_gf = [-item if j % 4 == 0 else item for j, item in enumerate(another_gf._gf)]
        return self.__class__(gf=self._gf + another_gf)

    def __mul__(self, another_gf):

        new_gf = []

        for j1 in xrange(len(self._gf / 5)):
            for j2 in xrange(j1+1, len(another_gf._gf / 5)):

                pos1 = np.array(self._gf[j1], self._gf[j1+1], self._gf[j1+2])
                pos2 = np.array(another_gf._gf[j1], another_gf._gf[j1 + 1], another_gf._gf[j1 + 2])

                exp1 = 1.0 / self._gf[j1 + 3]
                exp2 = 1.0 / another_gf._gf[j1 + 3]

                amp1 = 1.0 / self._gf[j1 + 3]
                amp2 = 1.0 / another_gf._gf[j1 + 3]

                new_pos = (exp1 * pos1 + exp2 * pos2) / (exp1 + exp2)
                new_exp = exp1 + exp2
                pos = pos1 - pos2
                new_amp = amp1 * amp2 * np.exp(-(exp1*exp2/new_exp)*np.dot(pos, pos))

                new_gf.extend(list(new_pos))
                new_gf.append(1.0 / new_exp)
                new_gf.append(new_amp)

        return self.__class__(gf=new_gf)
