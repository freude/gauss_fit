import math

a_Si = 0.5431 * 1e-9

h = 1.05e-34
q = 1.6e-19
m0 = 9.11e-31
eps0 = 8.85e-12

# ----------------------silicon material parameters - -----------------------

mper = 0.191 * 9.11e-31
mpar = 0.916 * 9.11e-31
eps1 = 11.4
k0 = 0.85 * 2 * math.pi / (0.543 * 1e-9)
ab = 4 * math.pi * 1.05e-34 / 1.6e-19 * 1.05e-34 / 1.6e-19 * 11.4 * 8.85e-12 / (0.19 * 9.11e-31)
E_Har = 1.6e-19 * 1.6e-19 /\
        (4 * math.pi * 1.05e-34 / 1.6e-19 * 1.05e-34 / 1.6e-19 * 11.4 * 8.85e-12 / (0.19 * 9.11e-31)) /\
        4 / math.pi / 8.85e-12 / 11.4
