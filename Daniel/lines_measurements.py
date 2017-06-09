import numpy as np
from scipy.optimize import minimize
from CALIFAUtils.scripts import read_one_cube
from matplotlib import pyplot as plt


def gauss(p, x):
    A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3 = p
    g1 = A1 * np.exp(-(x - mu1) ** 2 / (2. * sigma1 ** 2))
    g2 = A2 * np.exp(-(x - mu2) ** 2 / (2. * sigma2 ** 2))
    g3 = A3 * np.exp(-(x - mu3) ** 2 / (2. * sigma3 ** 2))
    return g1 + g2 + g3


K = read_one_cube('K0010', debug=True, config=-2)
N_zone = K.N_zone
l_obs = np.copy(K.l_obs)
f_obs__lz = np.copy(K.f_obs)
f_syn__lz = np.copy(K.f_syn)
v_0__z = np.copy(K.v_0)
v_d__z = np.copy(K.v_d)
K.close()
del K
zone = 0
f_obs__l = f_obs__lz[:, zone]
f_syn__l = f_syn__lz[:, zone]
f_res__l = f_obs__l - f_syn__l
f_res__l *= 1e16
Hb_window = np.bitwise_and(np.greater(l_obs, 6563-70), np.less(l_obs, 6563+70))
to_min = lambda p: np.square(f_res__l[Hb_window] - gauss(p, l_obs[Hb_window])).sum()
cons = (
    {'type': 'eq', 'fun' : lambda x: np.array(x[2]/x[1] - x[-1]/x[-2])},
    {'type': 'eq', 'fun' : lambda x: np.array(3*x[0] - x[6])}
)
minimize(to_min, x0=[1., 6548, 1., 1., 6563, 1., 1., 6583, 1.], constraints=cons, method='SLSQP')
