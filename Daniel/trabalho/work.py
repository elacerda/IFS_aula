# %matplotlib inline
import numpy as np
from pycasso import fitsQ3DataCube
from scipy import constants as c
from scipy.optimize import minimize

def gauss(p, x):
    A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3 = p
    g1 = A1 * np.exp(-(x - mu1) ** 2 / (2. * sigma1 ** 2))
    g2 = A2 * np.exp(-(x - mu2) ** 2 / (2. * sigma2 ** 2))
    g3 = A3 * np.exp(-(x - mu3) ** 2 / (2. * sigma3 ** 2))
    return g1 + g2 + g3


def adjust_N2Ha(l_obs, f_res, constrains, x0, bounds):
    to_min = lambda p: np.square(f_res - gauss(p, l_obs)).sum()
    return minimize(to_min, x0=x0, constraints=cons, method='SLSQP', bounds=bounds, options=dict(ftol=1e-8, eps=1e-5))

# load CALIFA 10 PyCASSO Voronoi SN20 cube
K = fitsQ3DataCube('K0010_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits')
K.loadEmLinesDataCube('K0010_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.EML.MC100.fits')

iN2e = K.EL.lines.index('6548')
iHa = K.EL.lines.index('6563')
iN2d = K.EL.lines.index('6583')

'''
x0 is an array with A1, lambda1, sigma1, A2, lambda2, sigma2, A3, lambda3, sigma3
initial values. They will be fitted.
'''
x0 = [1., 6548, 1., 1., 6563, 1., 3., 6583, 6583./6548.]
d_x0i = {
    'A1':0, 'lambda1':1, 'sigma1':2,
    'A2':3, 'lambda2':4, 'sigma2':5,
    'A3':6, 'lambda3':7, 'sigma3':8,
}
bounds = [
    [0,100], [6548-10, 6548+10], [1.65, 11],
    [0,100], [6563-10, 6563+10], [1.65, 11],
    [0,100], [6583-10, 6583+10], [1.65, 11],
]
# creating some constrains between measured lines
cons = (
    {'type': 'eq', 'fun' : lambda x: np.array((x[d_x0i['lambda1']] - 6548.)/6548. - (x[d_x0i['lambda3']] - 6583.)/6583.)},
    {'type': 'eq', 'fun' : lambda x: np.array(x[d_x0i['sigma1']]/x[d_x0i['lambda1']] - x[d_x0i['sigma3']]/x[d_x0i['lambda3']])},
    {'type': 'eq', 'fun' : lambda x: np.array(x[d_x0i['sigma2']]/x[d_x0i['lambda2']] - x[d_x0i['sigma3']]/x[d_x0i['lambda3']])},
    {'type': 'eq', 'fun' : lambda x: np.array(3*x[d_x0i['A1']] - x[d_x0i['A3']])}
)

# let's adjust all N2Ha for all spaxels
minimize__z = []
ampl__Lz = {'6548':np.zeros((K.N_zone)), '6563':np.zeros((K.N_zone)), '6583':np.zeros((K.N_zone))}
pos__Lz = {'6548':np.zeros((K.N_zone)), '6563':np.zeros((K.N_zone)), '6583':np.zeros((K.N_zone))}
sigma__Lz = {'6548':np.zeros((K.N_zone)), '6563':np.zeros((K.N_zone)), '6583':np.zeros((K.N_zone))}
N2Ha_window = np.bitwise_and(np.greater(K.l_obs, 6563-100), np.less(K.l_obs, 6563+100))
contin = []
for z in xrange(K.N_zone):
    l_obs = K.l_obs[N2Ha_window]
    f_res = K.f_obs[N2Ha_window, z] - K.f_syn[N2Ha_window, z]

    f_res *= 1e16

    blue_window = np.bitwise_and(np.greater(l_obs, 6525), np.less(l_obs, 6540))
    red_window = np.bitwise_and(np.greater(l_obs, 6591), np.less(l_obs, 6606))
    x = np.array([6525 + (6540 - 6525)/2., 6591 + (6606 - 6591)/2.])
    y = np.array([np.median(f_res[blue_window]), np.median(f_res[red_window])])
    p = np.polyfit(x, y, 1)
    contin.append(p)
    f_res = f_res - np.polyval(p, l_obs)

    result = adjust_N2Ha(l_obs, f_res, cons, x0, bounds)
    ampl__Lz['6548'][z] = result.x[d_x0i['A1']]
    pos__Lz['6548'][z] = result.x[d_x0i['lambda1']]
    sigma__Lz['6548'][z] = result.x[d_x0i['sigma1']]
    ampl__Lz['6563'][z] = result.x[d_x0i['A2']]
    pos__Lz['6563'][z] = result.x[d_x0i['lambda2']]
    sigma__Lz['6563'][z] = result.x[d_x0i['sigma2']]
    ampl__Lz['6583'][z] = result.x[d_x0i['A3']]
    pos__Lz['6583'][z] = result.x[d_x0i['lambda3']]
    sigma__Lz['6583'][z] = result.x[d_x0i['sigma3']]
    minimize__z.append(result)
