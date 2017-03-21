import sys
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy import constants as const


def phi(Mstar, t, burst_ini, burst_lenght):
    Tu = t[-1]
    A = Mstar/(burst_lenght * (1.-np.exp((burst_ini - Tu)/burst_lenght)))
    dMdt = A * np.exp(-(t - burst_ini)/burst_lenght)
    sel = np.less(t, burst_ini)
    dMdt[sel] = 0.
    return dMdt


def deltaM_integral(Mstar, tU, burst_ini, burst_lenght, tleft, tright):
    A = Mstar/(burst_lenght * (1.-np.exp((burst_ini - tU)/burst_lenght)))
    if tright > tU:
        tright = tU
    if tright < tleft:
        return 0.0
    return A * burst_lenght * (np.exp(-(burst_ini - tright)/burst_lenght) - np.exp(-(burst_ini - tleft)/burst_lenght))


def get_wl_of(header):
    from astropy import wcs
    Nl = header['NAXIS3']
    w = wcs.WCS(header).sub([3])
    l_obs = w.wcs_pix2world(np.arange(Nl), 0)[0]
    return l_obs


def plot_example_1(wl, flux, error, x, y, filename='example1.png'):
    f = plt.figure()
    ax = f.gca()
    ax.plot(wl, flux[:, y, x], 'b-', label='flux')
    #ax.plot(wl, 10 * error[:, y, x], 'r-', label='error x 10')
    ax.set_xlabel(r'$\lambda\ [\mathrm{\AA}]$')
    ax.set_ylabel(r'$F_\lambda\ [\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}]$')
    ax.legend()
    f.savefig('teste.png')
    plt.close(f)


def main_EW(argv):
    # reading FITS
    f = fits.open(argv[1])
    h = f[0].header
    print 'CALIFAID: %s [%s]' % (h['CALIFAID'], h['OBJECT'])
    # observed and rest-frame wavelength
    Nx = h['NAXIS1']
    Ny = h['NAXIS2']
    wl_of = get_wl_of(h)
    v = h['MED_VEL']
    z = v / c
    print 'Redshift: %.3f' % z
    wl_rf = wl_of / (1.0 + z)
    # observed spectra
    flux_unit = 1e-16  # erg/s/cm2/AA
    badpix_mask = np.greater(f['badpix'].data, 0)
    flux_of__lyx = np.ma.array(f['primary'].data, mask=badpix_mask, copy=True)
    flux_of__lyx *= (flux_unit * (1.0 + z))
    error_of__lyx = np.ma.array(f['error'].data, mask=badpix_mask, copy=True)
    error_of__lyx *= (flux_unit * (1.0 + z))
    # resampling spectra to the galaxy rest-frame
    flux_rf__lyx = np.ma.masked_all(flux_of__lyx.shape)
    error_rf__lyx = np.ma.masked_all(flux_of__lyx.shape)
    # print wl_of, wl_rf
    # for x in range(Nx):
        # for y in range(Ny):
            # flux_rf__lyx[:, y, x] = np.interp(wl_of, wl_rf, flux_of__lyx.data[:, y, x])
            # error_rf__lyx[:, y, x] = np.interp(wl_of, wl_rf, error_of__lyx.data[:, y, x])
            # print flux_of__lyx[200, y, x], flux_rf__lyx[200, y, x]
    # plot_example_1(wl_of, flux_of__lyx, error_of__lyx, 32, 32)
    # plot_example_1(wl_rf, flux_rf__lyx, error_rf__lyx, 32, 32)


def main_CSP(argv):
    from pystarlight.util.base import StarlightBase
    base_select = 'Padova2000.salp'
    base_file = '/Users/lacerda/LOCAL/data/Base.bc03.h5'
    base = StarlightBase(base_file, base_select, hdf5=True)
    M = 1e10
    tau = np.logspace(9, 11, 20)
    t0 = np.logspace(7, 10, 20)
    t = base.ageBase
    steps = (t[1:] - t[:-1])/2.0
    t_edges = np.empty((len(t) + 1))
    t_edges[1:-1] = t[1:] - steps
    t_edges[0] = t[0] - steps[0]
    if t_edges[0] < 0:
        t_edges[0] = 0
    t_edges[-1] = t[-1] + steps[0]
    if t_edges[-1] > t[-1]:
        t_edges[-1] = t[-1]
    timeline = t.max() - t[::-1]
    t_edges = t_edges[::-1]
    dM = []
    for i, (l, r) in enumerate(zip(t_edges[0:-1], t_edges[1:])):
        print l, timeline[i], r
        dM.append(deltaM_integral(M, timeline[i], 0, 1e9, l, r))
    dM = np.array(dM)
    F__l = base.f_ssp.sum(axis=0) * dM[::-1, np.newaxis]
    sel = np.bitwise_and(np.greater_equal(base.l_ssp, 3600), np.less(base.l_ssp, 7000))

if __name__ == '__main__':
    # main_EW(sys.argv)
    main_SSP(sys.argv)




























f
