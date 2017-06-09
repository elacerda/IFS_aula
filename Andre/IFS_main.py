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



def deltaM_integral(Mstar, tU, burst_ini, burst_lenght, tleft, tright):
    A = Mstar / (burst_lenght * (1. - np.exp((burst_ini - tU)/burst_lenght)))
    if tleft < burst_ini:
        tleft = burst_ini
    if tright > tU:
        tright = tU
    if tright < tleft:
        return 0.0
    return A * burst_lenght * (np.exp((burst_ini - tleft)/burst_lenght) - np.exp((burst_ini - tright)/burst_lenght))


def main_CSP(argv):
    from pystarlight.util.base import StarlightBase
    from pystarlight.util.redenninglaws import calc_redlaw
    base_select = 'Padova2000.salp'
    base_file = '/Users/lacerda/LOCAL/data/Base.bc03.h5'
    base = StarlightBase(base_file, base_select, hdf5=True)
    M = 1e10
    tauV = np.linspace(0, 1, 10)
    t0 = np.linspace(0, 20e9, 11)
    tau = np.logspace(9, 11, 12)
    q = calc_redlaw(base.l_ssp, R_V=3.1, redlaw='CCM')
    t = base.ageBase.max() - base.ageBase[::-1]
    steps = (t[1:] - t[:-1])/2.0
    t_edges = np.empty((len(t) + 1))
    t_edges[1:-1] = t[1:] - steps
    t_edges[0] = t[0] - steps[0]
    if t_edges[0] < 0:
        t_edges[0] = 0.
    t_edges[-1] = t[-1] + steps[-1]
    if t_edges[-1] > t[-1]:
        t_edges[-1] = t[-1]
    # shape_model = (len(t0), len(tau), len(tauV))
    M__t0tautauVl = np.ndarray((len(t0), len(tau), len(tauV), len(base.l_ssp)), dtype='float')
    dM__t0taut = np.ndarray((len(t0), len(tau), base.nAges), dtype='float')
    for it0, _t0 in enumerate(t0):
        for itau, _tau in enumerate(tau):
            dM = []
            for i in range(len(t)):
                l, r = t_edges[i], t_edges[i+1]
                dM.append(deltaM_integral(M, t.max(), _t0, _tau, l, r))
            # print 't0:', _t0,' tau:', _tau, ' dM:', dM[-1]
            dM = np.array(dM)
            dM__t0taut[it0, itau, :] = dM
            for itauV, _tV in enumerate(tauV):
                norm_lambda = 5635.0
                spec_window = np.bitwise_and(np.greater(base.l_ssp, norm_lambda - 45.0), np.less(base.l_ssp, norm_lambda + 45.0))
                # spec_window = (base.l_ssp > norm_lambda - 45.0) & (base.l_ssp < norm_lambda + 45.0)
                spec = np.tensordot(base.f_ssp.sum(axis=0) * np.exp(_tV * q), dM[::-1], (0, 0))
                spec_norm = np.median(spec[spec_window])
                M__t0tautauVl[it0, itau, itauV, :] = spec/spec_norm
    # Save FITS
    from astropy.io import fits
    hdu = fits.HDUList()
    header = fits.Header()
    header['NL'] = len(base.l_ssp)
    header['NBURSTINI'] = len(t0)
    header['NBURSTLENGTH'] = len(tau)
    header['NTAUV'] = len(tauV)
    hdu.append(fits.PrimaryHDU(header=header))
    hdu.append(fits.ImageHDU(data=t, name='ages'))
    hdu.append(fits.ImageHDU(data=t_edges, name='ages_edges'))
    hdu.append(fits.ImageHDU(data=t0, name='burst_ini'))
    hdu.append(fits.ImageHDU(data=tau, name='burst_length'))
    hdu.append(fits.ImageHDU(data=tauV, name='tau_V'))
    tmp = fits.ImageHDU(data=dM__t0taut, name='dM')
    tmp.header['COMMENT'] = 'Dimension [BURSTINI,BURSTLENGTH,AGES]'
    hdu.append(tmp)
    tmp = fits.ImageHDU(data=M__t0tautauVl, name='F_CSP')
    tmp.header['COMMENT'] = 'Dimension [BURSTINI,BURSTLENGTH,TAUV,LAMBDA]'
    hdu.append(tmp)
    hdu.append(fits.ImageHDU(data=base.l_ssp, name='L_CSP'))
    hdu.writeto('CSPModels.fits')


def doppler_resample_spec(lorig, v_0, Fobs__l, R=None):
    from astropy import constants as const
    from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform
    # doppler factor to correct wavelength
    dopp_fact = (1.0 + v_0 / const.c.to('km/s').value)
    # resample matrix
    if R is None:
        R = ReSamplingMatrixNonUniform(lorig=lorig / dopp_fact, lresam=lorig)
    return R, np.tensordot(R, Fobs__l * dopp_fact, (1, 0))


if __name__ == '__main__':
    main_CSP(sys.argv)
    import atpy
    from pycasso import fitsQ3DataCube
    from pystarlight import io
    from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform
    K = fitsQ3DataCube('/Users/lacerda/califa/legacy/q054/superfits/Bgstf6e/K0073_synthesis_eBR_v20_q054.d22a512.ps03.k1.mE.CCM.Bgstf6e.fits')
    hdu = fits.open('CSPModels.fits')
    t = atpy.Table(maskfile='Mask.mC', type='starlight_mask')
    wl_sel = (K.l_obs >= t[0]['l_up'])
    for i in range(1, len(t)):
        if (t[i]['weight'] == 0.0):
            wl_sel = wl_sel & ((K.l_obs < t[i]['l_low']) | (K.l_obs > t[i]['l_up']))
    t0 = hdu['BURST_INI'].data
    tau = hdu['BURST_LENGTH'].data
    tauV = hdu['TAU_V'].data
    F_CSP = hdu['F_CSP'].data
    chisquare = np.zeros((len(t0), len(tau), len(tauV), K.N_zone), dtype='float')
    R = ReSamplingMatrixNonUniform(lorig=hdu['L_CSP'].data, lresam=K.l_obs)
    F_CSP_resamp = np.tensordot(F_CSP, R, (-1, 1))
    print F_CSP.shape, F_CSP_resamp.shape, chisquare.shape
    # for it0, _t0 in enumerate(t0):
    #     for itau, _tau in enumerate(tau):
    #         for itauV, _tV in enumerate(tauV):
    #             F_CSP__L = F_CSP[it0, itau, itauV, :]
    for iz in xrange(K.N_zone):
        v0, O__l, err__l, flag__l, fnorm = K.v_0[iz], K.f_obs[:, iz], K.f_err[:, iz], K.f_flag[:, iz], K.fobs_norm[iz]
        _, O_rf__l = doppler_resample_spec(K.l_obs, v0, O__l)
        sel = np.bitwise_and(wl_sel, ~(np.greater(flag__l, 0)))
        mO_rf__l = np.ma.masked_array(O_rf__l/fnorm, mask=~sel)
        # mF_CSP__l = np.ma.masked_array(F_CSP_, mask=~sel)
        mErr__l = np.ma.masked_array(err__l/fnorm)
        N = mO_rf__l.count()
        # print N, iz, mO_rf__l[np.newaxis, np.newaxis, np.newaxis, :].shape, F_CSP_resamp.shape, mErr__l[np.newaxis, np.newaxis, np.newaxis, :].shape
        chisquare[..., iz] = (((mO_rf__l[np.newaxis, np.newaxis, np.newaxis, :] - F_CSP_resamp) / mErr__l[np.newaxis, np.newaxis, np.newaxis, :]) ** 2.).sum(axis=-1) / N
        print iz, chisquare[..., iz].any(), chisquare[5, 5, 5, iz]
    hdu.append(fits.ImageHDU(data=chisquare, name='chisquare'))
    hdu.writeto('CSPModels_CHISQ.fits')
