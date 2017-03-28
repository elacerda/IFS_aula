import sys
import numpy as np
import matplotlib as mpl
from pycasso import fitsQ3DataCube
from matplotlib import pyplot as plt


mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
rbinini = 0.
rbinfin = 3.
rbinstep = 0.2
R_bin__r = np.arange(rbinini, rbinfin + rbinstep, rbinstep)
R_bin_center__r = (R_bin__r[:-1] + R_bin__r[1:]) / 2.0
N_R_bins = len(R_bin_center__r)
transp_choice = False
dpi_choice = 100
dflt_kw_imshow = dict(origin='lower', interpolation='nearest', aspect='equal')


def plot_spectra(wl, F_bin, F_zones, rbin, K, sel_zones):
    from pytu.plots import plot_histo_ax
    import matplotlib.gridspec as gridspec
    from CALIFAUtils.plots import DrawHLRCircle
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    califaID = K.califaID
    line_window_edges = 25
    N2Ha_window = np.bitwise_and(np.greater(wl, 6563-(2.*line_window_edges)), np.less(wl, 6563+(2.*line_window_edges)))
    Hb_window = np.bitwise_and(np.greater(wl, 4861-line_window_edges), np.less(wl, 4861+line_window_edges))
    O3_window = np.bitwise_and(np.greater(wl, 5007-line_window_edges), np.less(wl, 5007+line_window_edges))
    N_cols, N_rows = 5, 2
    f = plt.figure(figsize=(N_cols * 5, N_rows * 5))
    gs = gridspec.GridSpec(N_rows, N_cols)
    ax_spectra = plt.subplot(gs[0, :])
    ax_map_v_0 = plt.subplot(gs[1, 0])
    ax_hist_v_0 = plt.subplot(gs[1, 1])
    ax_Hb = plt.subplot(gs[1, 2])
    ax_O3 = plt.subplot(gs[1, 3])
    ax_N2Ha = plt.subplot(gs[1, 4])
    ax_spectra.plot(wl, F_zones, '-g', alpha=0.3, lw=0.3)
    ax_spectra.plot(wl, F_bin, '-k', lw=1)
    ax_Hb.plot(wl[Hb_window], F_zones[Hb_window, :], '-g', alpha=0.3, lw=0.3)
    ax_Hb.plot(wl[Hb_window], F_bin[Hb_window], '-k', lw=1)
    ax_O3.plot(wl[O3_window], F_zones[O3_window, :], '-g', alpha=0.3, lw=0.3)
    ax_O3.plot(wl[O3_window], F_bin[O3_window], '-k', lw=1)
    ax_N2Ha.plot(wl[N2Ha_window], F_zones[N2Ha_window, :], '-g', alpha=0.3, lw=0.3)
    ax_N2Ha.plot(wl[N2Ha_window], F_bin[N2Ha_window], '-k', lw=1)
    ax_spectra.xaxis.set_major_locator(MultipleLocator(250))
    ax_spectra.xaxis.set_minor_locator(MultipleLocator(50))
    ax_Hb.set_title(r'H$\beta$', y=1.1)
    ax_O3.set_title(r'[OIII]', y=1.1)
    ax_N2Ha.set_title(r'[NII] and H$\alpha$', y=1.1)
    ax_Hb.xaxis.set_major_locator(MultipleLocator(25))
    ax_Hb.xaxis.set_minor_locator(MultipleLocator(5))
    ax_O3.xaxis.set_major_locator(MultipleLocator(25))
    ax_O3.xaxis.set_minor_locator(MultipleLocator(5))
    ax_N2Ha.xaxis.set_major_locator(MultipleLocator(50))
    ax_N2Ha.xaxis.set_minor_locator(MultipleLocator(10))
    ax_spectra.grid()
    ax_Hb.grid(which='both')
    ax_O3.grid(which='both')
    ax_N2Ha.grid(which='both')
    # v_0 map & histogram
    v_0_range = [K.v_0.min(), K.v_0.max()]
    plot_histo_ax(ax_hist_v_0, K.v_0[sel_zones], y_v_space=0.06, c='k', first=True, kwargs_histo=dict(normed=False, range=v_0_range))
    v_0__yx = K.zoneToYX(np.ma.masked_array(K.v_0, mask=~sel_zones), extensive=False)
    im = ax_map_v_0.imshow(v_0__yx, vmin=v_0_range[0], vmax=v_0_range[1], cmap='RdBu', **dflt_kw_imshow)
    the_divider = make_axes_locatable(ax_map_v_0)
    color_axis = the_divider.append_axes('right', size='5%', pad=0)
    cb = plt.colorbar(im, cax=color_axis)
    cb.set_label(r'v${}_0$ [km/s]')
    DrawHLRCircle(ax_map_v_0, a=K.HLR_pix, pa=K.pa, ba=K.ba, x0=K.x0, y0=K.y0, color='k', lw=1, bins=[0.5, 1, 1.5, 2, 2.5, 3])
    ax_map_v_0.set_title(r'v${}_0$ map')
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.suptitle(r'%s - R bin center: %.2f ($\pm$ %.2f) HLR - %d zones' % (califaID, rbin, rbinstep/2., F_zones.shape[-1]))
    f.savefig('%s_spectra_%.2fHLR.png' % (califaID, rbin), dpi=dpi_choice, transparent=transp_choice)
    plt.close(f)


def doppler_resample_spec(lorig, v_0, Fobs__l):
    from astropy import constants as const
    from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform
    # doppler factor to correct wavelength
    dopp_fact = (1.0 + v_0 / const.c.to('km/s').value)
    # resample matrix
    R = ReSamplingMatrixNonUniform(lorig=lorig / dopp_fact, lresam=lorig)
    return np.tensordot(R, Fobs__l * dopp_fact, (1, 0))


if __name__ == '__main__':
    K = fitsQ3DataCube(sys.argv[1])
    # K.EL = K.loadEmLinesDataCube(argv[2])
    # Set geometry
    K.setGeometry(*K.getEllipseParams())
    wl_of = K.l_obs
    O_rf__lR = np.ma.masked_all((K.Nl_obs, N_R_bins), dtype='float')
    # Loop in radial bins
    for iR, (ledge, redge) in enumerate(zip(R_bin__r[0:-1], R_bin__r[1:])):
        sel_zones = np.bitwise_and(np.greater_equal(K.zoneDistance_HLR, ledge), np.less(K.zoneDistance_HLR, redge))
        Nsel = sel_zones.astype('int').sum()
        if Nsel == 0:  # don't do anything with empty bins
            continue
        O_of__lz = K.f_obs[:, sel_zones]
        v_0__z = K.v_0[sel_zones]
        O_rf__lz = np.ma.zeros((K.Nl_obs, sel_zones.astype('int').sum()), dtype='float')
        for iz in range(Nsel):
            #  bring all spectra local rest-frame
            # TODO: resample f_flag and f_err also!
            O_rf__lz[:, iz] = doppler_resample_spec(wl_of, v_0__z[iz], O_of__lz[:, iz])
        O_rf__lR[:, iR] = O_rf__lz.mean(axis=1)
        plot_spectra(wl_of, O_rf__lR[:, iR], O_rf__lz, R_bin_center__r[iR], K, sel_zones)
