import numpy as np
from astropy import wcs
from astropy.io import fits
from matplotlib import pyplot as plt


save_fits = True


if __name__ == '__main__':
    hdu = fits.open('ngc3081_545_vardq.fits')
    sciheader = hdu['SCI'].header
    Nl = sciheader['NAXIS1']
    Npix = sciheader['NAXIS2']
    w = wcs.WCS(sciheader).sub([1])
    wl = w.wcs_pix2world(np.arange(Nl), 0)[0]
    z = 0.008
    l_rf = wl / (1. + z)
    flux_cal = sciheader['FLUXSCAL']
    pixel_scale = hdu[0].header['PIXSCALE']  # arcsec/pix
    flux__xl = hdu['SCI'].data / flux_cal
    eflux__xl = np.sqrt(hdu['VAR'].data) / flux_cal
    flux_rf__xl = flux__xl * (1. + z)
    eflux_rf__xl = eflux__xl * (1. + z)
    SN_rf__xl = flux_rf__xl / eflux_rf__xl
    cpix = flux_rf__xl.sum(axis=1).argmax()
    flag__xl = hdu['DQ'].data
    dist__x = np.abs([(i - cpix) * pixel_scale for i in range(Npix)])
    iS = np.argsort(dist__x)
    for i in range(Npix):
        i_spectra = iS[i]
        print i, iS[i]
        with open('spectra/ngc3081_545_spec%04d_%04d.txt' % (i, i_spectra), 'w') as f:
            for i_l in range(Nl):
                f.write('%.1f    %e    %e    %e\n' % (l_rf[i_l], flux_rf__xl[i_spectra, i_l], eflux_rf__xl[i_spectra, i_l], flag__xl[i_spectra, i_l]))

    if save_fits:
        hduout = fits.HDUList()
        hduout.append(fits.PrimaryHDU(header=hdu[0].header))
        hduout.append(fits.ImageHDU(data=wl, name='L_RF'))
        hduout.append(fits.ImageHDU(data=flux_rf__xl, name='F_RF__XL'))
        hduout.append(fits.ImageHDU(data=eflux_rf__xl, name='ERR_F_RF__XL'))
        hduout.append(fits.ImageHDU(data=flag__xl, name='FLAG__XL'))
        hduout.append(fits.ImageHDU(data=dist__x, name='DIST__X'))
        hduout.writeto('ngc3081_545_vardq_extract_spectra.fits', clobber=True)
