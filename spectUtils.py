import sys
import numpy as np
from pycasso import fitsQ3DataCube
from astropy import constants as const
from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform


# XXX TODO:
# def doppler_resample_spec():
# def doppler_resample_gal():


def main(argv):
    K = fitsQ3DataCube(argv[1])
    c = const.c.to('km/s').value
    zone = 10
    # observed frame wavelength
    wl_of = K.l_obs
    # starlight zo  ne velocity related to the nucleus
    v_0 = K.v_0[zone]
    # doppler factor to correct wavelength
    dopp_fact = (1.0 + v_0 / c)
    # Observed spectra
    O_of__l = K.f_obs[:, zone]
    # rest frame wavelength
    wl_rf = wl_of / dopp_fact
    # resample matrix
    R = ReSamplingMatrixNonUniform(lorig=wl_of, lresam=wl_rf)
    # Observed spectra in restframe
    O_rf__l = np.tensordot(R, O_of__l * dopp_fact, (1,0))


if __name__ == '__main__':
    main(sys.argv)
