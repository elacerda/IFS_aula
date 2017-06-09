from pyraf import iraf
import numpy as np


if __name__ == '__main__':

    iraf.gemini()
    iraf.gmos()
    
    star = 'S20100226S0135.fits'
    flat_star = 'S20100226S0136.fits'
    bias_star = 'gS20100225S0152_bias.fits'
    arc_star = 'S20100226S0252.fits'
    
    gal = ['S20100308S{:04d}.fits'.format(i) for i in [25,27,29]]
    flat = 'S20100308S0023.fits'
    bias = 'gS20100308S0093_bias.fits'
    arc = 'S20100308S0053.fits'
    
    rawdir = '../raw/'
    
    # Reduce the flat
    iraf.gsflat(
        rawdir + flat, specflat='flat.fits', rawpath=rawdir,
        bias=rawdir + bias, fl_over='no')
    iraf.gsflat(
        rawdir + flat_star, specflat='flat_star.fits', rawpath=rawdir,
        bias=rawdir + bias_star, fl_over='no')   
    
    # Reduce the arc image
    iraf.gsreduce(
        rawdir + arc, rawpath=rawdir, bias=rawdir + bias,
        fl_fixpix='yes', flat='flat.fits', fl_over='no')
    iraf.gsreduce(
        rawdir + arc_star, rawpath=rawdir, bias=rawdir + bias_star,
        fl_fixpix='yes', flat='flat_star.fits', fl_over='no')
    
    # Fit wavelength solution
    iraf.gswavelength('gs' + arc, fl_inter='no')
    iraf.gswavelength('gs' + arc_star, fl_inter='no')

    # Reduce standard star 
    iraf.gsreduce(
        rawdir + star, rawpath=rawdir, bias=rawdir + bias_star,
        fl_fixpix='yes', flat='flat_star.fits', fl_over='no',
        fl_gscrrej='yes')
    iraf.gstransform(
        'gs' + star, wavtraname='gs' + arc_star.replace('.fits', ''))   
    iraf.gsskysub(
        'tgs' + star, long_sample='150:200')
    iraf.gsextract(
        'stgs' + star)
    iraf.gsstandard(
        'estgs' + star, caldir='onedstds$ctionewcal/', starname='l3218')   
    
    for i in gal:
        iraf.gsreduce(
            rawdir + i, rawpath=rawdir, bias=rawdir + bias,
            fl_fixpix='yes', flat='flat.fits', fl_over='no',
            fl_gscrrej='yes')
        iraf.gstransform(
            'gs' + i, wavtraname='gs' + arc.replace('.fits', ''))
        iraf.gsskysub(
            'tgs' + i, long_sample='800:900')
        iraf.gscalibrate(
            'stgs' + i)

    a = iraf.hselect(
        (3*'cstgs{:s}[0],').format(*gal), 'xoffset,yoffset', 'yes', Stdout=1)
    
    pixscale = 0.292  # arcsec/pix along the direction of the slit
    b = np.array([[float(j) for j in i.split('\t')] for i in a]) / pixscale
    np.savetxt('offsets.dat', b)

    iraf.imcombine(
        (3*'cstgs{:s}[2],').format(*gal), 'ngc3081_545.fits',
        offsets='offsets.dat', combine='average', reject='sigclip')

