"""We need to synthesize the Balmer lines only in order to compute the regions where these broad lines are blend-free
 (and label those pixels as mask=1). This is not known a priori since the Balmer line strength and amount of
 metal-blends vary a lot between different stars. """
import numpy as np


# line0 is the balmer central peaks. We add the balmer lines to the line mask with widths dependent on metallicity and
# gravity as they make up cmin which the balmer line must be larger than to be added. This is older code taken
# directly from IDL and changed very little, including the variable names.
def balmer_mask(line0, sme_spectra, sme_variables, balmer_spectra):
    """
        Input:
            line0: The center of the balmer lines, given in galahsp4 or 1.
            sme_spectra: The spectra of the non-balmer run created in the Synthesize run previous. We take the synthes.
                            spectra from it to compare to the balmer spectra we just produced to compare to cmax with
                            the balmer run in "(sme_synth_array-balmer_synth_array)**2 < chimax"
            balmer_spectra: The spectra we mainly use to compare to the variables of chimax and cmin to ensure we
                            use only appropriate spectra.
        Output:
            line0: A list of the center wavelength of the balmer line. Strict values limited to given input
            line_st: The start of the balmer line. This varies more wildly due to varying inputs and ends up with
                    many similar linemasks overlapping.
            line_en: The same as line_st, but for the end of the mask.
    """
    teff = sme_variables['effective_temperature']
    grav = sme_variables['gravity']
    feh = sme_variables['metallicity']

    # Used to check for limits of the line mask later in the loop. Compare to chimax and chimin.
    sme_synth = sme_spectra['synth']
    balmer_synth = balmer_spectra['synth']

    dum1 = []
    dum2 = []
    dum3 = []
    # sme spectra is meant to be used for its synth, and with chimax for further restrictions.
    chimax = 5E-4
    eps = 1E-5
    sme_variables['ipres'] = 2500
    if sme_variables['ipres'] < 2000:
        chimax = 1E-2
    # Empirical flux-cut-off to exclude the line-cores of Balmer lines
    cmin = min([(0.6-(feh*0.04)+((5-grav)*0.09)), 0.95])
    print("Feh, grav", feh, grav)
    # How extended the mask is depends on teff
    delt = 5+(teff-3500)*5/1E3
    line_st = line0-delt
    line_en = line0+delt

    balmer_wave_list = []
    balmer_synth_list = []
    sme_synth_list = []
    # de segmentise them as this occurs after running pysme but before makestruct.
    # Will resegmentise afterwards if returned.
    # We make a desegmentised list/array of the wavelengths from the pysme run.
    for balmer_wave_value, balmer_synth_value, full_sme_synth_value in \
            zip(balmer_spectra['wave'], balmer_synth, sme_synth):
        balmer_wave_list.extend(balmer_wave_value)
        balmer_synth_list.extend(balmer_synth_value)
        sme_synth_list.extend(full_sme_synth_value)

    balmer_wave_array = np.asarray(balmer_wave_list)
    balmer_synth_array = np.asarray(balmer_synth_list)
    sme_synth_array = np.asarray(sme_synth_list)

    # For each balmer line. We are adding/extending linemask for the balmer line
    for balmer_line in range(0, len(line0)):
        # Which wavelengths are within the balmer start and end
        mob = np.where(
            np.logical_and(
                           np.logical_and(balmer_wave_array > line_st[balmer_line],
                                          balmer_wave_array < line_en[balmer_line]),
                           np.logical_and((sme_synth_array-balmer_synth_array)**2 < chimax,
                                          balmer_synth_array > cmin)
                           )
                       )
        # As long as some exist
        if len(mob[0]) != 0:
            # Remove tuple, just get the indexes
            mob = mob[0]
            # For each index value (e.g. each wavelength inside the balmer start and stop)
            for wavelength in range (0, len(mob)):
                # If it's not the first, but is in order. (E.g. is wavelength 14,15,16,17.)
                if wavelength != 0 and mob[wavelength] == mob[wavelength-1]+1:
                    # extends the linemak value until it reaches a new value that is NOT sequential (i.e. a diff. seg.)
                    dum2[len(dum2)-1] = balmer_wave_array[mob[wavelength]]+eps
                # If it's the first of a new segment selection (so each ordered collection of wavelengths)
                else:
                    dum1.append(balmer_wave_array[mob[wavelength]]-eps)
                    dum2.append(balmer_wave_array[mob[wavelength]]+eps)
                    dum3.append(line0[balmer_line])

    if len(dum3) <= 1:
        line0 = []
        line_st = []
        line_en = []
    else:
        line0 = dum3
        line_st = dum1
        line_en = dum2
    print("line0, line_st, line_en", line0, line_st, line_en)
    return line0, line_st, line_en



