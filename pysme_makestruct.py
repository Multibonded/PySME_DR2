"""
We read the segment mask segm, removing the comment lines and columns.
We proceed to open the ccd1-4 files and take their wavelength and resolution out to plot a resolution mask and
interpolating to be able to find resolution at peak wavelengths that we calculate at that point of the segment mask.
We open the 4 ccd fits files to produce a phoneenumber_Sp file that contains Wavelength array, observed flux sign (sob)
uncertainity in flux (uob)(essentially error), smod (not important now) and mod that we call in make_struct
to modify the wavelength array by the radial velocity doppler shift and make some sme variables.
Running create_structure should be the only thing needed and is called from pysme_execute
Running it for the first time will take a good 5-10 minutes, but after that it should take a few seconds at a time. It's
a trade off.
"""

import pysme_readlines
import pysme_run_sme
import pysme_interpolate_depth
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import date
import pickle
from astropy.io import fits

# We add the final products to the sme_input (makestruct dictionary) before we run pysme itself in the pysme_run_sme
# file.
def create_structure(makestruct_dict, reduction_variable_dict,
                     atomic_abundances_free_parameters=np.zeros(99), normalise_flag=False, run="Synthesise"):

    # Adjusts the accuracies ("resolution") of the spectra depending on the type of run.
    makestruct_dict = update_accuracies(makestruct_dict)

    makestruct_dict = update_atmosphere(makestruct_dict)
    # Updates the broadening profile (usually gauss) to adjust the spectra, and also the values for mu and nmu, the
    # number of angles to calculate intensity.
    makestruct_dict = update_profile(makestruct_dict)
    # Produce the basic blocks of which wavelengths etc are inside our segment masks. (The important atoms)
    # Produces an array of all wavelengths, and the indexes to find which represent the start and end of the segments,
    # and the wavelength values of those as well. Calls many functions inside.
    ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
        segment_begin_end, wavelength_start_end_index, ccd_resolution_of_segment_array = \
        object_array_setup(makestruct_dict)

    # Number of segments from our segment mask.
    number_of_segments = len(wavelength_start_end_index)
    # Opens the files containing data on the linemasks and/or continuum.
    continuum_mask_data, linemask_data = open_masks(makestruct_dict)

    # We always want an array for pysme. Sets the v_rad and cscale for PySME. Adjusted during its run.
    radial_velocity = np.zeros(number_of_segments)
    continuum_scale = np.ones(number_of_segments)

    print("Normalise flag", normalise_flag)
    # Only normalise the data if that's how this file is called.
    # Runs the normalisation run to take flux down from the 10s of thousands to below 1
    if normalise_flag:
        print("Running pre-normalise.")
        ccd_flux_norm_inside_segmask_array,  ccd_flux_norm_error_inside_segmask_array =\
            pre_normalise(ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array,
                          ccd_flux_error_inside_segmask_array, segment_begin_end)

    # AFTER prenorm has been run once, run sme will have saved these normalise fluxes, and they have been loaded
    # instead of un-normalied ones from galahsp3 earlier in load_spectra, so we don't need to modify them.
    else:
        # Just to change to the new normalised variable name.
        ccd_flux_norm_inside_segmask_array = ccd_flux_inside_segmask_array
        ccd_flux_norm_error_inside_segmask_array = ccd_flux_error_inside_segmask_array

    # Produce the array that tells us what fluxes to ignore, which are contiuum fluxes, error-laden, or atomic lines.
    flagged_fluxes = create_observational_definitions(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                                      ccd_flux_norm_inside_segmask_array,
                                                      ccd_flux_norm_error_inside_segmask_array,
                                                      segment_begin_end, continuum_mask_data, linemask_data)

    # Information on atomic lines such as wavelength, species, ionization. Primarily used during the PySME run itself.
    # During solve runs we take only atomic data in the segment mask, but that still appears to be the majority.
    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        produce_indexed_atomic_data(makestruct_dict, segment_begin_end, linemask_data, run=run)

    # if we set lineatomic to 0 due to a lack of lines we give up on the run. Likely a balmer run so we have to make
    # sure to flag that as False now, as we have just cancelled it.
    if isinstance(line_atomic, int):
        makestruct_dict['balmer_run'] = False
        print("Line atomic is 0, cancelling balmer run.")
        return makestruct_dict

    # If we have any value for atomic abundances free parameters then we combine it and the globfree (vrad etc)
    # So far un-needed and probably won't be used for a long time, until we adapt galah_ab
    if np.any(atomic_abundances_free_parameters):
        makestruct_dict["global_free_parameters"] = np.concatenate(
            np.asarray(makestruct_dict["global_free_parameters"]),
            atomic_abundances_free_parameters, axis=None)

    # Else we just use free global parameters. but add Vsin for non-normalisation runs if rot_vel is above 1.
    # If it's too low it's unimportant and we ignore it and set to 1 to adjust for it.
    if run == "Solve":
        if makestruct_dict['rotational_velocity'] > 1 and 'VSINI' not in makestruct_dict:
            makestruct_dict["global_free_parameters"].append('VSINI')
        else:
            makestruct_dict['rotational_velocity'] = 1

    # Array of ags (0 or 1), specifying for which of the spectral lines the gf values
    # are free parameters.
    spectral_lines_free_parameters = np.zeros(len(species))  # gf_free

    # Strings to tell PYSME what kind of atmosphere we're looking for.
    atmosphere_depth, atmosphere_interpolation, atmoshpere_geometry, atmosphere_method = set_atmosphere_string()

    # We create a dictionary of the indexed linelist to be input into PySME using the names they look for. We do it here
    # as we are modifying the parts like line_atomic earlier in other places. Included both by itself in a data frame
    # and in the dictionary of makestruct due to Pysme requirements.
    linelist_dict = {'atomic': line_atomic,
                     'lande': lande_mean,
                     'depth': depth,
                     'lineref': data_reference_array,
                     'species': species,
                     'line_extra': j_e_array,
                     'line_term_low': lower_level,
                     'line_term_upp': upper_level,
                     'line_lulande': lu_lande}

    # We convert our linelist into the desired format that PySME requests. We both output it as a variable and save it
    # as a file.
    linelist = dataframe_the_linelist(linelist_dict)

    # Turning our single large array into a list of smaller segmented arrays to comply with pysme iliffe standards.
    # Can be done earlier, but now currently the code is dependent on them being single arrays so this seems
    # like an easier fix. Plus arrays are better to handle for large data so it might still be correct.
    iliffe_wave, iliffe_flux_norm, iliffe_error_norm, iliffe_flags = \
        segmentise_arrays(ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                          ccd_flux_norm_error_inside_segmask_array, flagged_fluxes, segment_begin_end)

    # Updating the PySME input dictionary with the final variables. We modify the names to those that PySME accepts.
    sme_input =  {'obs_name': makestruct_dict['obs_name'], 'id': date.today(),
                 'spec': iliffe_flux_norm,
                 'teff': makestruct_dict['effective_temperature'],     'depth': depth,
                 'logg': makestruct_dict['gravity'],                   'species': species,
                 'feh': makestruct_dict['metallicity'],                'sob': ccd_flux_norm_inside_segmask_array,
                 'field_end': makestruct_dict['field_end'],            'uob': ccd_flux_norm_error_inside_segmask_array,
                 'monh': makestruct_dict['metallicity'],                'mu': makestruct_dict['intensity_midpoints'],
                 'vmic': makestruct_dict['microturbulence_velocity'],  'abundances': makestruct_dict['abundances'],
                 'vmac': makestruct_dict['macroturbulence_velocity'],  'lande': lande_mean,
                 'vsini': makestruct_dict['rotational_velocity'],      'chirat': 0.001,
                 'vrad': radial_velocity,                              'line_lulande': lu_lande,
                 'vrad_flag': 'each',                                  'atomic': line_atomic,
                 'cscale': continuum_scale,                            'nmu': makestruct_dict['specific_intensity_angles'],
                 'gam6': 1,                                            'nlte_abund': makestruct_dict['nlte_abund'],
                 'mask': iliffe_flags,
                 'accwi': makestruct_dict['wavelength_interpolation_accuracy'],
                 'accrt': makestruct_dict['specific_intensity_accuracy_min'],
                 'maxiter': (makestruct_dict['current_iterations']),        'line_term_low': lower_level,
                 'atmo': {'source': makestruct_dict['atmosphere_grid_file'],              'method': 'grid',
                          'depth': atmosphere_depth,                   'interp': atmosphere_interpolation,
                          'geom': atmoshpere_geometry},                'ipres': ccd_resolution_of_segment_array,
                 'object': makestruct_dict['obs_name'],                'iptype': makestruct_dict['broadening_profile'],
                        'uncs': iliffe_error_norm,
                 'gf_free': spectral_lines_free_parameters,            'lineref': data_reference_array,
                 'line_term_upper': upper_level,                       'nseg': number_of_segments,
                 'line_extra': j_e_array,
                 'wran': segment_begin_end,                            'wave': iliffe_wave,
                 'wob': ccd_wavelengths_inside_segmask_array,          'wind': wavelength_start_end_index[:, 1],
                 'balmer_run': makestruct_dict['balmer_run'],          'mob': flagged_fluxes,
                 'cscale_flag': makestruct_dict['continuum_scale_flag'],
                 'fitparameters': makestruct_dict["global_free_parameters"],
                 'cscale_type': makestruct_dict['continuum_scale_type'],
                 'run_type': run

                 }

    # Save our input dict using pickle to allow for later runnin of sme manually too. Only run on the first attempt
    # as those are the original inputs.
    if not makestruct_dict['load_file'] and not makestruct_dict['balmer_run']:
        store_sme_input(sme_input)
    if run == "Synthesise":
        print("Dumping latest synth input to OUTPUT")
        pickle.dump(sme_input, open("OUTPUT/Latest synth input", "wb"))
    "And here we go. We finally run pysme."
    pysme_run_sme.start_sme(sme_input, linelist, run)

    return makestruct_dict


# Updates the accuracies (resolutions) of the specta required depending on whether it's a balmer run or not. Also
# where we set the base accuracy.
def update_accuracies(makestruct_dict):
    """
    Input:
        makestruct_dict: Used to check whether it's a balmer run, and to add accuracy keys
    Output:
        makestruct_dict: Updated accuracy keys
    """
    # We want a higher accuracy for balmer runs but it's less important for non balmer due to the wider segments.
    if makestruct_dict['balmer_run']:
        makestruct_dict['wavelength_interpolation_accuracy'] = 0.00005
        makestruct_dict['specific_intensity_accuracy_min'] = 0.00005
    else:
        # Minimum accuracy for linear spectrum interpolation vs. wavelength.
        makestruct_dict['wavelength_interpolation_accuracy'] = 0.00005
        # accwi in IDL

        # Minimum accuracy for sme.sint (Specific intensities on an irregular wavelength grid given in sme.wint.)
        # at wavelength grid points in sme.wint. (Irregularly spaced wavelengths for specific intensities in sme.sint.)
        # Values above 10-4 are not meaningful.
        makestruct_dict['specific_intensity_accuracy_min'] = 0.00005
        # accrt in IDL

    return makestruct_dict


# We set the atmosphere file to the backup if we didn't set it earlier
def update_atmosphere(makestruct_dict):
    """
        Input:
            makestruct_dict: Used to check for atmosphere
        Output:
            makestruct_dict: Updated atmosphere
        """
    try:
        makestruct_dict['atmosphere_grid_file'] = makestruct_dict['atmosphere_grid_file']
    except AttributeError or NameError:
        # If we don't have a previously set atmosphere grid we use a backup 2012.
        # Try is faster as we usually will have it set.
        print("Using marcs2012 instead")
        makestruct_dict['atmosphere_grid_file'] = 'marcs2012.sav'
    return makestruct_dict

# Set the profile broadening instruments and the angles used to calculate specific intensity.
def update_profile(makestruct_dict):
    """
        Input:
            makestruct_dict: sme input dictionary, just to update its keys
        Output:
            makestruct_dict: Updated profile keys
        """
    # Number of "equal-area" angles at which to calculate specific intensity.
    # Helps calculate the midpoints of the intensities
    specific_intensity_angles = 7
    # nmu

    # Type of profile used for instrumental broadening. Possible values are gauss,
    # sinc, or table. See Section 3.4.
    broadening_profile = "gauss"
    # iptype

    # The equal-area midpoints of each equal-area annulus for which specific inten-
    # sities were calculated. values for Gaussian quadrature are not conducive
    # to subsequent disk integration, named mu in sme.
    intensity_midpoints = np.flip(
        np.sqrt(0.5 * (2 * np.arange(1, specific_intensity_angles + 1)) / specific_intensity_angles))
    makestruct_dict['specific_intensity_angles'] = specific_intensity_angles
    makestruct_dict['intensity_midpoints'] = intensity_midpoints
    makestruct_dict['broadening_profile'] = broadening_profile

    return makestruct_dict








# Sets up the arrays of the wavelengths and fluxes that are inside our segments. The output is not segmentised, that
# is done later. However, it would be a good idea to segmentise them at some point, but requires a decent amount
# of code re-writing.
def object_array_setup(makestruct_dict):
    """
    Input:
        makestruct: Contains wavelength and flux information, as well as file names to open.
    Output:
       ccd_wavelengths_inside_segmask_array: Wavelengths inside our segments only. One large array.
       ccd_flux_inside_segmask_array: Corresponding fluxes
       ccd_flux_error_inside_segmask_array: Corresponding error
       segment_begin_end: The first and final wavelength of each segment
       wavelength_start_end_index: The indexes of the first and final wavelength of each segment
       ccd_resolution_of_segment_array: Resolutions of the wavelengths we have
    """
    # Unique file to the object in question.
    segment_mask = str(makestruct_dict['setup_for_obs']) + '_Segm.dat'
    # Contains the wavelengths we are most interested in.
    segment_mask_data_with_res = segment_mask_creation(segment_mask)

    # Takes the last three digits and turns into an int. The last 3 represent the fibre of the ccd used that we want to
    # look at. Then we produce an interpolation equation for the resolutions.
    interpolation = resolution_interpolation(int(str(makestruct_dict['object_for_obs'])[-3:]))

    # Uses our resolution interpolation to find the resolution at the peak wavelengths that we also are finding here.
    segment_mask_data_with_res = interpolate_peak_resolution(makestruct_dict, segment_mask_data_with_res, interpolation)

    # Checks for a negative range
    if min(segment_mask_data_with_res['Wavelength_End'] - pd.to_numeric(
            segment_mask_data_with_res['Wavelength_Start'])) <= 0:
        print("Segment has a negative range!")
        return

    # Checks for overlapping segments if there's more than one.
    if len(segment_mask_data_with_res['Wavelength_End']) > 1:
        if max(segment_mask_data_with_res['Wavelength_End'][0:len(segment_mask_data_with_res['Wavelength_End'])]
               - pd.to_numeric(segment_mask_data_with_res['Wavelength_Start'][
                               1:len(segment_mask_data_with_res['Wavelength_Start'])])) > 0:
            print("Overlapping segments")
            return

    # We load in the wavelength etc variables to be used a lot in makestruct, first created in sp2 and later by SME.
    total_ccd_wavelength, total_ccd_flux, total_ccd_flux_error = load_spectra(makestruct_dict)
    # Adjust the wavelengths for the doppler shift.
    total_ccd_wavelength_dopplered = doppler_wavelengths(total_ccd_wavelength,
                                                         makestruct_dict['radial_velocity_global'])

    # We limit our wavelength array to those within the segment mask we loaded, and their appropriate fluxes and
    # resolutions.
    ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
        ccd_resolution_of_segment_array, wavelength_start_end_index = \
        wavelengths_flux_inside_segments(segment_mask_data_with_res, total_ccd_wavelength_dopplered, total_ccd_flux,
                                         total_ccd_flux_error)

    # Number of segments that contain visible spectra.
    if len(wavelength_start_end_index) == 0:
        print("No observations in segment mask")
        return

    # Creates an array with the beginning and end wavelengths of the segments.
    # Different to the start end array as that's indexes rather than the data itself.
    segment_begin_end = find_segment_limits(wavelength_start_end_index, total_ccd_wavelength_dopplered)
    return ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
           segment_begin_end, wavelength_start_end_index, ccd_resolution_of_segment_array


# Creates the segment mask and the resolution of it from the csv file that we name in the variable input. The segments
# represent the wavelengths of the atoms we are interested in.
def segment_mask_creation(segment_mask):
    """
    Input:
        segment_mask: a string of the name of the segment mask file for this object. Could be unique or general
    Output:
           segment_mask_data_with_res: The data of the segment mask file, particularly the wavelengths of each segment.
    """
    # Segm_mask is _Segm.data, unsurprisingly. It takes the start, end wavelength, and the resolution base guess of 3.5k
    # which is totally wrong. That's something to fix/modify when we're streamlining this.
    # The seperator separates via 2 spaces or a space and ; or . and space. We use the overflow column to account for
    # lines that begin with ; (commented out) which we immediately delete afterwards. Hard to separate comments when ;
    # was forgotten.
    segment_mask_data_with_res = pd.read_csv(
        "GALAH/DATA/" + segment_mask,
        delim_whitespace=True,
        header=None,
        names=["Wavelength_Start", "Wavelength_End", "Resolution", "comment", "overflow"],
        engine='python',
        skipinitialspace=True,
        usecols=["Wavelength_Start", "Wavelength_End",
                 "Resolution"])
    # ^ usecols auto deletes the comment and overflow columns

    # ~ asks for negation, removes any row that starts with ; in the first column.
    try:
        segment_mask_data_with_res = segment_mask_data_with_res[
            ~segment_mask_data_with_res['Wavelength_Start'].str.startswith(";")]
    # if there are no lines with a ; then it throws an error. Oops
    except AttributeError:
        pass
    # Reset the index to account for the now missing values
    segment_mask_data_with_res = segment_mask_data_with_res.reset_index(drop=True)
    # Sorts in ascending order of wavelength of starting wavelength, and re-orders the others.
    segment_mask_data_with_res.sort_values(by=['Wavelength_Start', 'Wavelength_End'])

    return segment_mask_data_with_res


# Creates an interpolation equation based on the resolutions we already have and their wavelengths.
# Also done in galah_sp2, one can be removed upon checking it runs the same.
def resolution_interpolation(object_pivot):
    """
    Input:
        object_pivot: The pivot of the object, used in selecting the file appropriate to the object we are viewing.
    Output:
        interpolation: The interpolation equation to find the resolution of other wavelengths not in our data files.
    """
    temp_ccd_piv_y = []
    temp_ccd_wave_res = []

    # Grabbing the resolution from the ccd files and concaneating them into two large arrays to interpoltae and allow
    # creation of resolution at other wavelengths (as our wavelength is modified later)
    for ccd_number in range(1, 5):
        ccd_res_file = fits.open(r'GALAH/DATA/ccd{0}_piv.fits'.format(ccd_number), ext=0)
        ccd_res_data = ccd_res_file[0].data

        # We're making an array of the resolution of the (actually x) axis
        # Extracting the row of data of ccd1 that matches the piv number (-1 as piv starts at 1)
        temp_ccd_piv_y.extend(ccd_res_data[object_pivot - 1])

        # Creates a wavelength list from starting CRVAL1 (4700) in steps of CRDELT1 (0.1)
        # until it matches the array len of NAXIS1
        resolution_wavelengths = \
            ccd_res_file[0].header['CRVAL1'] + \
            (ccd_res_file[0].header['CDELT1'] * np.arange(ccd_res_file[0].header['NAXIS1']))

        temp_ccd_wave_res.extend(resolution_wavelengths)

    wavelength_res_x_collection, wavelength_res_y_collection = \
        np.asarray(temp_ccd_wave_res), np.asarray(temp_ccd_piv_y)

    interpolation = interp1d(wavelength_res_x_collection, wavelength_res_y_collection)

    return interpolation


# Uses our previously created resolution interpolation to interpolate at the peak wavelength (which we create)
def interpolate_peak_resolution(makestruct_dict, segment_mask_data_with_res, interpolation):
    """
    Input:
        makestruct_dict: Has the resolution factor from galah_sp1
        segment_mask_data_with_res: Has the wavelngth information of the segments
        interpolation: Equation for producing resolution at wavelength center.
    Output:
        segment_mask_data_with_res: With now added center wavelengths and their resolution as an addition column.
    """
    # Loop through each wavelength range (consisting of a start and end) in the data and take the peak value and
    # its res factor.
    # The idl code seems to ignore dum1 column, and instead calls seg_st and en
    # We make a list to avoid calling pandas data frame repeatedly, I believe this is faster and avoids any copy errors.
    temporarywavelengthpeaklist = []
    for wavelengthband in range(0, len(segment_mask_data_with_res['Resolution'])):
        # Calculating the peak wavelength
        wlpeak = 0.5 * (float(segment_mask_data_with_res.loc[wavelengthband, 'Wavelength_Start'])
                        + float(segment_mask_data_with_res.loc[wavelengthband, 'Wavelength_End']))
        # Appending it to a list to add as a column to segment_mask_data_with_res
        temporarywavelengthpeaklist.append(wlpeak)
        # Interpolating the resolution at the wavelength peak and replacing the resolution of that index with it
        segment_mask_data_with_res.loc[wavelengthband, 'Resolution'] = interpolation(
            wlpeak) * makestruct_dict['resolution_factor']
    # We insert the peak list into the dataframe as a new column. I believe faster and easier than inserting a new row
    # each loop.
    segment_mask_data_with_res.insert(0, "Wavelength_Peak", temporarywavelengthpeaklist)
    return segment_mask_data_with_res


# Opens the file with the newly updated parameters of Teff, etc, produced by SME if instructed to. If it's the first
# run before SME, we just use the un-normalised spectra which we will later normalise.
def load_spectra(makestruct_dict):
    """
    Input:
        makestruct_dict: Information on the run type and whether we need to load the data file, and its file name.
    Output:
        spectra_data: Segmented data of wavelength, flux, and error
    """
    # Change it if load file is set to the one SME outputs.
    if makestruct_dict['load_file']:
        # During non balmer synth runs we load the normal spectra, and then save a duplicate version for the balmer
        # run to be able to copy eactly all the data that the normalise run is using.
        if not makestruct_dict['balmer_run']:
            spectra_data = \
                pickle.load(open("OUTPUT/SPECTRA/" + makestruct_dict['obs_name'] + "_SME_spectra.pkl", "rb"))
            # Creating a duplicate for the next balmer run
            pickle.dump(spectra_data, open("OUTPUT/SPECTRA/Temp_Balmer_spectra_input.pkl", "wb"))

        # For the balmer run we want to load the same spectra that the normalisation run just loaded, NOT what it
        # created due to the doppler shifting and potential pysme edits.
        elif makestruct_dict['balmer_run']:
            spectra_data = pickle.load(open("OUTPUT/SPECTRA/Temp_Balmer_spectra_input.pkl", "rb"))
    # This is for before SME runs we open the one created in sp2.
    else:
        spectra_data = makestruct_dict['unnormalised_spectra']
    # If the first value in the wavelength is an array with all wavelengths in the first segment.
    # If it's not an array it's not segmented, the first value is just a float and we don't need to desegmentise.
    # mask exists in the dict, but not used in calculations.
    if isinstance(spectra_data['wave'][0], np.ndarray):
        wave, flux, error = desegmentise(spectra_data)
        return wave, flux, error

    return spectra_data['wave'], spectra_data['flux'], spectra_data['error']


# Adjusts the wavelengths according to the doppler shift and returns it.
def doppler_wavelengths(total_ccd_wavelength, radial_velocity_global):
    """
    Input:
        total_ccd_wavelengths: All our wavelengths
        radial_velocity_global: The radial velocity of the star, used to adjust for its doppler shifting of light
    Output:
        total_ccd_wavelength: New wavelengths adjusted for the doppler effect
    """
    # Setting speed of light
    c = 2.99792458E8
    print("radial vel", radial_velocity_global)
    print("applying doppler", total_ccd_wavelength)
    # Shifts the wavelengths to account for the doppler effect.
    total_ccd_wavelength_dopplered = total_ccd_wavelength / ((radial_velocity_global[0] * (1E3 / c)) + 1E0)

    #total_ccd_wavelength_dopplered = total_ccd_wavelength / ((radial_velocity_global[0] * (1E3 / c)) + 1E0)
    print("After doppler,", total_ccd_wavelength_dopplered)

    return total_ccd_wavelength_dopplered


# Finds the wavelengths that we have that are also inside the segments.
def wavelengths_flux_inside_segments(segment_mask_data_with_res, total_ccd_wavelength_dopplered, total_ccd_flux,
                                     total_ccd_flux_error):
    """
    Input:
        segment_mask_data_with_res: Contains the start and end of the desired segments of wavelengths
        total_ccd_wavelength_dopplered: All wavelength array
        total_ccd_flux: Flux of the wavelengths
        total_ccd_flux_error: Error of flux
    Output:
        ccd_wavelengths_inside_segmask_array: A large array of ALL wavelengths inside ALL segments
        ccd_flux_inside_segmask_array: Their corresponding fluxes
        ccd_flux_error_inside_segmask_array: Their corresponding error
        ccd_resolution_of_segment_array: Resolutions of the x/y pair
        wavelength_start_end_index: The indexes in our array that represent the start and end of the segments.
    """
    # Can't be sure how many data points there will be, so arrays can't be used. Instead we use a temp list.
    ccd_wavelengths_inside_segmask = []
    ccd_flux_inside_segmask = []
    ccd_flux_error_inside_segmask = []
    ccd_resolution_of_segment = []
    # Array for the first and final wavelength indexes of each segment respectively.
    wavelength_start_end_index = np.zeros((len(segment_mask_data_with_res["Wavelength_Start"]), 2))

    # For each segment in segmask, find the values of dopplered wavelength (and associated flux from indexing)
    # that are inside.
    # Despite having this array we still use np.where most times to find the wavelengths in the segments,
    # potential change for that.
    for segment in range(0, len(segment_mask_data_with_res["Wavelength_Start"])):
        # Beginning wavelength and end of that segment. Put as variables here for readability.
        seg_start = (pd.to_numeric(segment_mask_data_with_res["Wavelength_Start"][segment]))
        seg_stop = (pd.to_numeric(segment_mask_data_with_res["Wavelength_End"][segment]))

        # Finding the index of values inside the segment, using "logical and" is a neccesity.
        wavelength_inside_segmask_index = np.where(
            np.logical_and(seg_stop >= total_ccd_wavelength_dopplered, total_ccd_wavelength_dopplered >= seg_start))

        # Adding the wavelengths inside the segment to our list of wavelengths of ALL segment wavelengths.
        # We segmentise it later.
        ccd_wavelengths_inside_segmask.extend(total_ccd_wavelength_dopplered[wavelength_inside_segmask_index])
        ccd_flux_inside_segmask.extend(total_ccd_flux[wavelength_inside_segmask_index])
        ccd_flux_error_inside_segmask.extend(total_ccd_flux_error[wavelength_inside_segmask_index])

        # Numpy array of indexes of the first and final wavelengths per segment with column 0 being the first
        if wavelength_inside_segmask_index[0].size != 0:
            wavelength_start_end_index[segment, 0] = (wavelength_inside_segmask_index[0][0])
            wavelength_start_end_index[segment, 1] = (wavelength_inside_segmask_index[-1][-1])
        ccd_resolution_of_segment.append(segment_mask_data_with_res['Resolution'][segment])

    # Turning lists into arrays for numpy indexing with np.where.
    ccd_wavelengths_inside_segmask_array = np.array(ccd_wavelengths_inside_segmask)
    ccd_flux_inside_segmask_array = np.array(ccd_flux_inside_segmask)
    ccd_flux_error_inside_segmask_array = np.array(ccd_flux_error_inside_segmask)
    ccd_resolution_of_segment_array = np.array(ccd_resolution_of_segment)
    return ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
           ccd_resolution_of_segment_array, wavelength_start_end_index


# Finds the first and final wavelengths of the segments that we have.
def find_segment_limits(wavelength_start_end_index, total_ccd_wavelength_dopplered):
    """
    Input:
        Wavelength_start_end_index: The indexes of the beginning and ending of each segment in our wavelength array
        total_ccd_wavelength_dopplered: An array of all wavelengths in all segments
    Output:
        segment_begin_end: A two column, multi row wavelength array. [:, 0] is the start, [:, 1] the end of each segment
    """
    number_of_segments = len(wavelength_start_end_index)

    # An array with two columns, the first and last recorded wavelength in each segment
    segment_begin_end = np.copy(wavelength_start_end_index)  # copy to avoid overwriting wind. Just using it for size.
    for windex_row in range(0, number_of_segments):
        # At indexes 0,0 and 0,1 (and then 1,0etc) of the index array 'wavelength_start_end_index' we take the value and
        # apply it to the wavelngtharray as the values we have taken are indexes of the first and last wavelength of
        # each segment. windrow, 0 is the segment beginning. , 1 is the end.
        segment_begin_end[windex_row, 0] = total_ccd_wavelength_dopplered[
            int(wavelength_start_end_index[windex_row, 0])]
        segment_begin_end[windex_row, 1] = total_ccd_wavelength_dopplered[
            int(wavelength_start_end_index[windex_row, 1])]
    return segment_begin_end


# Our code is actually set up to not use segmented arrays, so we desegment it here, run it through the code, and later
# RE-segment it. Obviously this is not ideal.
def desegmentise(spectra):
    wave, flux, error = [], [], []

    for value in spectra['wave']:
        wave.extend(value)
    wave = np.asarray(wave)

    for value in spectra['flux']:
        flux.extend(value)
    flux = np.asarray(flux)

    for value in spectra['error']:
        error.extend(value)
    error = np.asarray(error)

    return wave, flux, error


# This is the function to prenormalise the observed spectra line. Calls on autonormalise,
# removes data that is too far away from the continuum and uses the closer data to normalise itself.
def pre_normalise(ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array,
                  ccd_flux_error_inside_segmask_array, segment_begin_end):
    """
    Input:
        ccd_wavelengths_inside_segmask_array: All wavelengths in all segments
        ccd_flux_inside_segmask_array: Their fluxes
        ccd_flux_error_inside_segmask_array: The error in flux
        segment_begin_end: The start and final wavelengths of each segment, to segmentise ccd_wave.. using np.where()
    Output:
        ccd_flux_inside_segmask_array: Flux is now normalised
        ccd_flux_error_inside_segmask_array: As is error

    """
    # Number of segments we have. Made each function to avoid potential length errors.
    number_of_segments = len(segment_begin_end)
    # Pre-normalization steps. ";Performs first-guess normalisation by robustly converging straight line fit to high
    # pixels"
    for segment_band in range(0, number_of_segments):

        # Finds the index where the wavelengths are between the start and end of each segment, to be able to loop each
        # seg as i. We repeat this step quit often in the code, and things similar to it to find "inside" the segment.
        # Sometimes it's useless as we already have a list of inside the segments, but this time it is segmenting it
        # all.
        segment_indexes = (np.where(np.logical_and(ccd_wavelengths_inside_segmask_array >=
                                                   segment_begin_end[segment_band, 0],
                                                   ccd_wavelengths_inside_segmask_array <=
                                                   segment_begin_end[segment_band, 1])))[0]

        # If count is greater than 20, we can normalise. len(segindex) is the number of values that fit our criteria.
        if np.size(segment_indexes) > 20:
            # Take the coefficients of the polyfit, and the flux of it too using the equation from IDL and then applies
            # a ploynomial fit, removing outlying values until it no longer changes. The outlying value limit is
            # hardcoded. We then use the equation fit to it to normalise it
            continuous_function = autonormalisation(
                ccd_wavelengths_inside_segmask_array[segment_indexes],
                ccd_flux_inside_segmask_array[segment_indexes], 1, 0)

            # Puts it to a relative flux out of 1 that we see in the abundance charts.
            # Have to take relative indexes for cont_func to have the correct shape.
            ccd_flux_inside_segmask_array[segment_indexes] = ccd_flux_inside_segmask_array[
                                                                 segment_indexes] / continuous_function
            ccd_flux_error_inside_segmask_array[segment_indexes] = ccd_flux_error_inside_segmask_array[
                                                                       segment_indexes] / continuous_function

        # If we don't have enough points, we just use the mean value instead. Numpy mean did not work sometimes.
        else:
            # np.mean had issues with our results. Making a variable for readability.
            flux_mean = (sum(ccd_wavelengths_inside_segmask_array[segment_indexes])) / len(
                ccd_wavelengths_inside_segmask_array[segment_indexes])

            # Must be ordered correctly or we modify the sob before we use it to modify uob!
            ccd_flux_error_inside_segmask_array[segment_indexes] = \
                ccd_flux_error_inside_segmask_array[segment_indexes] / flux_mean

            ccd_flux_inside_segmask_array[segment_indexes] = \
                (ccd_flux_inside_segmask_array[segment_indexes]) / flux_mean

    # end of prenormalisation. Again we return the modified observational data

    return ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array


# We use a hard coded equation t aid in the normalisation of the flux depending on the polynomial number we choose.
def autonormalisation(wavelength_array, flux_array, polynomial_order, fit_parameters):
    """
    Input:
        wavelength_array: Wavelengths in the segment
        flux_array: fluxes of the segment
        polynomial_order: Type of polynomial this segment is
        fit_parameters: More adjustment for the fit
    Output:
        continuous_function: A function to apply to the flux to normalise it
    """
    # To be used when checking to see if we've removed too many values and when creating cont_fn.
    original_wave = wavelength_array
    if polynomial_order == 0:
        polynomial_order = 2
    if fit_parameters == 0:
        fit_parameters = 1.5

    # Stops the IDE throwing a "not created" fit. Unimportant and completely useless otherwise.
    polyfit_coefficients = 1
    inlier_index = 0
    continuous_function = 1

    # Using numpy polyfit to replace the idl Robust_poly_fit. Simply gets a polynomial fit for the spectra, and repeats
    # until either converged, or reaches 99.
    for polyfit_loop in range(0, 99):
        # Gets the coefficients for a fit of the order polynomial_order
        polyfit_coefficients = np.polynomial.polynomial.polyfit(wavelength_array, flux_array, polynomial_order)

        # Uses these to get an array of the y values of this line
        fitted_flux = np.polynomial.polynomial.polyval(wavelength_array, polyfit_coefficients)

        # Creates an array of the error (sigma) of the line compared to original data, to find outliers by getting the
        # standard deviation of the difference
        fitted_sigma = np.std(flux_array - fitted_flux)
        # Find the fluxes that exist below the linear fit + error but above the lower error boundary (* p)
        # So not outliers, but inliers. We take the first value from the output which is the index array itself
        inlier_index = (np.where(np.logical_and(flux_array < (fitted_flux + (2 * fitted_sigma)),
                                                (flux_array > (fitted_flux - (fit_parameters * fitted_sigma))))))[0]

        # If poly_order is wrong, we just stick with a value of 1 to keep everything the same
        continuous_function = 1
        if polynomial_order == 2:
            continuous_function = polyfit_coefficients[0] + (polyfit_coefficients[1] * original_wave) \
                                  + (polyfit_coefficients[2] * original_wave ** 2)
        elif polynomial_order == 1:
            continuous_function = polyfit_coefficients[0] + (polyfit_coefficients[1] * original_wave)

        # Stops when no more convergence occurs, breaks the loop. Again, np where gives a tuple with dtype
        # the second condition uses the original non edited wavelength array
        if len(inlier_index) == len(wavelength_array) or len(inlier_index) / len(original_wave) <= 0.1:
            break
        if polyfit_loop >= 98:
            print("Did not converge")
            break
        # Replace the array with only values that lie inside the error boundaries we have.
        wavelength_array = wavelength_array[inlier_index]
        flux_array = flux_array[inlier_index]
    # Currently non-useful due to not neding a 2nd order polynomial fit.
    if polynomial_order == 2:
        # co in idl, compared to the continuous flux of c. These variables, man.. Does not get returned in make struct
        # Additional just means there's something different that I don't know.
        continuous_function_additional = polyfit_coefficients[0] + polyfit_coefficients[1] * \
                                         wavelength_array[inlier_index] + polyfit_coefficients[2] * \
                                         wavelength_array[inlier_index] ** 2
    elif polynomial_order == 1:
        continuous_function_additional = polyfit_coefficients[0] + polyfit_coefficients[1] * wavelength_array[
            inlier_index]
    else:
        continuous_function_additional = 1

    return continuous_function


# Open up the line and continuum masks for making mob/flagged fluxes. Linemask is DRX_Sp, where X is 2 for testing.
# They contain the information on the important atom wavelengths, and the length of our desired continuum.
def open_masks(makestruct_dict):
    """
    Input:
        makestruct_dict: Contains the file name unique to the object if created.
    Output:
        continuum_mask_data: Information on the length and size of the continuum
        linemask_data: Information on the wavelengths and species of important atomic segmnts
    """
    # Reads out the columns which are centre/peak wavelength, start and end of wavelength peak (all simulated,
    # not observed), and atomic number. Seperated by 2 spaces (sep), no headers as it is only data, and the names are
    # what we assign them as to be used later with linemask_data["Sim_wave" etc.]
    # Careful with the python engine, it's slower. If we are looking at BIG data files this might be bad.
    print("Linemask name here is ", makestruct_dict['line_mask'])
    linemask_data = pd.read_csv(makestruct_dict['line_mask'], delim_whitespace=True, header=None,
                                engine='python', names=[
            "Sim_Wavelength_Peak", "Sim_Wavelength_Start", "Sim_Wavelength_End", "Atomic_Number"])
    try:
        # Removes all rows that begin with ; as it's commented out. Required to be able to modify the data by column
        # rather than by row
        linemask_data = linemask_data[~linemask_data['Sim_Wavelength_Peak'].str.startswith(";")]

    except AttributeError:
        pass
    # Reset the index to account for the now missing values
    linemask_data = linemask_data.reset_index(drop=True)

    # Read the start and end of the continuum
    continuum_mask = str(makestruct_dict['setup_for_obs']) + "_Cont.dat"
    continuum_mask_data = pd.read_csv("GALAH/DATA/" + continuum_mask, delim_whitespace=True, header=None,
                                      engine='python', names=["Continuum_Start", "Continuum_End"])

    return continuum_mask_data, linemask_data


# Sums the flagged flux creation functions in one to produce a final product. Flags the wavelengths of the peaks,
# continuum, and ones that need to be removed.
# 2: Continuum, 1: Atomic line, 0: Removed/Ignore
def create_observational_definitions(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                     ccd_flux_norm_inside_segmask_array, ccd_flux_norm_error_inside_segmask_array,
                                     segment_begin_end, continuum_mask_data, linemask_data):
    """
    Input:
        makestruct_dict:
        ccd_wavelengths_inside_segmask_array: Wavelengths to associate with atomic lines from linemask
        ccd_flux_norm_inside_segmask_array: Fluxes to see what is a good continuum line, and what is not
        ccd_flux_norm_error_inside_segmask_array: Identify high error poor continuum choices to remove
        segment_begin_end: Find the beginning and end of each segment, as it must be flagged individually
        continuum_mask_data: information on the continuum we want to use, start and end wavelengths
        linemask_data: Wavelengths of the desired atomic lines to look at and their species.
    Output:
        flagged_fluxes: An array indicating which fluxes to remove, keep, and which are the important atomic lines
    """
    # We first flag fluxes that appear to be absorption lines,
    # then the continuum, then remove those with too low a flux to be used
    # inputting the previously created array each time.
    flagged_fluxes = flag_absorption(linemask_data, ccd_wavelengths_inside_segmask_array,
                                     ccd_flux_norm_inside_segmask_array, ccd_flux_norm_error_inside_segmask_array)

    # We only need the continuum during synthesize runs where cscale flag is linear, otherwise it's fixed.
    # Same as flagging the absorption lines, we compared to noise and flag all non-AL as continuum that we then
    # narrow down in the next function.
    if makestruct_dict['continuum_scale_flag'] == 'linear':
        flagged_fluxes = flag_continuum(ccd_wavelengths_inside_segmask_array,
                                        ccd_flux_norm_inside_segmask_array, ccd_flux_norm_error_inside_segmask_array,
                                        flagged_fluxes, continuum_mask_data)

    # Now we remove the fluxes that are below our desired continuum line as they will affect our steady flux = 1
    # normalisation. We leave the absorption lines alone here.
    flagged_fluxes = cutting_low_flux(ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                                      segment_begin_end, flagged_fluxes)
    # We remove the lines close to the NLTE-dominated cores
    flagged_fluxes = removing_nlte(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                   ccd_flux_norm_inside_segmask_array, flagged_fluxes)

    return flagged_fluxes


# Flag the fluxes that correspond to the atomic absorption lines. Other functions set the continuum and undesirables.
# All values inside Sp are flagged as they are the desired wavelengths.
# Checks the fluxes averages over 4 data points each and compares it to the noise to check suitability.
def flag_absorption(linemask_data, ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                    ccd_flux_norm_error_inside_segmask_array):
    """
    Input:
        linemask_data: Locating the segments to check their fluxes and wavelengths
        ccd_wavelengths_inside_segmask_array: All wavelengths
        ccd_flux_norm_inside_segmask_array: Their fluxes (Normalised)
        ccd_flux_norm_error_inside_segmask_array: Error of flux
    Output:
        flagged_fluxes: An array of 0s (Default ignores) and 1s (Absorption lines). 2s are added in the continuum
                        function. Same size (and represents) the wavelengths.
    """

    # Mask of observed pixels, just for placement to then flag them as continuum, etc.
    flagged_fluxes = np.zeros(len(ccd_wavelengths_inside_segmask_array))

    # We flag the fluxes that are probably peaks inside our segments that we care about that are probably atomic
    # absorbtion lines. We loop for each segment.
    for line_loop in range(0, len(linemask_data['Sim_Wavelength_Start'])):

        # Usual case of making sure the wavelengths we want to use are in the lines.
        wavelengths_inside_linemask_index = np.where(np.logical_and(
            ccd_wavelengths_inside_segmask_array >= linemask_data['Sim_Wavelength_Start'][line_loop],
            ccd_wavelengths_inside_segmask_array <= linemask_data['Sim_Wavelength_End'][line_loop]))

        # running_snr in idl, sets values to 1 if they're below the max noise spike. This means we flag all good points
        # at 1 that are below spikes in noise to be used. Demonstrates the ratio of signal to noise for the points
        signal_to_noise = []
        # We're trying to find signal to noise ratios, where 1.5 is the limit for our noise
        # Averages over +/-4 values.
        for flux_row in range(0, len(ccd_flux_norm_inside_segmask_array)):
            # Indexes the obs flux from ii-4 to ii + 4 (or limits of length)
            signal_to_noise.append(max(
                [1 + 10 / (np.mean(ccd_flux_norm_inside_segmask_array[
                                   max(0, flux_row - 4):min(flux_row + 4, len(ccd_flux_norm_inside_segmask_array) - 1)]
                                   /
                                   ccd_flux_norm_error_inside_segmask_array[
                                   max(0, flux_row - 4):min(flux_row + 4, len(ccd_flux_norm_inside_segmask_array) - 1)]
                                   )),
                 1.5]))
        signal_to_noise = (np.array(signal_to_noise))

        # If there are some wavelengths in the segment, we can see if they're peaks.
        if len(wavelengths_inside_linemask_index[0]) > 0:

            # If the flux exists and is less than the noise, set a marker to 1 to indicate this. (Atomic line)
            if min(ccd_flux_norm_inside_segmask_array[wavelengths_inside_linemask_index[0]]) > 0 and \
                    max(ccd_flux_norm_inside_segmask_array[wavelengths_inside_linemask_index[0]]) < \
                    max(signal_to_noise[wavelengths_inside_linemask_index[0]]):
                # 1 is a good thing! it means it fits nicely in the peak and the noise is nothing to worry about
                flagged_fluxes[wavelengths_inside_linemask_index[0]] = 1
    # Return our array of which wavelengths are flagged as possible peaks to be modified further in contiuum and more.
    return flagged_fluxes


# Flags continuum lines during Synthesize runs, sets non absorption lines that also meet our error/noise requirements
# to 2 to represent continuum. If they don't meet the noise requirements, sets to 0 as they are unusabe.
def flag_continuum(ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                   ccd_flux_norm_error_inside_segmask_array, flagged_fluxes, continuum_mask_data):
    """
    Input:
        continuum_mask_data: Locating the continuum to check its wavelength region
        ccd_wavelengths_inside_segmask_array: All wavelengths
        ccd_flux_norm_inside_segmask_array: Their fluxes (Normalised)
        ccd_flux_norm_error_inside_segmask_array: Error of flux
        flagged_fluxes: The previous located absorption lines, as we DO NOT want these as continuum. Never.
    Output:
        flagged_fluxes: An array of 0s (Default ignores) and 1s (Absorption lines) and 2s (Continuum).
                        Same size (and represents) the wavelengths.
    """

    # For each segment in the continuum file, often will be one large range,
    for continuum_loop in range(0, len(continuum_mask_data['Continuum_Start'])):

        # A list to append to with the signal to noise ratios, either 1.5 or 1 + 10/mean(flux/error)
        # from i to ii -/+4. So the higher the flux and lower the error, the more likely to have 1.5 as our baseline
        signal_to_noise = []
        for flux_row in range(0, len(ccd_flux_norm_inside_segmask_array)):
            signal_to_noise.append(max([1 + 10 /
                                        np.mean(ccd_flux_norm_inside_segmask_array[
                                                max(0, flux_row - 4):
                                                min(flux_row + 4, len(ccd_flux_norm_inside_segmask_array))]
                                                / ccd_flux_norm_error_inside_segmask_array[
                                                  max(0, flux_row - 4):
                                                  min(flux_row + 4, len(ccd_flux_norm_inside_segmask_array))]),
                                        1.5]))
        signal_to_noise = (np.array(signal_to_noise))

        # Indexes where the wavelengths are inside the continuum.
        wavelengths_inside_continuum_index = np.where((np.logical_and(
            ccd_wavelengths_inside_segmask_array >= continuum_mask_data['Continuum_Start'][continuum_loop],
            ccd_wavelengths_inside_segmask_array <= continuum_mask_data['Continuum_End'][continuum_loop])))

        # This is cleaner code (albeit slower I imagine) than having 5 np logical ands in the above statement.
        # If we haven't already flagged the flux as peak flux, and it's less than the noise then we flag it with '2'
        # to mark it as continuum
        if len(wavelengths_inside_continuum_index[0]) > 0:
            for continuum_index in wavelengths_inside_continuum_index[0]:

                if flagged_fluxes[continuum_index] != 1 and \
                        0 < ccd_flux_norm_inside_segmask_array[continuum_index] < signal_to_noise[continuum_index]:
                    flagged_fluxes[continuum_index] = 2

    return flagged_fluxes


# This function removes the lowest 70% of fluxes while retaining enough points on both sides of peak to have a continuum
def cutting_low_flux(ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array, segment_begin_end,
                     flagged_fluxes):
    """
    Input:
        ccd_wavelengths_inside_segmask_array: Wavelengths assigned to the fluxes, to see where they lie in the spectra.
                                            Used to ensure a continuum on both sides of the absorption line.
        ccd_flux_norm_inside_segmask_array: The fluxes checked to remove those with the lowest fluxes that are no
                                            use for the continuum.
        segment_begin_end: The beginning and end wavelengths for each segment, as each segment needs its own continuum
        flagged_fluxes: Current continuum and absorption lines flagged. We do NOT remove any atomic lines.
    Output:
        flagged_fluxes: Now with the low fluxes set to 0 so they are ignored and not used in the continuum.
    """
    # Number of segments we have. Made each function to avoid potential length errors.
    number_of_segments = len(segment_begin_end)

    # Deselect 70% lowest continuum points in each segment using synthesis. Ensure both ends have continuum points
    for segment_band in range(0, number_of_segments):
        # The fraction of points we want to remove. Potentially modified in the while loop hence kept outside of it
        # and reset for each segment.
        fraction = 0.7
        # We take the wavelength indexes of the wavelengths that exist in the current segment of the loop.
        wavelength_inside_segment_index = np.where(
            np.logical_and(ccd_wavelengths_inside_segmask_array >= segment_begin_end[segment_band, 0],
                           segment_begin_end[segment_band, 1] >= ccd_wavelengths_inside_segmask_array))

        # While the fraction to remove is not 0% we continue looping. We check that we have enough points for the
        # continuum before cutting fluxes, and if we don't we reduce the fraction by 10%, hence if it hits 0% we have to
        # stop. Otherwise if we do, we set fraction to 0 to break the loop.
        # Python is setting it to E-17 in loop so this fixes it by using 0.01 instead of == 0.
        while fraction > 0.01:
            # The value of the fraction of the fluxes we chose, so how many IS 70% for example.
            value_of_fraction = int(len(ccd_flux_norm_inside_segmask_array[wavelength_inside_segment_index]) * fraction)

            # Takes the index of the 70%th lowest value (our cut off point) from a sorted list. Sorting takes a long
            # time but this doesn't appear to be the bottleneck. @@@
            cutting_flux_value = sorted(
                ccd_flux_norm_inside_segmask_array[wavelength_inside_segment_index])[value_of_fraction]

            # We take a list of indexes where the flux in the segment and is below our cut off point. No longer sorted.
            cutting_flux_index = np.where(np.logical_and(
                ccd_flux_norm_inside_segmask_array < cutting_flux_value,
                np.logical_and(ccd_wavelengths_inside_segmask_array >= segment_begin_end[segment_band, 0],
                               segment_begin_end[segment_band, 1] >= ccd_wavelengths_inside_segmask_array)))

            # We need to count how many values there are at the extreme ends of the segment,
            # as we need a continuum on both sides.
            # Here we see how many continuum points are left in total.
            # Those considered also have to be flagged as 2 - so no absorption lines are involved here.
            saved_continuum_index = np.where(np.logical_and(
                ccd_flux_norm_inside_segmask_array >= cutting_flux_value,
                np.logical_and(ccd_wavelengths_inside_segmask_array >= segment_begin_end[segment_band, 0],
                               segment_begin_end[segment_band, 1] >= ccd_wavelengths_inside_segmask_array)))

            # Again the [0] to take the values of the array, and ignore dtype= which numpy gives us.
            # This tells us how many values are in the top and bottom 2/3 and 1/3,
            # we use indexes here as the wavelengths are ordered so it works out easily as [133] > [132] for example.
            # If the wavelengths are somehow NOT ordered, then we must change how to determine how many are in the
            # top and bottom 2/3 and 1/3.
            low_continuum_index = np.where(
                saved_continuum_index[0] <=
                wavelength_inside_segment_index[0][int(len(wavelength_inside_segment_index[0]) / 3)])

            high_continuum_index = np.where(
                saved_continuum_index[0] >=
                wavelength_inside_segment_index[0][int(len(wavelength_inside_segment_index[0]) * 2 / 3)])
            # If we don't have enough points, decrease the fraction we remove.
            if len(low_continuum_index[0]) < 5 or len(high_continuum_index[0]) < 5:
                fraction -= 0.1

            # If we have enough points on both sides, we can continue and remove them by looping through the indexes of
            # the low fluxes.
            else:
                for index_loop in cutting_flux_index[0]:
                    # Checks if it's 2, as we don't want to remove spectra values that are at 1.
                    if flagged_fluxes[index_loop] == 2:
                        # print("2")
                        flagged_fluxes[index_loop] = 0
                # And now we break this loop as we have applied what we needed to.
                fraction = 0

    return flagged_fluxes


# We remove the lines close to the NLTE-dominated cores
def removing_nlte(makestruct_dict, ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                  flagged_fluxes):
    """
    Input:
        makestruct_dict: Information on the line cores, and stellar metallicity
        ccd_wavelengths_inside_segmask_array: Wavelengths assigned to the fluxes, comparing to the line core wavelengths
        ccd_flux_norm_inside_segmask_array: We check to see if they are below the preset minimum if they lie close to
                                            the line cores.
        flagged_fluxes: Current continuum and absorption lines flagged. We do NOT remove any atomic lines.
    Output:
        flagged_fluxes: Now with low fluxes near the line cores removed.
    """
    # Avoid strong NLTE-dominated cores in mask if the flux is below a certain threshhold.
    core_minimum = 0.6
    if makestruct_dict['metallicity'] < -2:
        core_minimum = 0.72
    elif makestruct_dict['metallicity'] < -1:
        core_minimum = 0.65

    # Checks for line_cores existing from galahsp1, if no variable exists then we can't do this.
    try:
        print("Using the line cores:", makestruct_dict['line_cores'])
        # if it's near the value of the line core wavelength, and the flux is below a preset min, we're setting it to 0.
        for line_core_loop in range(0, len(makestruct_dict['line_cores'])):
            line_core_wavelength_index = np.where(
                np.logical_and(
                    abs(ccd_wavelengths_inside_segmask_array - makestruct_dict['line_cores'][line_core_loop]) < 4,
                    ccd_flux_norm_inside_segmask_array < core_minimum))
            flagged_fluxes[line_core_wavelength_index] = 0
    except AttributeError:
        print("No line_cores")
    return flagged_fluxes


# We produce an indexed version of the large amount of atomic data that is contained in the files used, taking only the
# atomic lines that are either in the segments only, or in the linemask depending on what we send the variable 'run' to.
def produce_indexed_atomic_data(makestruct_dict, segment_begin_end, linemask_data, run="Synthesise"):
    """
    Input:
        makestruct_dict: Balmer run, depth, and broadline data.
        segment_begin_end: We use segmented loops to check for broadline data
        linemask_data: Information on the wavelengths of the line_mask
        run: We use linemask or segment mask depending on the run.
    Output:
        Many variables, all containing information on atomic lines such as ionization or their species, or the depth
        observed.
    """

    # Finds the files containing atomic data and creates arrays to hold them in a format suitable for PySME
    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        indexed_stellar_information(makestruct_dict)
    print("line_atomic", len(line_atomic))

    # If run is Solve, we want the indexes of wavelengths in segments we already have,
    # else we want them from the linemask ranges.
    if run == "Synthesise":
        print("Running synth here")
        desired_atomic_lines_index = atomic_lines_in_segments(makestruct_dict,
                                                              segment_begin_end, line_atomic, depth)
    else:
        print("Running", run)
        desired_atomic_lines_index = atomic_lines_in_linemask(makestruct_dict,
                                                              line_atomic, depth, linemask_data)
    print("Balmer run is: ", makestruct_dict['balmer_run'])
    # If we can't find ANY atomic lines in the segments, we need to return something to continue despite this failed
    # balmer run.
    if len(desired_atomic_lines_index) == 0:
        print("No atomic lines within chosen Segments. Likely during a Balmer line run, moving on.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    # Is now an array as we're doing adding to it, and we take only the unique values using Unique.
    desired_atomic_lines_index = np.unique(desired_atomic_lines_index)
    nselect = len(desired_atomic_lines_index)

    # Showing how many lines were not used. But might be ambiguous, as these are the duplicates,
    # not the ones out of bounds.
    # The out of bounds ones were discared 
    print(nselect, "unique spectral lines are selected within wavelength segments out of", len(line_atomic))

    # We want to index the atomic data, and so apply it to line_atomic first, as we then order them according to
    # wavelength, which we want to apply directly to the other atomic information to keep the information in the right
    # order.
    line_atomic = line_atomic[desired_atomic_lines_index]
    # Now we also sort them according to wavelength.
    sort_line_index = np.argsort(line_atomic[:, 2])

    # So now we apply these indexes to the information taken from smerdlin. As these indexes were taken useing
    # line atomic then the position of the index shoudl be fine.
    species = species[desired_atomic_lines_index][sort_line_index]
    line_atomic = line_atomic[sort_line_index]
    lande_mean = lande_mean[desired_atomic_lines_index][sort_line_index]
    depth = depth[desired_atomic_lines_index][sort_line_index]
    data_reference_array = data_reference_array[desired_atomic_lines_index][sort_line_index]
    lower_level = lower_level[desired_atomic_lines_index][sort_line_index]
    upper_level = upper_level[desired_atomic_lines_index][sort_line_index]
    j_e_array = j_e_array[desired_atomic_lines_index][sort_line_index]
    lu_lande = lu_lande[desired_atomic_lines_index][sort_line_index]

    return line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande


# Produces the arrays with the stellar information from the master file for every wavelength. such as line_atomic etc.
# Either loads a previously modified linelist, or makes a modified linelist using the data we load from GALAH in
# load_linelist, and then modifying various parameters within it such as the species name to include ionization.
def indexed_stellar_information(makestruct_dict):
    """
    Input:
        makestruct_dict: Information on the chosen line list, and whether it's already been modified.
    Output:
        Many variables containing atomic information on the wavelengths in the line list.
        In format:
                    |Description|      |Content|                     |Key|
                atomic/data array:    Atomic number,           -> ['atomic']
                                      Ionic number
                                      Lambda
                                      E_Low
                                      Log_gf
                                      Rad_damp
                                      Stark_Damp
                                      Vdw_Damp
                Lande_Mean:           Lande_mean               -> lande
                Depth:                Depth                    -> depth
                Lu_Lande:             lande_lo,                -> line_lulande
                                      lande_up
                j_e_array:            j_low,                   -> line_extra
                                      e_up,
                                      j_up
                lower_level:          label_low                -> line_term_low
                upper_level:          label_up                 -> line_term_upp
                Data_reference_array: lambda_ref,              -> lineref
                                      e_low_ref,
                                      log_gf_ref,
                                      rad_damp_ref,
                                      stark_damp_ref,
                                      vdw_damp_ref,
                                      lande_ref


    """
    # The new location for us to save and load our linelist.
    makestruct_dict['line_list_location'] = r'OUTPUT/LINELIST/'
    # We try to open the indexed linelist (with modifications made to logg, etc) that is created if makestruct has been
    # run before. We're checking to see if the original linelist+modified exists as that's what we create.
    try:
        # During the first run '_modified' isn't part
        # of line_list so we need to make sure we try opening it with _modified attached in case we created it in
        # a completely different run before, but still on this same object, as we keep the linelist around after the run
        if '_modified' in makestruct_dict['line_list']:
            sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
                                            makestruct_dict['line_list'], "rb"))
        # If the linelist hasn't been modified before in THIS run (a.k.a first Synthsize run), we double check to see
        # if we made it in a run at another time.
        else:
            # [:-5] removes the .fits part of the file name
            sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
                                            makestruct_dict['line_list'][:-5] + '_modified.csv', "rb"))
            makestruct_dict['line_list'] = makestruct_dict['line_list'][:-5] + '_modified.csv'

        print("Line_merge data file found, this will be faster!")
    # If we never ran this object before, we must modify the data in the line list (that we have carried forward in
    # master line list) to be used with PySME. Modifications include adding ionization to species name and more.
    # Because this takes a while, we then save it as a file to be used in later runs if we ever want to analyse the
    # object again, or any others in the same data release/linelist
    except FileNotFoundError:
        print("No line_merge data file created previously. "
              "\nRunning a line merger to modify the data to allow for pysme "
              "input. \nThis could take several minutes, and will "
              "create a data file for later use for any star in this data release.")
        # Loads the data from the linelist initially
        master_line_list = load_linelist(makestruct_dict)

        # This function modifies the linelist data and saves the output in the OUTPUT/ directory
        pysme_readlines.run_merger(master_line_list, makestruct_dict['line_list_location'],
                                   makestruct_dict['line_list'])
        # The linelist is now modified so we want to always call this new one as it has an updated depth
        makestruct_dict['line_list'] = makestruct_dict['line_list'][:-5] + '_modified.csv'

        # Now we know it exists, we can do what we were trying to before.
        sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
                                        makestruct_dict['line_list'], "rb"))

    # Finds a file in GALAH with the same parameter information as ours. However, our depth is not correct,
    # as it is stored in these files, so we open it to find the newly corrected depth. Repeated with all new parameters
    sme_linelist['depth'] = pysme_interpolate_depth.reduce_depth(makestruct_dict)
    # We don't save and load this as it changes based on the alllines index. Which can vary. These are the indexes
    # for the atomic lines in our linelist that lie within our segments.
    # data_index = desired_atomic_indexes(makestruct_dict, sme_linelist)
    
    # Unpack them here to modify later.
    """line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        sme_linelist['atomic'][data_index], sme_linelist['lande'][data_index], \
        sme_linelist['depth'][data_index], sme_linelist['lineref'][data_index], \
        sme_linelist['species'][data_index], sme_linelist['line_extra'][data_index], \
        sme_linelist['line_term_low'][data_index], sme_linelist['line_term_upp'][data_index], \
        sme_linelist['line_lulande'][data_index]"""
    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        sme_linelist['atomic'], sme_linelist['lande'], \
        sme_linelist['depth'], sme_linelist['lineref'], \
        sme_linelist['species'], sme_linelist['line_extra'], \
        sme_linelist['line_term_low'], sme_linelist['line_term_upp'], \
        sme_linelist['line_lulande']

    return line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande


# Finds a file in GALAH with the same parameter information as ours. However, our depth is not correct,
# as it is stored in these files, so we open it to find the newly corrected depth.
def load_linelist(makestruct_dict):
    """
    Input:
        makestruct_dict: Information on the linelist file
    Output:
        master_line_list: Line list information on the desired atomic lines with the correct depth values.
    """
    # We now use that linelist to find the indexes of the values we want. Must do this for each new line list each run.
    # We start with the galah master file, and change it to our own file after creating it in the outer function
    # But we don't want to redo the depth modifications after we make our modified linelist so we change its name
    # to a unique one that indicates this work has been done.

    # The first run doesn't have a linelist modified created but our linelistlocation is still the same from
    # the previous run so it can't find the regular galah H file in OUTPUT/LINELIST. so it's an exception to that error.
    print("First run, opening original line list.")
    master_line_list = fits.open(makestruct_dict['original_location'] + makestruct_dict['line_list'])[1]

    return master_line_list


# Taking in the linelist to see which atomic lines lie within our segments. This is always the modified pickle
# file as it happens after readlines which creates a pickle file from the fits file.
def desired_atomic_indexes(makestruct_dict, modded_line_list):
    """
    Input:
        makestruct_dict: Information on the segment mask file
    Output:
        all_lines_index: Indexes of the atomic lines that lie within our segments.
    """
    # Grabs segment wavelength information with updated resolutions
    segment_mask_data_with_res = adjust_resolution(makestruct_dict)
    # the list that we turn into an np array containing the indexes of the parts of the master file we care for
    # Must apply it to sme rdlin to save time.
    all_lines_index = []
    # We're taking the desired lines in the linelist that are inside our segment mask and above min depth
    # or are Hydrogen( :, 2) is the lambda (wavelength) which is set in readline.
    # It was done that way in IDL so I just had to copy it.
    for wavelength_band in range(0, len(segment_mask_data_with_res['Wavelength_Start'])):
        # Finds the lines in the master linelist that are inside our wavelength stars and end, or is hydrogen
        single_line = np.where(np.logical_and(np.logical_and(
            modded_line_list['atomic'][:,2] >= float(segment_mask_data_with_res[
                                                         'Wavelength_Start'][wavelength_band]),
            modded_line_list['atomic'][:,2] <= float(segment_mask_data_with_res[
                                                         'Wavelength_End'][wavelength_band])),
            np.logical_or(modded_line_list['depth'] > makestruct_dict['depthmin'],
                          str(modded_line_list['species']) == 'H')))
        # If there are no non broad lines, all_lines_index are just broad, else combine the two.
        # These are the INDEXES
        # but we turn it into the real thing when creating the smaller linelist of obsname.fits for makestruct
        # all_lines_index is plines in idl
        all_lines_index.extend(single_line[0])

    broad_line_index = []
    if 'broad_lines' in makestruct_dict:
        # list over numpy array to store the indexes of the broad lines where they equal the linelist.
        for broadline in makestruct_dict['broad_lines']:
            broad_line_index.extend((np.where(broadline == modded_line_list['atomic'][:, 2]))[0])

    # If we have broad lines in the local variable definitions we want to add them.
    # Out of loop to prevent repeated adding of the same ones.
    if 'broad_lines' in makestruct_dict:
        # all lines is plines in idl. Contains the regular lines in the wavelength bands, and the broad ones
        # that impact it but with peaks that are out of the range.
        # So theoretically, it could try to concatenate b l i if it doesn't exist if the previous if statement
        # is skipped, but it can't happen as they have the same if requirements, so what's the issue?
        # np.sort to keep in numerical order.
        all_lines_index.extend(broad_line_index)
    # Avoid pesky duplicates of broad lines which we otherwise get.
    all_lines_index = np.unique(np.asarray(all_lines_index))

    return all_lines_index


# Load the segmnent mask to grab the appropriate resolution information to be able to add new resolutions for our
# created wavelength peaks.
def adjust_resolution(makestruct_dict):
    """
    Input:
        makestruct_dict: Information on the segment mask for resolution.
    Output:
        segment_mask_data_with_res: Segment information with new resolutions for wavelength peaks.
    """

    segment_mask_data_with_res = pd.read_csv(
        "GALAH/DATA/" + makestruct_dict['segment_mask'], delim_whitespace=True, header=None,
        names=["Wavelength_Start", "Wavelength_End", "Resolution", "comment", "overflow"],
        engine='python', skipinitialspace=True, usecols=["Wavelength_Start", "Wavelength_End", "Resolution"])

    # ~ asks for negation, removes any row that starts with ; in the first column.
    try:
        segment_mask_data_with_res = segment_mask_data_with_res[~segment_mask_data_with_res[
            'Wavelength_Start'].str.startswith(";")]
    except AttributeError:
        print("No comments found in Segm, would throw an error if not for 'try' in Sp4.")
    # Reset the index to account for the now missing values
    segment_mask_data_with_res = segment_mask_data_with_res.reset_index(drop=True)
    # Sorts in ascending order of wavelength of starting wavelength, and re-orders the others.
    segment_mask_data_with_res.sort_values(by=['Wavelength_Start', 'Wavelength_End'])

    # Now we get the resolution from the resolution map.
    # We make a list to avoid calling pandas data frame repeatedly,
    temporarywavelengthpeaklist = []
    for wavelength_band_resolution in range(0, len(segment_mask_data_with_res['Resolution'])):
        # Calculating the peak wavelength
        temporarywavelengthpeaklist.append(0.5 * (float(segment_mask_data_with_res.loc[
                                                            wavelength_band_resolution, 'Wavelength_Start']) +
                                                  float(segment_mask_data_with_res.loc[
                                                            wavelength_band_resolution, 'Wavelength_End'])))
        # Interpolating the resolution at the wavelength peak and replacing the resolution of that index with it
        segment_mask_data_with_res.loc[wavelength_band_resolution, 'Resolution'] = makestruct_dict['interpolation'](
            temporarywavelengthpeaklist[wavelength_band_resolution]) * makestruct_dict['resolution_factor']

    return segment_mask_data_with_res


# Indexes the total atomic lines to those that fit within the wavelength segments we have (which themselves were
# indexed to the segment mask file)
def atomic_lines_in_segments(makestruct_dict, segment_begin_end, line_atomic, depth):
    """
    Input:
        Makestruct_dict: For depth minimum, and broad line information
        segment_begin_end: To compare to atomic lines to find those within segments.
        line_atomic: Find the atomic lines in the segments of segment_begin_end.
        depth: Compare to depth min to only select lines above a minimum depth
    Output:
        desired_atomic_lines_index: Indexed atomic data to limit to the wavelength segments.
    """

    # The list we put the indexes of the esired atomic lines into
    desired_atomic_lines_index = []
    number_of_segments = len(segment_begin_end)
    buffer = 0.7
    # Index the atomic data for only ones in our wavelength segments, with a given buffer.
    for segment in range(0, number_of_segments):

        # Here we reference the data created in sme_rdlin, which took about 70 seconds for the full 300k lines.
        # we test to see which parts are in the segments we have in segment_begin_end which is segment mask.
        desired_atomic_lines_index.extend(np.where(
            np.logical_and(
                np.logical_and(line_atomic[:, 2] > segment_begin_end[segment, 0] - buffer,
                               line_atomic[:, 2] < segment_begin_end[segment, 1] + buffer),
                depth > makestruct_dict['depthmin']))[0])

        # If broad lines are near (but not inside), select those too.
        for broad_line_single in makestruct_dict['broad_lines']:

            # If any line is within 100a of a broadline we'll include it
            if np.any(abs(broad_line_single - segment_begin_end[segment]) < 100):
                # Where does this broad line exist in our atomic line array? rounds automatically in np where for
                # the array.
                # We have duplicates in line_atiomic and (therefore?) the d_a_l_index, do we want to remove those?
                # We add all(?) the lineatomic to the index, but that makes sense as it is preselected to include
                # the indexes of the wavelengths inside our segments.
                desired_atomic_lines_index.extend(np.where(line_atomic[:, 2] == broad_line_single)[0])

    return desired_atomic_lines_index


# Now we see which are in the line list, which is the list of important wavelengths to look out for, run during Solve.
# This is if we aren't doing atomiclinesinsegments. Different to when we indexed the wavelengths as that used seg_mask
def atomic_lines_in_linemask(makestruct_dict, line_atomic, depth, linemask_data):
    """
    Input:
        Makestruct_dict: For depth minimum, and broad line information
        linemask_data: To compare to atomic lines to find those within the linemasks.
        line_atomic: Find the atomic lines in the segments of segment_begin_end.
        depth: Compare to depth min to only select lines above a minimum depth
    Output:
        desired_atomic_lines_index: Indexed atomic data to limit to the linemasks.
    """
    # the list we put the indexes of the esired atomic lines into
    desired_atomic_lines_index = []

    # de in makestruct.pro. A buffer to fit AROUND The edges of the linemask
    buffer = 0.7

    # Used to take the certain number of linemask indexes
    nrline = 20 + ((8000 - makestruct_dict['effective_temperature']) / 1.3E3) ** 4
    # We see which parts of our atomic line data lies within the linemask (the important wavelengths) as well as
    # identifying the absorption wavelengths if we can, and the broad lines. We get a lot of duplicates here
    for line in range(0, len(linemask_data['Sim_Wavelength_Peak'])):

        # We find the wavelengths within the ranges, but this time it'sthe linemask ranges, not segments.
        inside_linemask_index = np.where(
            np.logical_and(
                np.logical_and(
                    line_atomic[:, 2] > linemask_data['Sim_Wavelength_Start'][line] - buffer,
                    line_atomic[:, 2] < linemask_data['Sim_Wavelength_End'][line] + buffer),
                depth > makestruct_dict['depthmin']))
        # We reverse it to take the the largest indexes which also mean the highest wavelengths
        inside_linemask_index = np.flip(inside_linemask_index)

        # Take either the last nrline number or all the index, whatever is smaller.
        desired_atomic_lines_index.extend((inside_linemask_index[0:min([nrline, len(inside_linemask_index)])])[0])

        # ;always select the main line if it's present
        peak_index = (np.where(line_atomic[:, 2] == float(linemask_data['Sim_Wavelength_Peak'][line])))[0]

        if peak_index.size != 0:
            desired_atomic_lines_index.extend(peak_index)
        else:
            print("No peak line (",
                  linemask_data['Sim_Wavelength_Peak'][line],
                  ") available in the atomic line list for line", line,
                  "(", linemask_data['Sim_Wavelength_Start'][line], "to", linemask_data['Sim_Wavelength_End'][line],
                  ")")

        # And of course, always select broad lines when close just like before (but that was broad lines in segments)
        for broad_line_single in makestruct_dict['broad_lines']:
            # If any line is within 100a of a broadline we'll include it
            if np.any(abs(broad_line_single - float(linemask_data['Sim_Wavelength_Peak'][line])) < 100):
                print("Broad line", broad_line_single, " at", linemask_data['Sim_Wavelength_Peak'][line])
                desired_atomic_lines_index.extend(np.where(line_atomic[:, 2] == broad_line_single)[0])

    return desired_atomic_lines_index


# Strings to tell PYSME what kind of atmosphere we're looking for. Similar to in Galah1 but more specific atmosphere
# variables.
def set_atmosphere_string():
    """
    Output:
        atmosphere_depth: Type of depth
        atmosphere_interpolation: Method of interpolation
        atmosphere_geometry: Assumptumption of geometry (E.G PP = Plane parallel)
        atmosphere_method: How to organise the atmosphere.
    """
    # Tau is the optical depth at some wavelength reference for continuum to determine the deepest point of interest.
    # You can plot T against log tau and when x is 0, it is opaque so anything below that we are viewing.
    # Rhox is the other option of column mass (accumulated mass) of the atmosphere. These are both pretty bad as
    # they cause some kind of spike in the abundance of stars at temperatures but people are working on replacing them
    # with a neural network.

    atmosphere_depth = "RHOX"
    atmosphere_interpolation = "TAU"
    # Plane parallel = PP
    atmosphere_geometry = "PP"
    atmosphere_method = "grid"

    return atmosphere_depth, atmosphere_interpolation, atmosphere_geometry, atmosphere_method


def dataframe_the_linelist(linelist):
    """
    Input:
        Linelist: The atmoic line data including species, ionization, and depth and lande. Used to convert its data
                    into a PySME format.
    Output:
        linedata: A slightly modified version of the linelist, both returned and saved to an OUTPUT/LINELIST/ directory
    """
    data = {
        "species": linelist['species'],
        "atom_number": linelist['atomic'][:, 0],
        "ionization": linelist['atomic'][:, 1],
        "wlcent": linelist['atomic'][:, 2],
        "excit": linelist['atomic'][:, 3],
        "gflog": linelist['atomic'][:, 4],
        "gamrad": linelist['atomic'][:, 5],
        "gamqst": linelist['atomic'][:, 6],
        "gamvw": linelist['atomic'][:, 7],
        "lande": linelist['lande'],
        "depth": linelist['depth'],
        "reference": linelist['lineref']
    }

    # PySME doesn't import lineref as an array, so we do something to combine all values in the array each line to
    # allow us to modify it in the same way
    linerefjoined = np.array(([''.join(array) for array in linelist['lineref']]))

    """
    The errors in the VALD linelist are given as strings in various formats. (e.g. “N D”, “E 0.05”). 
    Parse_line_error takes those and puts them in the same format, I.e. relative errors.
    Now at the moment at least, the relative errors on the line data, do not influence the calculations in any way. 
    So if you can’t find them in your data, it won’t be a problem.
    """
    error = [s[0:11] for s in linerefjoined]  # .strip removed
    error = np.ones(len(error), dtype=float)
    # We set error to be a vague amount as it does not yet influence the calculation.
    error.fill(0.5)
    data["error"] = error
    data["lande_lower"] = linelist['line_lulande'][:, 0]
    data["lande_upper"] = linelist['line_lulande'][:, 1]
    data["j_lo"] = linelist['line_extra'][:, 0]
    data["e_upp"] = linelist['line_extra'][:, 1]
    data["j_up"] = linelist['line_extra'][:, 2]
    data["term_lower"] = [t[10:].strip() for t in linelist['line_term_low']]
    data["term_upper"] = [t[10:].strip() for t in linelist['line_term_upp']]

    # We have an issue with lineref being 2d. Needs to be 1d. But it contains all 7 arrays from IDL
    # So instead we input the previously mentioned linerefjoined to conglomorate all the arrays into one row per row.
    data['reference'] = linerefjoined
    linedata = pd.DataFrame.from_dict(data)
    linedata.to_pickle("OUTPUT/LINELIST/pysme_linelist_dataframe")
    return linedata


# IDL saves it as an inp file, but we don't like that. This is a copy of what IDL does and where it saves it,
#  Called from make_struct function. We input it directly by importing sme,
def store_sme_input(makestruct_dict):
    """
    Input:
        makestruct_dict: Name of object, and all input data to save.
    Output:
        None: But the original SME input file is saved to the OUTPUT/ folder.
    """
    print("Saving the SME input to 'OUTPUT/FullOutput/",
          makestruct_dict['obs_name'], "_Original_SME_input.pkl'")
    input_file = open(r'OUTPUT/FullOutput/' + makestruct_dict['obs_name'] + '_Original_SME_input.pkl', 'wb')

    pickle.dump(makestruct_dict, input_file)
    input_file.close()


# A function to take the wavelength etc arrays that are just a single array, and turn it into a list of arrays seperated
# per segment, as required from pysme.
def segmentise_arrays(wavelengths, flux, flux_error, flagged_flux, total_segments):
    """
    Input:
        Non segmentised arrays
    Output:
        The same arrays but sementised. Each array has many arrays inside of it, each unique to our segment_mask choices
    """
    # Create lists to store the arrays of each type in.
    pysme_wave_list, pysme_flux_list, pysme_error_list, pysme_flagged_flux_list = ([], [], [], [])
    # Run through each segment and find the indexes of wavelengths inside it.
    for segment in total_segments:
        wavelength_segment_indexes = np.where(np.logical_and(wavelengths >= segment[0], wavelengths <= segment[1]))
        # Then apply the indexes to wavelength arrays, and the other flux etc.
        pysme_wave_list.append(wavelengths[wavelength_segment_indexes])
        pysme_flux_list.append(flux[wavelength_segment_indexes])
        pysme_error_list.append(flux_error[wavelength_segment_indexes])
        pysme_flagged_flux_list.append(flagged_flux[wavelength_segment_indexes])

    return pysme_wave_list, pysme_flux_list, pysme_error_list, pysme_flagged_flux_list
