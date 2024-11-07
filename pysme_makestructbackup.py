"""
We read the segment mask segm, removing the comment lines and columns.
We proceed to open the ccd1-4 files and take their wavelength and resolution out to plot a resolution mask and
interpolating to be able to find resolution at peak wavelengths that we calculate at that point of the segment mask.
We open the 4 ccd fits files to produce a phoneenumber_Sp file that contains Wavelength array, observed flux sign (sob)
uncertainity in flux (uob)(essentially error), smod (not important now) and mod (Idk) that we call in make_struct
to modify the wavelength array by the radial velocity doppler shift and make some sme variables.
Running create_structure should be the only thing needed.
Running it for the first time will take a good 5-10 minutes, but after that it should take a few seconds at a time. It's
a trade off.
"""

import pysme_readlines
import pysme_run_sme
import pysme_idl_conversion
import pysme_interpolate_depth
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import date
import pickle
from astropy.io import fits
import pysme_update_logg

# runs all the rest of 'em and creates the sme structure!
"""Run this one! Makestruct dict is set in pysmegalah4, and contains 
makestruct_dict.update({'setup_for_obs': setup_for_obs, 'resolution_factor' : resolution_factor,
                            'obs_name': obs_name,           'object_for_obs': object_for_obs, 'line_cores': line_cores,
                            'depthmin': depthmin,           'broad_lines': broad_lines,
                            'atmosphere_grid_file': atmosphere_grid_file,
                            'auto_alpha': auto_alpha,       'nonlte_flag': nonlte_flag,
                            'nonlte_element_flags': nonlte_element_flags,
                            'nonlte_atmosphere_grids': nonlte_atmosphere_grids,
                            'effective_temperature': effective_temperature,
                            'gravity': gravity,
                            'radial_velocity_global': radial_velocity_global,
                            'microturbulence_velocity': microturbulence_velocity,
                            'macroturbulence_velocity': macroturbulence_velocity,
                            'rotational_velocity': rotational_velocity,
                            'iterations': iterations,       'radial_velocity_flag': radial_velocity_flag})
"""


# aafp is modified in galah ab but not in sp or makestruct
def create_structure(makestruct_dict, reduction_variable_dict,
                     atomic_abundances_free_parameters=np.zeros(99), normalise_flag=False, run="Synthesise"):
    # ; set remaining variables, usually fixed?
    # Have I set these elsewhere with better names? @@@@@@@@@@@@@
    # Global correction factor to all van der Waals damping constants. Values of
    # 1.5 to 2.5 are sometimes used for iron.
    global_waals_correction = 1
    # gam6

    # Minimum accuracy for linear spectrum interpolation vs. wavelength.
    wavelength_interpolation_accuracy = 0.005
    # accwi

    # Minimum accuracy for sme.sint (Specific intensities on an irregular wavelength grid given in sme.wint.)
    # at wavelength grid points in sme.wint. (Irregularly spaced wavelengths for specific intensities in sme.sint.)
    # Values below 10-4 are not meaningful.
    specific_intensity_accuracy_min = 0.005
    # accrt

    # If we have any value for atomic abundances free parameters then we combine it and the globfree (vrad etc)
    # Very unsure about this as we have never had to do it so far @@@@@@@@@@@
    if np.any(atomic_abundances_free_parameters):
        makestruct_dict["global_free_parameters"] = np.concatenate(
            np.asarray(makestruct_dict["global_free_parameters"]),
            atomic_abundances_free_parameters, axis=None)

    # Else we just use free glob. But add Vsin for non-normalisation runs if rot_vel is above 1.
    if run == "Solve":
        if makestruct_dict['rotational_velocity'] > 1 and 'VSINI' not in makestruct_dict:
            makestruct_dict["global_free_parameters"].append('VSINI')
        else:
            # Feels weird, are we really ignoring rot velocity after all this?
            makestruct_dict['rotational_velocity'] = 1

    # Fractional change in sme.chisq (Chi-square weighted by the observed spectrum
    # flux for each iteration below which convergence is assumed.) below which convergence is assumed.
    chi_square_convergence = 0.001
    # chirat

    # Number of "equal-area" angles at which to calculate specic intensity.

    specific_intensity_angles = 7
    # nmu

    # Dunno
    # obs_type = 3

    # Type of prole used for instrumental broadening. Possible values are gauss,
    # sinc, or table. See Section 3.4.
    broadening_profile = "table"
    # iptype

    # The equal-area midpoints of each equal-area annulus for which specific inten-
    # sities were calculated. values for Gaussian quadrature are not conducive
    # to subsequent disk integration (???).    Is mu.
    intensity_midpoints = np.flip(
        np.sqrt(0.5 * (2 * np.arange(1, specific_intensity_angles + 1)) / specific_intensity_angles))

    id_date = date.today()

    # atmo_pro = 'interp_atmo_grid'
    try:
        atmosphere_grid_file = makestruct_dict['atmosphere_grid_file']
    except AttributeError or NameError:
        # If we don't have a previously set atmosphere grid we use a backup 2012.
        # Try is faster as we usually will have it set.
        print("Using marcs2012 instead")
        atmosphere_grid_file = 'marcs2012.sav'
    # atmo_grid_vers
    atmosphere_grid_version = 4.0

    # Produce the basic blocks of which wavelengths etc are inside our segment masks.
    ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
        segment_begin_end, wavelength_start_end_index, ccd_resolution_of_segment_array = \
        object_array_setup(makestruct_dict)

    number_of_segments = len(wavelength_start_end_index)

    continuum_mask_data, linemask_data = open_masks(makestruct_dict)

    # we always want an array for pysme. but what does this even DO.
    radial_velocity = np.zeros(number_of_segments)

    continuum_scale = np.ones(number_of_segments)

    # Only normalise the data if that's how this file is called.
    # IS those correct way to determine for prenorm? @@@@@@@@@@@@
    # Runs the normalisation run to take flux own from the 10s of thousands to below 1 duh
    print("\n\n\nBIG OL LEN OF WAVLENGTH AT END", ccd_wavelengths_inside_segmask_array, "\n\n\n")

    if normalise_flag:
        print("Running pre-normalise.")
        ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array, \
            ccd_flux_norm_error_inside_segmask_array =\
            pre_normalise(ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array,
                          ccd_flux_error_inside_segmask_array, segment_begin_end)

    # Fixed cscale flag for stellar parameter iterations, as instruted by karin
    # AFTER prenorm has been run once, run sme will have saved these normalise fluxes, and they have been loaded
    # instead of un-normalied ones from galahsp3 earlier in load_spectra, so we don't need to modify them.
    else:
        # Just to change to the new normalised variable name.
        ccd_flux_norm_inside_segmask_array = ccd_flux_inside_segmask_array
        ccd_flux_norm_error_inside_segmask_array = ccd_flux_error_inside_segmask_array

    # Produce the array that tells us what fluxes to ignore, which are contiuum or peak. make_mob in idl.
    flagged_fluxes = create_observational_definitions(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                                      ccd_flux_norm_inside_segmask_array,
                                                      ccd_flux_norm_error_inside_segmask_array,
                                                      segment_begin_end)

    # make_line, we get the indexed atomic data and all. Except it's indexed from
    # makestruct_dict['obs_name']+makestruct_dict['line_list']+'_index.csv' from spr part 4...
    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        produce_indexed_atomic_data(makestruct_dict, segment_begin_end, linemask_data, run=run)

    print("line atomic:", line_atomic)
    # If there are no segments within atomic lines, break out and move on.
    """if np.all(line_atomic) == 0:
        print("No lines in segment, moving on.")
        return"""

    # Array of ags (0 or 1), specifying for which of the spectral lines the gf values
    # are free parameters.
    spectral_lines_free_parameters = np.zeros(len(species))  # gf_free
    # Tau is the optical depth at some wavelength reference for continuum to determine the deepest point of interest.
    # You can plot T against log tau and when x is 0, it is opaque so anything below that we are viewing.
    # Rhox is the other option of column mass (accumulated mass) of the atmosphere. These are both pretty bad as
    # they cause some kind of spike in the abundance of stars at temperatures but people are working on replacing them
    # with a neural network.
    # Something to do with setdefaultvalue but I can't figure out if it does anything else but set to this string.
    # They're meant to only be set if newsme but I can't find them being made elsewhere

    # Strings to tell PYSME what kind of atmosphere we're looking for.
    atmosphere_depth, atmosphere_interpolation, atmoshpere_geometry = set_atmosphere_string()

    """; Temporary fix for stagger grid, which has only EITHER tau or rhox depth scale available:
  if strmatch(atmogrid_file, '*stagger-t*') then atmo_depth = 'TAU'
  if strmatch(atmogrid_file, '*stagger-r*') then atmo_interp = 'RHOX'"""

    "We create the linelist here that pysme requires. It uses only data from the sme input, but is for 'legacy reasons'"
    # We create a dictionary of the indexed linelist to be input into PySME using the names they look for. We do it here
    # As we are modifying the parts like line_atomic earlier in other places.
    linelist_dict = {'atomic': line_atomic,
                     'lande': lande_mean,
                     'depth': depth,
                     'lineref': data_reference_array,
                     'species': species,
                     'line_extra': j_e_array,
                     'line_term_low': lower_level,
                     'line_term_upp': upper_level,
                     'line_lulande': lu_lande}

    # Calling on IDLtoPySME_linelist then converts that to a data frame they're looking for. To be honest,
    # could be done in sme_rdlin, but it's not like that in IDL, as this is a new addition to match up with PySME needs.
    linelist = pysme_idl_conversion.dataframe_the_linelist(linelist_dict)

    # Turning our single large array into a list of smaller segmented arrays to comply with pysme iliffe standards.
    # Probably should have done this originally, but now al the code is dependent on them being arrays so this seems
    # like an easier fix. Plus arrays ARE better to handle for large data so it might still be correct.
    iliffe_wave, iliffe_flux_norm, iliffe_error_norm, iliffe_flags = \
        segmentise_arrays(ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                          ccd_flux_norm_error_inside_segmask_array, flagged_fluxes, segment_begin_end)

    # Always two for fits files I think.
    short_format = 2
    # Pyssme combines ab and glob free (and pname but not sure what that is)
    # fitparameters = [global_free_parameters, atomic_abundances_free_parameters]
    # finally we set the dictionary! Contains the info for sme! Should grav be logg for pysme? feh becomes monh
    # obs_name renamed to object
    # Wave is renamed to wob, but "wave still exists but as an Iliffe vector" as do spec,
    # uncs, mask, synth but we don't have synth here.
    sme_input = {'version': atmosphere_grid_version,                   'spec': iliffe_flux_norm,
                 'id': id_date,                                        'auto_alpha': makestruct_dict['auto_alpha'],
                 'teff': makestruct_dict['effective_temperature'],     'depth': depth,
                 'logg': makestruct_dict['gravity'],                   'species': species,
                 'feh': makestruct_dict['metalicity'],                 'sob': ccd_flux_norm_inside_segmask_array,
                 'field_end': makestruct_dict['field_end'],            'uob': ccd_flux_norm_error_inside_segmask_array,
                 'monh': makestruct_dict['metalicity'],                'mu': intensity_midpoints,
                 'vmic': makestruct_dict['microturbulence_velocity'],  'abund': makestruct_dict['abundances'],
                 'vmac': makestruct_dict['macroturbulence_velocity'],  'lande': lande_mean,
                 'vsini': makestruct_dict['rotational_velocity'],      'chirat': chi_square_convergence,
                 'vrad': radial_velocity,                              'line_lulande': lu_lande,
                 'vrad_flag': makestruct_dict['radial_velocity_flag'], 'atomic': line_atomic,
                 'cscale': continuum_scale,                            'nmu': specific_intensity_angles,
                 'gam6': global_waals_correction,                      'mask': iliffe_flags,
                 'accwi': wavelength_interpolation_accuracy,           'accrt': specific_intensity_accuracy_min,
                 'maxiter': max(makestruct_dict['iterations']),        'line_term_low': lower_level,
                 'atmo': {'source': atmosphere_grid_file,              'method': 'grid',
                          'depth': atmosphere_depth,                   'interp': atmosphere_interpolation,
                          'geom': atmoshpere_geometry},                'ipres': ccd_resolution_of_segment_array,
                 'object': makestruct_dict['obs_name'],                'iptype': broadening_profile,
                 'ab_free': atomic_abundances_free_parameters,         'uncs': iliffe_error_norm,
                 'gf_free': spectral_lines_free_parameters,            'lineref': data_reference_array,
                 'line_term_upper': upper_level,                       'nseg': number_of_segments,
                 'short_format': short_format,                         'line_extra': j_e_array,
                 'wran': segment_begin_end,                            'wave': iliffe_wave,
                 'wob': ccd_wavelengths_inside_segmask_array,          'wind': wavelength_start_end_index[:, 1],
                 'balmer_run': makestruct_dict['balmer_run'],          'mob': flagged_fluxes,
                 'cscale_flag': makestruct_dict['continuum_scale_flag'],
                 'fitparameters': makestruct_dict["global_free_parameters"],

                 }
    print("\n\n\nBIG OL LEN OF WAVLENGTH AT SME INPUT", ccd_wavelengths_inside_segmask_array, "\n\n\n")

    # Only do so if nlte is set on (1)
    if makestruct_dict['nonlte_flag']:
        nltestruct = {'nlte_pro': 'sme_nlte', 'nlte_elem_flags': makestruct_dict['nonlte_element_flags'],
                      'nlte_subgrid_size':
                          [3, 3, 3, 3], 'nlte_grids': makestruct_dict['nonlte_atmosphere_grids'], 'nlte_debug': 1}
        # In idl this is made super weirdly, there's a 'NLTE' in there if it's newsme?
        # Is it a dictionary with create_struct
        # or not? It says they are assigned values but it's so weird. what I've done is the literal translation of
        # the code but it's IDL so who fricking knows.
        sme_input['NLTE'] = nltestruct

    # Save our input dict using pickle to allow for later runnin of sme manually too. Only run on the first attempt
    # as those are the original inputs.
    if not makestruct_dict['load_file']:
        store_sme_input(sme_input, makestruct_dict)
    "And here we go. We finally run pysme."
    pysme_run_sme.start_sme(makestruct_dict['obs_name'], reduction_variable_dict, sme_input, linelist, run)
    return makestruct_dict





# ccd_flux_inside_segmask_array is sob
# ccd_wavelengths_inside_segmask_array is wave

# Need to check with Karin and Sven that this works appropriately @@@@@@@@@@@@@@@@@@@@@
# wavelegnth x   flux y  order of fn( 0=linr)
# We use fit_param to modify the index selections, but it is a tuple here. So we use its 1st value
# We're inputting the wavelengths and fluxes of the readings in the first segment


def autonormalisation(wavelength_array, flux_array, polynomial_order, fit_parameters):
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

    # Using numpy polyfit to replace the idl Robust_poly_fit, is this a fair replacement? @@@@@@@@@@@@@@@@
    for polyfit_loop in range(0, 99):
        # Gets the coefficients for a fit of the order polynomial_order
        polyfit_coefficients = np.polynomial.polynomial.polyfit(wavelength_array, flux_array, polynomial_order)

        # Uses these to get an array of the y values of this line
        fitted_flux = np.polynomial.polynomial.polyval(wavelength_array, polyfit_coefficients)

        # Creates an array of the error (sigma) of the line compared to original data, to find outliers by getting the
        # standard deviation of the difference
        fitted_sigma = np.std(flux_array - fitted_flux)
        # Find the fluxes that exist below the linear fit + error but above the lower error boundary (* p or something)
        # So not outliers, but inliers :) we take the first value from the output which is the index array itself
        # IDL is if o eq 2 then ind=where(fo lt y+2.*s and fo gt y-p*s)
        # where r=robust_poly_fit(wo,fo,o,y,s), o is polynomial order, w and f are wl and flux, y and s are then rturnd.
        inlier_index = (np.where(np.logical_and(flux_array < (fitted_flux + (2 * fitted_sigma)),
                                                (flux_array > (fitted_flux - (fit_parameters * fitted_sigma))))))[0]

        # If poly_order is messed up, we just stick with a value of 1 to keep everything the same
        continuous_function = 1
        # cont flux is c in idl.
        if polynomial_order == 2:
            continuous_function = polyfit_coefficients[0] + (polyfit_coefficients[1] * original_wave) \
                                  + (polyfit_coefficients[2] * original_wave ** 2)
        elif polynomial_order == 1:
            continuous_function = polyfit_coefficients[0] + (polyfit_coefficients[1] * original_wave)

        # Stops when no more convergence occurs I suppose, breaks the loop. Again, np where gives a tuple with dtype
        # the second condition uses the original non edited wavelength array
        if len(inlier_index) == len(wavelength_array) or len(inlier_index) / len(original_wave) <= 0.1:
            #print("Converged after", polyfit_loop, "loops")
            break
        if polyfit_loop >= 98:
            print("Did not converge")
            break
        # Replace the array with only values that lie inside the error boundaries we have.
        wavelength_array = wavelength_array[inlier_index]
        flux_array = flux_array[inlier_index]
    # This is 'stopp' in idl. These variables are unused @@@@
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
    # Something to do with further sigma calculations again  maybe. idl has it as sn. This is unused @@@@
    sigma_flux_comparison = 1 / (np.std(flux_array[inlier_index] / [continuous_function_additional]))
    """import matplotlib.pyplot as plt
    plt.plot(wavelength_array, flux_array)
    plt.plot(wavelength_array, fitted_flux)
    plt.show(block=True)
    """
    return continuous_function


# Two things (I know, should only be one): changes the depth o the linelist if we haven't run through
# makestruct before, and also finds the indexes of the lines we are interested in. Can probably put into galah4
# and import the result to be honest, and would avoid doing this again each time.
def line_index_depth(makestruct_dict):

    # We now use that linelist to find the indexes of the values we want. Must do this for each new line list
    # We start with the galah master file, and change it to our own file aftercreating it in the indexed_Stell...info
    # But we don't want to redo the depth modifications after we make our modified linelist.
    # We try because during the balmer run we've already changed the location, but the linelist is reset back
    # to a basic galah_H so it can't find it in the OUTPUT location!
    print("linelist", makestruct_dict['line_list_location'] + makestruct_dict['line_list'])

    try:
        # If it's modified, we saved it as a pickle file csv so we can load it with pickle. There was some bug
        # with fits.open due to a pickle update or SOMETHING update.
        if 'modified' in makestruct_dict['line_list']:
            print('modified')
            master_line_list = pickle.load(open(makestruct_dict['line_list_location'] + makestruct_dict['line_list'], "rb"))
        else:

            master_line_list = fits.open(makestruct_dict['line_list_location'] + makestruct_dict['line_list'])[1]
            master_line_list.data['depth'] = pysme_interpolate_depth.reduce_depth(makestruct_dict)

            print("not modified")

    # The first balmer run doesn't have a linelist modified created but our linelistlocation is still the same from
    # the previous run so it can't find the regular
    # galah H file in OUTPUT/LINELIST. I'm beginning to think this is poorly written code. HMM.
    # OH well what can you do. I just wanted to make everything a variable so it was easy to modify if needed
    except FileNotFoundError:
        print("File not found error, balmer run")
        master_line_list = fits.open(makestruct_dict['original_location'] + makestruct_dict['line_list'])[1]

    # We adjust the resolutions of the segment mask using the interpolation we created in part 2. How important is this?
    # @@@@@@@@@@@@@@@@@ What does it really DO? @@@@@@@ What's it FOR? @@@@@@@
    segment_mask_data_with_res = adjust_resolution(makestruct_dict)

    all_lines_index = desired_atomic_indexes(makestruct_dict, master_line_list, segment_mask_data_with_res)

    return all_lines_index, master_line_list


# Taking in the linelist and our lambda values to see which lines are ones we want.
def desired_atomic_indexes(makestruct_dict, master_line_list, segment_mask_data_with_res):
    # the list that we turn into an np array containing the indexes of the parts of the master file we care for
    # Must apply it to sme rdlin to save time.
    all_lines_index = []

    # If it's not modified we're using fits.open so we need to add the '.data' part which we can't do if it's a pickle
    if 'modified' not in makestruct_dict['line_list']:
        master_line_list = master_line_list.data

    for wavelength_band in range(0, len(segment_mask_data_with_res['Wavelength_Start'])):
        # Finds the lines in the master linelist that are inside our wavelength stars and end, or is hydrogen
        single_line = np.where(np.logical_and(np.logical_and(
            master_line_list['lambda'] >= float(segment_mask_data_with_res[
                                                         'Wavelength_Start'][wavelength_band]),
            master_line_list['lambda'] <= float(segment_mask_data_with_res[
                                                         'Wavelength_End'][wavelength_band])),
            np.logical_or(master_line_list['depth'] > makestruct_dict['depthmin'],
                          str(master_line_list['name'][0]).strip() == 'H')))
        # Above min depth OR is hydrogen I guess
        # If there are no non broad lines, all_lines_index are just broad, else combine the two.
        # These are the INDEXES
        # but we turn it into the real thing when creating the smaller linelist of obsname.fits for makestruct
        # all_lines_index is plines in idl
        all_lines_index.extend(single_line[0])

    broad_line_index = []
    if 'broad_lines' in makestruct_dict:
        # list over numpy array to store the indexes of the broad lines where they equal the linelist. So the
        # broad lines that were find? @@@@@@@@@@@@@@@@@@@@@@
        # Not sure why we get the index yet, or do we just want the value of the broad lines?
        for broadline in makestruct_dict['broad_lines']:
            broad_line_index.extend((np.where(broadline == master_line_list['lambda']))[0])

    # If we have broad lines in the local variable definitions we want to add them.
    # Out of loop to prevent repeated adding of the same ones.
    if 'broad_lines' in makestruct_dict:
        # all lines is plines in idl. Contains the regular lines in the wavelength bands, and the broad ones
        # that impact it but with peaks that are out of the range.
        # So theoretically, it could try to concatenate b l i if it doesn't exist if the previous if statement
        # is skipped, but it can't happen as they have the same if requirements, so what's the issue?
        # np.sort to keep in numerical order.
        all_lines_index.extend(broad_line_index)
    # Avoid pesky duplicates of broad lines! Which we otherwise get.
    all_lines_index = np.unique(np.asarray(all_lines_index))

    return all_lines_index


def adjust_resolution(makestruct_dict):

    segment_mask_data_with_res = pd.read_csv(
        "GALAH/DATA/" + makestruct_dict['segment_mask'], delim_whitespace=True, header=None,
        names=["Wavelength_Start", "Wavelength_End", "Resolution", "comment", "overflow"],
        engine='python', skipinitialspace=True, usecols=["Wavelength_Start", "Wavelength_End", "Resolution"])
    # Here we auto delete the comment and overflow columns

    # These lines I guess are not needed when not using IDL created files? @@@@@@@@@@@@@@@@@@@@@@@@@
    # ~ asks for negation, removes any row that starts with ; in the first column.
    # Unsure what the str does but required.
    try:
        segment_mask_data_with_res = segment_mask_data_with_res[~segment_mask_data_with_res[
            'Wavelength_Start'].str.startswith(";")]
    except AttributeError:
        print("No comments found in Segm, would throw an error if not for 'try' in Sp4.")
    # Reset the index to account for the now missing values
    segment_mask_data_with_res = segment_mask_data_with_res.reset_index(drop=True)
    # This sorts out segments 35
    # \n i=sort(seg_st) & seg_st=seg_st[i] & seg_en=seg_en[i] & seg_ipres=seg_ipres[i]
    # Sorts in ascending order of wavelength of starting wavelength, and re-orders the others.
    # But it's already ordered? weird. Will leave it in case data comes in unordered
    segment_mask_data_with_res.sort_values(by=['Wavelength_Start', 'Wavelength_End'])

    # Now we get the resolution from the resolution map.
    # We make a list to avoid calling pandas data frame repeatedly,
    # I believe this is faster and avoids any copy errors.
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


# Produces the arrays with the stellar information from the master file for every wavelength. such as line_atomic etc.
def indexed_stellar_information(makestruct_dict):
    # The reduced amount of spectral lines we need to look at, applied to masterfile, brings it down from 300k
    # also changes the depth to the true values, done in there as it also requires depth info.
    # Only updates depth during normalisation runs. DOn't actually need half of it every time...
    data_index, master_line_list = line_index_depth(makestruct_dict)

    # We try to open the indexed linelist (with modifications made to loggf, etc) that is created if makestruct has been
    # run before. We're checking to see if the original linelist+modified exists as that's what we create
    #
    # The new location of the linelist. Either it's already there or we put a modified version there. Have to change
    # after we open the original linelist during the first run. Perhaps this is the downside of using the same variable
    # each time.
    makestruct_dict['line_list_location'] = r'OUTPUT/LINELIST/'

    try:
        # Look I wanted to have linelist and location as a variable. So during the first run _modified isn't part
        # of line_list so we need to make sure we try opening it with _modified attached in case we created it in
        # a previous run YES I KNOW IT'S BAD CODE GET OVER IT IT'S BLOODY FIDDLY
        if '_modified' in makestruct_dict['line_list']:
            sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
                                            makestruct_dict['line_list'], "rb"))
        else:
            sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
                                            makestruct_dict['line_list'][:-5] + '_modified.csv', "rb"))
            # -5 removes the .fits
            makestruct_dict['line_list'] = makestruct_dict['line_list'][:-5] + '_modified.csv'

        print("Line_merge data file found, this will be faster!")

    except FileNotFoundError:
        print("No line_merge data file created previously. "
              "\nRunning a line merger to modify the data to allow for pysme "
              "input. \nThis could take several minutes, and will "
              "create a data file for later use for any star in this data release.")

        pysme_readlines.run_merger(master_line_list, makestruct_dict['line_list_location'],
                                   makestruct_dict['line_list'])

        # The linelist is now modified so we want to always call this new one as it has an updated depth
        makestruct_dict['line_list'] = makestruct_dict['line_list'][:-5] + '_modified.csv'

        # Now we know it exists, we can do what we were trying to before.
        sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
                                        makestruct_dict['line_list'], "rb"))

        # Then we assign our desired data indexes to the big ol array
    # Then we assign our desired data indexes to the big ol array. There are several nested arrays as follows. We
    # convrt them to the pysme names later. shown with ->

    """ 
    atomic/data array: Atomic number,           -> ['atomic']
                Ionic number
                Lambda
                E_Low
                Log_gf
                Rad_damp
                Stark_Damp
                Vdw_Damp
    Lande_Mean: Lande_mean                      -> lande
    Depth: Depth                                -> depth
    Lu_Lande:   lande_lo,                       -> line_lulande
                lande_up
    j_e_array:  j_low,                          -> line_extra
                e_up, 
                j_up
    lower_level: label_low                      -> line_term_low
    upper_level: label_up                       -> line_term_upp
    Data_reference_array: lambda_ref,           -> lineref
                          e_low_ref, 
                          log_gf_ref, 
                          rad_damp_ref, 
                          stark_damp_ref, 
                          vdw_damp_ref, 
                          lande_ref

    """
    # We don't save and load this and it changes based on the alllines index. The modifications we made to the linelist
    # depen

    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        sme_linelist['atomic'][data_index], sme_linelist['lande'][data_index], \
        sme_linelist['depth'][data_index], sme_linelist['lineref'][data_index], \
        sme_linelist['species'][data_index], sme_linelist['line_extra'][data_index], \
        sme_linelist['line_term_low'][data_index], sme_linelist['line_term_upp'][data_index], \
        sme_linelist['line_lulande'][data_index]

    return line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande


# Creates the segment mask and the resolution of it from the csv file that we name in the variable input.
def segment_mask_creation(segment_mask):
    # What is status? At one point during make_obs, status it says if status = 0 it's an error.
    # It gets set to 1 only at 1 point after selecting wavelength regions inside segments, before
    # starting the pre normalisation procedure. Unused in our code. @@@@@@@@@@
    # status = 0

    # Segm_mask is _Segm.data, unsurprisingly. It takes the start, end wavelength, and the resolution base guess of 3.5k
    # which is totally wrong. That's something to fix/modify when we're streamlining this.
    # The seperator separates via 2 spaces or a space and ; or . and space. We use the overflow column to account for
    # lines that begin with ; (commented out) which we immediately delete afterwards. Hard to separate comments when ;
    # was forgotten. Can't do lower case potentials because it randomly selects lower case letters in the middle of
    # sentences. Something is wrong there.
    # Same as in galah part 4, this should be a dummy file with only one segment, but we're trying to avoid that so we
    # are using the full regular segment mask NOT obsname.
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
    # But it's already ordered? weird. Will leave it in case data comes in unordered
    segment_mask_data_with_res.sort_values(by=['Wavelength_Start', 'Wavelength_End'])

    return segment_mask_data_with_res


# Creates an interpolation equation based on the resolutions we already have and their wavelgnths. This is done earlier
# ion galah sp part 2! @@@@@@@@@@@@
def resolution_interpolation(object_pivot):
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


# Adjusts the wavelengths according to the doppler shift and returns it.
def doppler_wavelengths(total_ccd_wavelength, radial_velocity_global):
    # Setting speed of light
    c = 2.99792458E8

    # Shifts the wavelengths to account for the doppler effect.
    total_ccd_wavelength_dopplered = total_ccd_wavelength / ((radial_velocity_global[0] * (1E3 / c)) + 1E0)

    return total_ccd_wavelength_dopplered


# Finds the wavelengths that we have that are also inside the segments.
def wavelengths_flux_inside_segments(segment_mask_data_with_res, total_ccd_wavelength_dopplered, total_ccd_flux,
                                     total_ccd_flux_error):
    # Skipping the "if not normalising, make sure there are lines in the segment" part as I think we can do that here
    # Unsure how many there will be, so arrays can't be used. Instead we use a temp list.
    ccd_wavelengths_inside_segmask = []
    ccd_flux_inside_segmask = []
    ccd_flux_error_inside_segmask = []
    ccd_resolution_of_segment = []
    # array for the first and final wavelength indexes of each segment respectively.
    # Remember SME only wants the one col I blieve. But PYSME might use both?
    wavelength_start_end_index = np.zeros((len(segment_mask_data_with_res["Wavelength_Start"]), 2))
    print("BIG OL LEN", (total_ccd_wavelength_dopplered))

    # For each segment in segmask, find the values of dopplered wavelength (and associated flux from indexing)
    # that are inside.
    # This is the ;select wavelength regions inside segments      part of makestruct
    # Despite having this array we still use np where each time to find the wavelengths in the segments
    for segment in range(0, len(segment_mask_data_with_res["Wavelength_Start"])):
        # Beginning wavelength and end of that segment. Put as variables here for readability.
        seg_start = (pd.to_numeric(segment_mask_data_with_res["Wavelength_Start"][segment]))
        seg_stop = (pd.to_numeric(segment_mask_data_with_res["Wavelength_End"][segment]))

        # Finding the index of values inside the segment, using logical and is a neccesity.
        wavelength_inside_segmask_index = np.where(
            np.logical_and(seg_stop >= total_ccd_wavelength_dopplered, total_ccd_wavelength_dopplered >= seg_start))

        ccd_wavelengths_inside_segmask.extend(total_ccd_wavelength_dopplered[wavelength_inside_segmask_index])
        ccd_flux_inside_segmask.extend(total_ccd_flux[wavelength_inside_segmask_index])
        ccd_flux_error_inside_segmask.extend(total_ccd_flux_error[wavelength_inside_segmask_index])

        # Numpy array of indexes of the first and final wavelengths per segment with column 0 being the first.
        # the wind SME wants seems to just be the final values so keep a note of that @@@@@@@@@@@@@@@
        # If there's some discrepency between the sme save spectra and the DataReleaseSp we get past a value error
        # of an empty array
        if wavelength_inside_segmask_index[0].size != 0:
            wavelength_start_end_index[segment, 0] = (wavelength_inside_segmask_index[0][0])
            wavelength_start_end_index[segment, 1] = (wavelength_inside_segmask_index[-1][-1])
        ccd_resolution_of_segment.append(segment_mask_data_with_res['Resolution'][segment])

    # Turning lists into arrays for numpy indexing with np.where.
    ccd_wavelengths_inside_segmask_array = np.array(ccd_wavelengths_inside_segmask)
    ccd_flux_inside_segmask_array = np.array(ccd_flux_inside_segmask)
    ccd_flux_error_inside_segmask_array = np.array(ccd_flux_error_inside_segmask)
    ccd_resolution_of_segment_array = np.array(ccd_resolution_of_segment)
    print("\n\n\nBIG OL LEN OF WAVLENGTH AT MIDDLEEEE", ccd_wavelengths_inside_segmask_array, "\n\n\n")

    return ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
           ccd_resolution_of_segment_array, wavelength_start_end_index


# Finds the first and final wavelengths of the segments that we have.
def find_segment_limits(wavelength_start_end_index, total_ccd_wavelength_dopplered):
    number_of_segments = len(wavelength_start_end_index)

    # wran is segment_begin_end
    # An array with two columns, the first and last recorded wavelength in each segment
    # Why do we have both the index AN D the values in two separate arrays? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
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




# make_obs in idl, sets up the important wavelength, flux, and error of the important segments of wavelengths we have.
# This is the first one to run I think from create struct.
def object_array_setup(makestruct_dict):
    # We use the full setup file instead of the dummy/temp obsname file as we're trying to avoid constand looping.
    # We set this earlier.
    segment_mask = str(makestruct_dict['setup_for_obs']) + '_Segm.dat'
    segment_mask_data_with_res = segment_mask_creation(segment_mask)

    # Takes the last three digits and turns into an int. The last 3 represent the fibre of the ccd used that we want to
    # look at. Then we produce an interpolation equation for the resolutions.
    interpolation = resolution_interpolation(int(str(makestruct_dict['object_for_obs'])[-3:]))

    # Uses our resolution interpolatino to find the resolution at the peak wavelengths that we also find.
    segment_mask_data_with_res = interpolate_peak_resolution(makestruct_dict, segment_mask_data_with_res, interpolation)

    # Checks for a negative range which I assume would cause issues. Check it works @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if min(segment_mask_data_with_res['Wavelength_End'] - pd.to_numeric(
            segment_mask_data_with_res['Wavelength_Start'])) <= 0:
        print("Segment has a negative range!")
        return

    # Checks for overlapping segments if there's more than one. Double check this works when you have data @@@@@@@@@@@@
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
    print("\n\n\n\ntotal_ccd_wavelength1 b4 wavelengths flux inside seg", len(total_ccd_wavelength_dopplered))

    ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
        ccd_resolution_of_segment_array, wavelength_start_end_index = \
        wavelengths_flux_inside_segments(segment_mask_data_with_res, total_ccd_wavelength_dopplered, total_ccd_flux,
                                         total_ccd_flux_error)

    print("\n\n\n\ntotal_ccd_wavelength2 after wavelengths flux inside seg", len(ccd_wavelengths_inside_segmask_array))

    # Number of segments that contain visible spectra. Wind (wavelength_start_end_index)
    # is the array of indexes of first and final wavelengths per segment.
    number_of_segments = len(wavelength_start_end_index)
    if number_of_segments == 0:
        print("No observations in segment mask")
        return

    # Creates an array with the beginning and end wavelenghts of the segments.
    # different to the start end array as that's indexes.
    segment_begin_end = find_segment_limits(wavelength_start_end_index, total_ccd_wavelength_dopplered)

    # No clue. I think just confirms it reached the end of the function? Seems ridiculous. Removed. @@
    # status = 1

    # Returning the variables used for prenormalisation and the rest of makestruct.
    # The first three are the important data.
    # Segment begin end is just made from the data (beginning and end wavelengths), but best not remake it each time.
    return ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array, \
           segment_begin_end, wavelength_start_end_index, ccd_resolution_of_segment_array


# Open the file containing wave/sob/uob that we iteratively update each loop of makestruct/sp4 run.
def load_spectra(makestruct_dict):
    # Change it if load file is set to the one SME outputs.
    if makestruct_dict['load_file']:
        spectra_data = \
            pickle.load(open("OUTPUT/SPECTRA/" + makestruct_dict['obs_name'] + "_SME_spectra.pkl", "rb"))

    # This is for before SME runs we open the one created in sp2.
    else:
        spectra_data = \
            pickle.load(open("OUTPUT/SPECTRA/" + makestruct_dict['obs_name'] + "_unnormalised_spectra.pkl", "rb"))
    print("Immediately after loading spectra:", spectra_data['wave'])
    # If the first value in the wavelength is an array with all wavelengths in the first segment.
    # If it's not an array it's not segmented, the first value is just a float and we don't need to desegmentise.
    # mask exists in the dict, but not used in calculations.
    if isinstance(spectra_data['wave'][0], np.ndarray):
        wave, flux, error = desegmentise(spectra_data)
        return wave, flux, error

    return spectra_data['wave'], spectra_data['flux'], spectra_data['error']


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
# removes data that is too far away from the point and uses the close data to normalised itself.
# Doesn't affect the non-inlier data which is confusing and weird as it is not normalised for them.?
def pre_normalise(ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array,
                  ccd_flux_error_inside_segmask_array, segment_begin_end):
    # Number of segments we have. Made each function to avoid potential length errors.
    number_of_segments = len(segment_begin_end)
    # Pre-normalization steps. ";Performs first-guess normalisation by robustly converging straight line fit to high
    # pixels"
    # I removed the requirement for sob > 0. @@@@@@@@@@
    for segment_band in range(0, number_of_segments):

        # Finds the index where the wavelengths are between the start and end of each segment, to be able to loop each
        # seg as i.
        # We do this in wavelength_inside_segmask_index already, but we know that the IDL code is so damn convoluted
        # then maybe a case occurs where we didn't make that. Doesn't make any sense to me, but is possible.
        # If this is slow, we can optimise somehow probably. @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        segment_indexes = (np.where(np.logical_and(ccd_wavelengths_inside_segmask_array >=
                                                   segment_begin_end[segment_band, 0],
                                                   ccd_wavelengths_inside_segmask_array <=
                                                   segment_begin_end[segment_band, 1])))[0]

        #print("Beginning and end of segment band", segment_begin_end[segment_band])
        # If count is greater than 20, do this. len segindex is the number of values that fit our criteria.
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

            # print(wavelength_array, "\n",ccd_flux_inside_segmask_array[segment_indexes])
            # print(ccd_flux_inside_segmask_array[segment_indexes])
            # import matplotlib.pyplot as plt
            # plot(ccd_wavelengths_inside_segmask_array[segment_indexes], \
            # ccd_flux_inside_segmask_array[segment_indexes])
            # plt.show()

        # If we don't have enough points, we just use the mean value instead. Numpy mean wasn#t working. Still need to
        # Double check this is correct, as it was wrong for seemingly no reason before @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        else:

            # np.mean had issues with our results. Making a variable for readability.
            flux_mean = (sum(ccd_wavelengths_inside_segmask_array[segment_indexes])) / len(
                ccd_wavelengths_inside_segmask_array[segment_indexes])

            # Gotta be ordered correctly or we modify the sob before we use it to modify uob!
            # Is this the right way around? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            ccd_flux_error_inside_segmask_array[segment_indexes] = \
                ccd_flux_error_inside_segmask_array[segment_indexes] / flux_mean

            ccd_flux_inside_segmask_array[segment_indexes] = \
                (ccd_flux_inside_segmask_array[segment_indexes]) / flux_mean
            # import matplotlib.pyplot as plt
            # plt.plot(ccd_wavelengths_inside_segmask_array[segment_indexes], \
            # ccd_flux_inside_segmask_array[segment_indexes])
            # plt.show()
    # end of prenormalisation! Phew! Again we return the modified observational data

    return ccd_wavelengths_inside_segmask_array, ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array


# Open up the line and conituum masks for making mob/flagged fluxes. Linemask is Sp.
def open_masks(makestruct_dict):
    # Reads out the columns which are centre/peak wavelength, start and end of wavelength peak (all simulated),
    # and atomic number. Seperated by 2 spaces (sep), no headers as it is only data, and the names are what we
    # Assign them as to be used later with linemask_data["Sim_wave" etc.]
    # Careful with the python enginer, it's slower. If we are looking at BIG data files this might be bad.
    # line0, line_st, and line_en in idl. they ignore atomic number for some reason @@@@@@@@@@@@@@@@@@@
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

    continuum_mask = str(makestruct_dict['setup_for_obs']) + "_Cont.dat"
    continuum_mask_data = pd.read_csv("GALAH/DATA/" + continuum_mask, delim_whitespace=True, header=None,
                                      engine='python', names=["Continuum_Start", "Continuum_End"])

    return continuum_mask_data, linemask_data


# Flag the fluxes that correspond to the peaks. All values inside Sp are flagged because of reasons that karin said.
def flag_flux_peaks(makestruct_dict, ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                    ccd_flux_norm_error_inside_segmask_array):
    # Opens the data files containing the line and continuum data. we use these limits to flag the continuum
    # and peak fluxes. We take only the 2nd variable as this is the linemask peak function, not the continuum. (SP)
    linemask_data = open_masks(makestruct_dict)[1]

    """This part is confusing and I HOPE I did it okay, must test. @@@@@@@@@@@@"""
    # mob in idl. Mask of observed pixels
    flagged_fluxes = np.zeros(len(ccd_wavelengths_inside_segmask_array))

    # We flag the fluxes that are probably peaks inside our segments that we care about that are probably atomic absorb.
    for line_loop in range(0, len(linemask_data['Sim_Wavelength_Start'])):

        # Usual case of making sure the wavelengths we want to use are in the lines. (They're already made sure to be in
        # the segement mask. What's the diff? It's to separate by segment for each loop.
        wavelengths_inside_linemask_index = np.where(np.logical_and(
            ccd_wavelengths_inside_segmask_array >= linemask_data['Sim_Wavelength_Start'][line_loop],
            ccd_wavelengths_inside_segmask_array <= linemask_data['Sim_Wavelength_End'][line_loop]))

        # running_snr in idl, sets values to 1 if they're below the max noise spike. This means we flag all good points
        # at 1 that are below spikes in noise to be used.
        signal_to_noise = []
        # We're trying to find signal to noise ratios I guess, where 1.5 is the limit for our noise?
        # Averages over +/-4 values. Unsure about the 11 though
        for flux_row in range(0, len(ccd_flux_norm_inside_segmask_array)):  # Should this not be uob? for 2nd len()
            #              Indexes the obs flux from ii-4 to ii + 4 (or limits of length)
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

            # If the flux exists and is less than the noise, set a marker to 1 to indicate this.
            # Is this working? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if min(ccd_flux_norm_inside_segmask_array[wavelengths_inside_linemask_index[0]]) > 0 and \
                    max(ccd_flux_norm_inside_segmask_array[wavelengths_inside_linemask_index[0]]) < \
                    max(signal_to_noise[wavelengths_inside_linemask_index[0]]):
                # 1 is a good thing! it means it fits nicely in the peak and the noise is nothing to worry about
                flagged_fluxes[wavelengths_inside_linemask_index[0]] = 1
    # Return our array of which wavelengths are flagged as possible peaks to be modified further in contiuum and more.
    return flagged_fluxes


# And the same as flagging but for continuum
def flag_continuum(makestruct_dict, ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                   ccd_flux_norm_error_inside_segmask_array, flagged_fluxes):
    """; cont mob  - continuum points selected between 0 and 1.2 or 1+3*sigma, where there are no line masks
    ;             avoid buffer zone at edges
    """
    # Opens the data files containing the line and continuum data. we use these limits to flag the continuum
    # and peak fluxes. We take only the 1st variable as this is the continuum.
    continuum_mask_data = open_masks(makestruct_dict)[0]

    # Throws     (flagged_fluxes[cutting_flux_index] == 2)))
    # TypeError: 'NoneType' object is not subscriptable if flagged fluxes are not returned
    if makestruct_dict['continuum_scale_flag'] == 'linear':

        # For each segment in the continuum file. Currently only one massive range so meh.
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
    # Number of segments we have. Made each function to avoid potential length errors.
    number_of_segments = len(segment_begin_end)
    for segment_band in range(0, number_of_segments):
        ";Deselect 70% lowest continuum points in each segment using synthesis. Ensure both ends have continuum points"
        # The fraction of points we want to remove. Potentially modified in the while loop hence kept outside of it.
        fraction = 0.7
        # We take the wavelength indexes of the wavelengths that exist in the segments as always. We just repeat it
        # each time to be able to work on each segment individually I suppose? @@@@@@@@@
        wavelength_inside_segment_index = np.where(
            np.logical_and(ccd_wavelengths_inside_segmask_array >= segment_begin_end[segment_band, 0],
                           segment_begin_end[segment_band, 1] >= ccd_wavelengths_inside_segmask_array))

        # While the fraction to use is not 0 we continue looping. Python is setting it to E-17 in loop so this fixes it
        # by using 0.01 instead of 0.
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
            # Here we see how many are in the first 1/3, and how many in the last 2/3.
            # Also have to be flagged as 2 - so not peaks.
            saved_continuum_index = np.where(np.logical_and(
                ccd_flux_norm_inside_segmask_array >= cutting_flux_value,
                np.logical_and(ccd_wavelengths_inside_segmask_array >= segment_begin_end[segment_band, 0],
                               segment_begin_end[segment_band, 1] >= ccd_wavelengths_inside_segmask_array)))

            # again the [0] to tak the values and ignore dtype
            # How many values are in the top and bottom 1/3,
            # we use indexes here as the wavelengths are ordered so it works
            # out easily as [133] > [132] for example.
            low_continuum_index = np.where(
                saved_continuum_index[0] <=
                wavelength_inside_segment_index[0][int(len(wavelength_inside_segment_index[0]) / 3)])

            high_continuum_index = np.where(
                saved_continuum_index[0] >=
                wavelength_inside_segment_index[0][int(len(wavelength_inside_segment_index[0]) * 2 / 3)])

            # If we don't have enough points, decrease the fraction we remove.
            if len(low_continuum_index[0]) < 5 or len(high_continuum_index[0]) < 5:
                fraction -= 0.1
                # print("Adjusting fraction down 10% to", fraction)
                # if fraction < 0.1:

                # print(len(wavelength_inside_segment_index[0]))
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

    """import matplotlib.pyplot as plt

    plt.plot(ccd_wavelengths_inside_segmask_array,
             flagged_fluxes)
    plt.plot(ccd_wavelengths_inside_segmask_array,
             ccd_flux_norm_inside_segmask_array)

    plt.show()"""
    # Signal to break the while loop and continue iteration. Unused?
    # fraction = 0

    """this stuff might work or it might not?! CHECK IT SOMEHOW! @@@@@@@@@@@@@@@@@@@@@@@@@@@@^^^^^^^^^^^^"""

    return flagged_fluxes


# Honestly unsure what this does. soemthing to do with cores and "avoiding strong NLTE-dominated cores in mask" @@@@
def removing_nlte(makestruct_dict, ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                  flagged_fluxes):
    # ;Avoid strong NLTE-dominated cores in mask, what. @@@@@@@@@@@@@
    core_minimum = 0.6
    if makestruct_dict['metalicity'] < -2:
        core_minimum = 0.72
    elif makestruct_dict['metalicity'] < -1:
        core_minimum = 0.65

    # Checks for line_cores existing from galahsp1, if no variable exists then we can't do this can we.
    try:
        print("Using the line cores:", makestruct_dict['line_cores'])
        # Not sure what line cores are, but basically if it's near the value of the wavelength,
        # and the flux is low we're setting it to 0.
        for line_core_loop in range(0, len(makestruct_dict['line_cores'])):
            line_core_wavelength_index = np.where(
                np.logical_and(
                    abs(ccd_wavelengths_inside_segmask_array - makestruct_dict['line_cores'][line_core_loop]) < 4,
                    ccd_flux_norm_inside_segmask_array < core_minimum))
            flagged_fluxes[line_core_wavelength_index] = 0
    except AttributeError:
        print("No line_cores")
    return flagged_fluxes


# Sums the flagged flux creation functions in one to produce a final product. make_mob in idl
def create_observational_definitions(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                     ccd_flux_norm_inside_segmask_array, ccd_flux_norm_error_inside_segmask_array,
                                     segment_begin_end):
    # Open masks is called in flag functions so no need to do it here for input.
    # We irst flag ones that appear to be peaks, then the continuum, then remov those with too low a flux to be used ,
    # inputting the previously created array each time.
    flagged_fluxes = flag_flux_peaks(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                     ccd_flux_norm_inside_segmask_array, ccd_flux_norm_error_inside_segmask_array)

    flagged_fluxes = flag_continuum(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                    ccd_flux_norm_inside_segmask_array, ccd_flux_norm_error_inside_segmask_array,
                                    flagged_fluxes)

    flagged_fluxes = cutting_low_flux(ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array,
                                      segment_begin_end, flagged_fluxes)

    flagged_fluxes = removing_nlte(makestruct_dict, ccd_wavelengths_inside_segmask_array,
                                   ccd_flux_norm_inside_segmask_array, flagged_fluxes)

    """import matplotlib.pyplot as plt
    plt.plot(ccd_wavelengths_inside_segmask_array, ccd_flux_norm_inside_segmask_array)
    plt.plot(ccd_wavelengths_inside_segmask_array, flagged_fluxes)
    plt.show(block=True)
    """
    return flagged_fluxes


# Compare atomic_line array to segments we have to see which fit in. But we already indexed atomic array against
# segments in galahpart4 so this might be useless@@@@
# Search for de= in makestruct to find its equivalent

def atomic_lines_in_segments(makestruct_dict, desired_atomic_lines_index, segment_begin_end, line_atomic, depth):
    number_of_segments = len(segment_begin_end)
    # The list to add the indexes that are in the segments and linemask. Overall it seems to inclujde almost everything.
    # currently we take every single value out of the 24k - I guess because we already detailed the important values
    # in rdlin when making the index to create line_atomic. @@@@@@@@@@@@@@@@@@@@
    # Selecting lines within wavelength segments (We've done that SO MUCH!!) and deep than a given depth
    # this is de in idl, I think it's just a buffer?
    buffer = 0.7
    # Is this even needed? We index lineatomic in smerdlin using the index of important wavelengths in galahpart4 @@
    # So we already know they're in the segments. @@@@
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
                # the indexes of the wavelengths inside our segments. guess it's more useful for if we run through
                # a single wavelength segment like in idl.
                desired_atomic_lines_index.extend(np.where(line_atomic[:, 2] == broad_line_single)[0])

    return desired_atomic_lines_index


# Now we see which are in the line list, which is the list of important wavelengths to look out for.
# This is if we aren't doing atomic+lnes)in)segments
def atomic_lines_in_linemask(makestruct_dict, desired_atomic_lines_index, line_atomic, depth, linemask_data):
    buffer = 0.7

    # Used to take the certain number of linemask indexes (roughly 20ish ) what does it stand for? @@@@@@@@@@@@@@@@@
    nrline = 20 + ((8000 - makestruct_dict['effective_temperature']) / 1.3E3) ** 4

    # We see which parts of our atomic line data lies within the linelist (the important wavelengths) as well as
    # identifying the peak wavelength if we can, and the broad lines. We get a lot (12kish) duplicates here
    # I don't see why we do this, we've already indexed to only include wavelengths in SegM in galah part 4, so when
    # woudl we have data that is out of that and in the linemask? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    for line in range(0, len(linemask_data['Sim_Wavelength_Peak'])):

        # Ok.. again we find the wavelengths within the ranges, but this time it'sthe linemask ranges, not segments.
        inside_linemask_index = np.where(
            np.logical_and(
                np.logical_and(
                    line_atomic[:, 2] > linemask_data['Sim_Wavelength_Start'][line] - buffer,
                    line_atomic[:, 2] < linemask_data['Sim_Wavelength_End'][line] + buffer),
                depth > makestruct_dict['depthmin']))

        # We reverse it to take the the largest indexes? But that doesn't necessarily mean the strongest lines
        # does it? @@@@@@@@@@@@@@@@@@@@
        inside_linemask_index = np.flip(inside_linemask_index)

        # Take either the last nrline number or all the index? How odd, why is nrline a thing@@@@@@@@@@@@@@@@@@@@@@@@@@
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

        # And of course, always select broad lines when close just like before (but that was broad lines in segments?)
        # What's the diff betwee nthe broad lines? And why ARE we using the line list now and adding it to the same
        # list as the segments information? @@@@@@@@@@@@@@@@@@@@@@@@@@
        for broad_line_single in makestruct_dict['broad_lines']:
            # If any line is within 100a of a broadline we'll include it
            if np.any(abs(broad_line_single - float(linemask_data['Sim_Wavelength_Peak'][line])) < 100):
                # Where does this broad line exist in our atomic line array?
                # rounds automatically in np where for the array.
                # We have duplicates in line_atiomic and (therefore?) the d_a_l_index, do we want to remove those?
                # We add all(?) the lineatomic to the index, but that makes sense as it is preselected to include
                # the indexes of the wavelengths inside our segments. guess it's more useful for if we run through
                # a single wavelength segment like in idl.
                desired_atomic_lines_index.extend(np.where(line_atomic[:, 2] == broad_line_single)[0])

    # but why are we adding linemask AND segment mask lines into the same array and do they ever combine
    return desired_atomic_lines_index


# I think this is make_line in idl. We produce an indexed version of the data from indexed_stellar_information that
# takes only the atomic lines that are either in the segments (but they are already?? @@) or in the linemask, depending
# on what we send the variable 'run' to.
def produce_indexed_atomic_data(makestruct_dict, segment_begin_end, linemask_data, run="Synthesise"):
    """ Here's where they run sme rdlin,
    but we do that at the beginning and only if we don't have the premade data file"""
    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        indexed_stellar_information(makestruct_dict)

    # If run is 2 we want the indexes of wavelengths in segments, else we want them from the linelist ranges.
    desired_atomic_lines_index = []
    if run == "Synthesise":
        desired_atomic_lines_index = atomic_lines_in_segments(makestruct_dict, desired_atomic_lines_index,
                                                              segment_begin_end, line_atomic, depth)
    else:
        desired_atomic_lines_index = atomic_lines_in_linemask(makestruct_dict, desired_atomic_lines_index,
                                                              line_atomic, depth, linemask_data)
        print(run, len(desired_atomic_lines_index))

    if len(desired_atomic_lines_index) == 0:
        print("No atomic lines within chosen Segments. Likely during a Balmer line run, moving on.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    # Is now an array as we're doing adding to it, and we take only the unique values using Unique as there's a lot of
    # overlap between the segment mask and linemask
    desired_atomic_lines_index = np.unique(desired_atomic_lines_index)

    nselect = len(desired_atomic_lines_index)

    # Showing how many lines were not used. But might be ambiguous, as these are the duplicates,
    # not the ones out of bounds.
    # The out of bounds ones were discared 
    print(nselect, "unique spectral lines are selected within wavelength segments out of", len(line_atomic))

    line_atomic = line_atomic[desired_atomic_lines_index]
    # Now we also sort them according to wavelength. Not sure how needed this is. @@@@@@@@@@@
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


# Strings to tell PYSME what kind of atmosphere we're looking for.
def set_atmosphere_string():
    atmosphere_depth = "RHOX"
    atmosphere_interpolation = "TAU"
    # Plane parallel
    atmoshpere_geometry = "PP"

    return atmosphere_depth, atmosphere_interpolation, atmoshpere_geometry


# IDL saves it as an inp file, but we don't like that. This is a copy of what IDL does and where it saves it,
#  Called from make_struct function. We input it directly by importing sme,
#  but we save it out in case we want it I guess.
def store_sme_input(sme_input, makestruct_dict):
    print("Saving the SME input to 'OUTPUT/VARIABLES/",
          makestruct_dict['obs_name'], "Original_SME_input_variables.pkl'")
    input_file = open(r'OUTPUT/VARIABLES/' + makestruct_dict['obs_name'] + '_SME_input_variables.pkl', 'wb')

    pickle.dump(sme_input, input_file)
    input_file.close()


# A function to take the wavelength etc arrays that are just a single array, and turn it into a list of arrays seperated
# per segment, as required from pysme.
def segmentise_arrays(wavelengths, flux, flux_error, flagged_flux, total_segments):
    # Create lists to store the arrays of eachg type in.
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
