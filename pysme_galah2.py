"""read in spectra and perform skyline and tellurics correction. "Reads in 4 band spectra (ext. 0), so for 4 seperate
files, each representing a different ccd from Galah."""

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
import pickle


# We open the four ccd files as we do in the makestruct_createdatfiles to retrieve wavelength, flux, and error.
def get_wavelength_flux(object_for_obs=150405000901378):
    """
    Input:
        Object number: For indexing files
    Output:
        total_ccd_wavelength: Wavelengths given by ccd files for given object
        total_ccd_sob_flux: Fluxes given by ccd files for given object
        total_ccd_relative_flux_error: Error in fluxes given by ccd files for given object
    """
    # Temporary lists to append to while in a loop to open ccd files.
    temp_ccd_wave = []
    temp_ccd_flux = []
    temp_ccd_error = []
    # For each of the four ccd files, 1-4, we open and take wavelength, flux, and error
    for ccd_number in range(1, 5):
        # Opens fit file in given folder in GALAH
        resolution_file = fits.open(r"GALAH/SPECTRA/test/" + str(object_for_obs) + str(ccd_number) + str(".fits"))
        # Creates a list of the wavelengths starting from crval1 and using the number of steps naxis with the change
        # in wavelength per step cdelt1
        temp_ccd_wave.extend(resolution_file[0].header['CRVAL1'] +
                             (resolution_file[0].header['CDELT1'] * np.arange(
                                resolution_file[0].header['NAXIS1'])))

        # the .data in the file is just the flux and error, no equations needed.
        temp_ccd_flux.extend(resolution_file[0].data)
        temp_ccd_error.extend(resolution_file[1].data)

    total_ccd_wavelength, total_ccd_sob_flux, total_ccd_relative_flux_error = \
        np.asarray(temp_ccd_wave), np.asarray(temp_ccd_flux), np.asarray(temp_ccd_error)

    return total_ccd_wavelength, total_ccd_sob_flux, total_ccd_relative_flux_error


# Interpolation to produce resolutions for different wavelengths we produce during the run (e.g from doppler shift)
def get_resolution(object_for_obs=150405000901378):
    """
    Input:
        Object number to find the correct files and data.
    Output:
        interpolation: Interpolation equation to find resolution at other wavelengths
        resolution_factor: Whether we have a high or low resolution. It's applied to the later resolutions."""


    # All ccd files have the same resolution information, so we simply pick the first one.
    resolution_file = fits.open(r"GALAH/SPECTRA/test/" + str(object_for_obs) + str("1.fits"))

    # We decide whether we have a high or low resolution based on the information given in the resolution file.
    if resolution_file[0].header['SLITMASK'] == 'IN      ':  # High res
        resolution_factor = 1.789
    else:  # low
        resolution_factor = 1.0

    # The final 3 numbers in the object reprsent its pivot value, where the pivots are the resolution files.
    object_pivot = int(str(object_for_obs)[-3:])
    # Temporary lists for the wavelengths and y value of the res files
    temp_ccd_piv_y = []
    temp_ccd_wave_res = []

    # Grabbing the resolution from the ccd files and concaneating them into two large arrays to interpoltae and allow
    # creation of resolution at other wavelengths (as our wavelength is modified later)
    for ccd_number in range(1, 5):
        # Opens the large ccd files with all objects inside it.
        ccd_res_file = fits.open(r'GALAH/DATA/ccd{0}_piv.fits'.format(ccd_number), ext=0)
        # Grabs the data from the fits format.
        ccd_res_data = ccd_res_file[0].data

        # We're making an array of the resolution.
        # Extracting the row of data of ccd1 that matches the piv number (-1 as piv starts at 1) for our object
        temp_ccd_piv_y.extend(ccd_res_data[object_pivot - 1])

        # Creates a wavelength list from starting CRVAL1 (4700) in steps of CRDELT1 (0.1)
        # until it matches the array len of NAXIS1
        resolution_wavelengths = \
            ccd_res_file[0].header['CRVAL1'] + \
            (ccd_res_file[0].header['CDELT1'] * np.arange(ccd_res_file[0].header['NAXIS1']))

        # Extends the list to combine each CCD until we have the final result and can np.array it.
        temp_ccd_wave_res.extend(resolution_wavelengths)

    wavelength_res_x_collection, wavelength_res_y_collection = \
        np.asarray(temp_ccd_wave_res), np.asarray(temp_ccd_piv_y)

    interpolation = interp1d(wavelength_res_x_collection, wavelength_res_y_collection)

    return interpolation, resolution_factor


" Read in reduction pipeline output. This is where we get all our data such as vmic and vmac. Mainly used in galahsp3"
def data_release_index(object_for_obs=150405000901378):
    """
    Input:
        Object number: to find the correct files and data.
    Output:
        reduction_and_analysis_data: Dictionary containing variables for our star only, rather than the large fits file.
    """

    # A very large fil containing information on all stars in its data release. Takes a long time to open.
    # We open it to obtain variables such as macroturbulent velocity in our star.
    reduction_and_analysis = fits.open(r'GALAH/DATA/sobject_iraf_53'
                                       r'_2MASS_GaiaDR2_WISE_PanSTARRSDR1_BailerJones_K2seis_small.fits')

    # Finds the index of where we can find our object id.
    reduction_and_analysis_index = np.where(reduction_and_analysis[1].data['sobject_id'] == object_for_obs)
    # Checks the index exists therefore the object id exists in the place we're looking for.
    if not reduction_and_analysis_index[0].size:
        print("Object does not exist in chosen DR folder.")
        exit()

    # Applies the index to take the data only.
    reduction_and_analysis_data = reduction_and_analysis[1].data[reduction_and_analysis_index]

    return reduction_and_analysis_data


"Telluric correction (Incr errors), removes the wavelengths that the earths atmosphere produces"


# We correct for the telluric values to get a more accurate wavelength array, returning it as a clean dictionary
def error_correction(velocity_barycenter, total_ccd_wavelength, total_ccd_sob_flux,
                     total_ccd_relative_flux_error):
    """
    Input:
        velocity_barycenter: Barycentric velocity to correct for telluric values
        total_ccd_wavelength: Wavelengths of spectra
        total_ccd_sob_flux: Flux of spectra
        total_ccd_relative_flux_error: Error of flux of spectra
    Output:
        ccd_data_dict: Dictionary containing the corrected wavelength, flux, and flux error.
    """

    # We're interpolating the telluric data on to the grid dimensions of the spectra we read.
    telluric = fits.open(r'GALAH/DATA/telluric_noao_21k.fits')
    # Taking the wavelengths from the telluric fits file
    telluric_wavelengths = telluric[1].data['wave'] / (1 - (velocity_barycenter / 299792.458))
    # Made as a python variable so we can append a new flux to it if out of bounds.
    telluric_flux = telluric[1].data['flux']
    # Extends the edges of telluric values if the current range isn't enough to fit the wavelength
    # These are not completly accurate but are the best we do at the moment.
    if max(telluric_wavelengths) < max(total_ccd_wavelength):
        # Adds a new max wavelength to telluric at the end, and sets the flux of it to 1.
        telluric_wavelengths = np.append(telluric_wavelengths, max(total_ccd_wavelength))
        telluric_flux = np.append(telluric_flux, 1)
        print("Warning, telluric range not long enough. Adding Flux at end.")
    if min(telluric_wavelengths) > min(total_ccd_wavelength):
        # Adds a minimum wavelength at the beginning of the array, and sets flux to 1.
        telluric_wavelengths = np.insert(telluric_wavelengths, 0, min(total_ccd_wavelength))
        telluric_flux = np.insert(telluric_flux, 0, 1)
        print("Warning, telluric range not long enough. Adding Flux at beginning.")

    # Finally make the interpolation function coefficient to apply to the telluric fix to our wavelengths.
    telluric_interpolation = interp1d(telluric_wavelengths, telluric_flux)
    telluric_interpolated_ccd_wavelength_array = telluric_interpolation(total_ccd_wavelength)
    # We find the results out of a rough continuum area and change them to be more in line with the continuum.
    telluric_below_zero = np.where(telluric_interpolated_ccd_wavelength_array < 0.81)
    telluric_above_one = np.where(telluric_interpolated_ccd_wavelength_array > 0.998)
    # Prevents the equation below dividing by 0.
    telluric_interpolated_ccd_wavelength_array[telluric_below_zero] = 0.81
    telluric_interpolated_ccd_wavelength_array[telluric_above_one] = 1.0
    # Increasing the error due to these corrections.
    total_ccd_relative_flux_error = total_ccd_relative_flux_error / (telluric_interpolated_ccd_wavelength_array * 5 - 4)

    "Skyline correction (increasing errors)"
    # Again we need to have a case for if the wavelengths we input are out of the range of the sky mask
    sky_mask = fits.open(r'GALAH/DATA/Skyspectrum_161105.fits')
    # Adjustin for sky mask, similar to telluric information
    sky_mask_wavelengths = ((sky_mask[1].data['wave']) / (1 - (velocity_barycenter / 299792.458)))
    sky_flux = sky_mask[1].data['sky']

    # If there are more wavelengths in our ccd that go beyond the sky mask boundaries, we add a new min or max
    # and set its flux to 1.
    if max(sky_mask_wavelengths) < max(total_ccd_wavelength):
        # Adds a new max wavelength to sky at the end, and sets the flux of it to 1.
        sky_mask_wavelengths = np.append(sky_mask_wavelengths, max(total_ccd_wavelength))
        sky_flux = np.append(sky_flux, 1)

    if min(sky_mask_wavelengths) > min(total_ccd_wavelength):
        # Adds a minimum wavelength at the beginning of the array, and sets flux to 1.
        sky_mask_wavelengths = np.insert(sky_mask_wavelengths, 0, min(total_ccd_wavelength))
        sky_flux = np.insert(sky_flux, 0, 1)

    # A repeat of telluric interpolation.
    sky_mask_interpolation = interp1d(sky_mask_wavelengths, sky_flux)
    sky_mask_interpolated = sky_mask_interpolation(total_ccd_wavelength)
    total_ccd_relative_flux_error = total_ccd_relative_flux_error + sky_mask_interpolated

    # Relative error to actual error.
    total_ccd_flux_error_uob = total_ccd_relative_flux_error * total_ccd_sob_flux
    # Collects final results in a clean dictionary.
    ccd_data_dict = {"wave": total_ccd_wavelength, "flux": total_ccd_sob_flux, "error": total_ccd_flux_error_uob}

    return ccd_data_dict

