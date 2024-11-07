from astropy.io import fits
import numpy as np
import pysme_update_logg

"""Taking the cannon file and finding the object id desired inside it, then taking those parameters and 
modifying them based on cannon guess - if it was wrong we adjust it to a baseline - and output the parameters
This file currently gives the parameters such as Effective Temp. Surface Grav. metallicity. Micro Turblnce. 
Rot Vel. Radial Velocity. Macro Turb"""


# Uses either CANNON or GUESS to produce starting parameters for our star which we iterate on to make more accurate.
# Inside those functions, they call on the data provided to them and modify it to account for physics not included
# in raw data. For example, modifies gravity for cool dwarf stars.
def cannon_guess_choice(object_cannon_data, cannon_quality, reduction_and_analysis_data):
    """
    Input:
        object_cannon_data: Parameter data from the CANNON file
        cannon_quality: The quality of said CANNON Data. Too low and we use GUESS instead.
        reduction_and_analysis_data: Less accurate but more reliable GUESS data for our starting parameters.
    Output:
        starting_parameters: A dictionary containing several parameters, listed below:
            effective_temperature
            gravity
            metallicity
            radial_velocity_global
            rotational_velocity
            microturbulence_velocity
            macroturbulence_velocity
    """

    # The most important is the len(). Some stars aren't available in Cannon, so we use GUESS instead.
    # Same occurs for if cannon quality is too low. Either way we obtain the starting parameters.
    if cannon_quality <= 1 and len(object_cannon_data) != 0:
        starting_parameters = cannon_attempt_1(object_cannon_data)
    else:
        print("Cannon quality is TOO LOW or does not exist. Using GUESS instead.")
        starting_parameters = guess_attempt_2(reduction_and_analysis_data)

    # This one comes from the iraf/bailer jones, not cannon, regardless for if we were using cannon before so is the
    # same call for both functions, hence we have it out here.
    starting_parameters['radial_velocity_global'] = reduction_and_analysis_data['rv_guess']

    return starting_parameters


# We find the index and then the data and quality of the cannon (machine attempt) data for the observations first guess
def cannon_index(object_for_obs):
    """
    Input:
        Object number: to find the correct files and data.
    Output:
        object_cannon_data: The estimated starting data such as for Teff, Logg, etc. for the object.
        cannon_quality: The quality of the cannon estimate. Too low and we try the GUESS estimate instead.
    """
    # Open the cannon data file containing machine learning estimates on starting variables for all objects in Galah
    cannon_data = fits.open(r"GALAH/DATA/sobject_iraf_iDR2_cannon_small.fits")

    # Find the index of our desired object
    object_index = np.where(cannon_data[1].data['sobject_id'] == object_for_obs)
    # Then obtains the data for just our object.
    object_cannon_data = (cannon_data[1].data[object_index])
    # If .size is bigger than 0, it exists. If it == 0, we stop the script and return 0s for data
    # and 10 for quality so it starts using GUESS instead.
    if not object_index[0].size:
        print("Object does not exist in Cannon.")
        object_cannon_data, cannon_quality = np.array(()), 10
        return object_cannon_data, cannon_quality

    # this is the quality of the cannon approx. Above 1 and we want to do a different method. 0 is ideal.
    cannon_quality = (object_cannon_data['flag_cannon'])

    print("Cannon flag quality is at ", cannon_quality)

    return object_cannon_data, cannon_quality


# Micro and macro turbulence affect the flux we see, changing our guesses on gravity and T etc
# But macro turbulence effects are so similar to radial velocity effects that our resolution cannot distinguish them
# hence we scrap the macro and just use vsini (radial velocity)
def microturbulence_macroturbulence(temperature, gravity):
    # Prevents warning on reference being assigning.
    ai1 = 0
    ai2 = 0
    ai3 = 0
    aa1 = 0
    ba1 = 0
    ca1 = 0
    if gravity <= 4.2 or temperature >= 5500:
        ai1 = 1.1
        ai2 = 1E-4
        ai3 = 4E-7
        aa1 = 1.5
        ba1 = 0
        ca1 = 1E-6
        # ai1 = 1 ai2 = -2.5E-4 ai3 = 6.5E-7
    elif gravity >= 4.2 and temperature < 5500:
        ai1 = 1.1
        ai2 = 1.6E-4
        ai3 = 0
        aa1 = 1.5
        ba1 = 0.2E-3
        ca1 = 0
    # It should not be able to miss the two other if statements.
    if ai1 == 0:
        print("Something wrong with microturbulence code and its if requirements. Check it out.")
        exit()
    temperature_base = 5500
    gravity_base = 4.0
    # vmic =      ai1 + ai2*(t -t0) + ai3 * (t-t0)^2 + bi1 + bi2 * (g-g0) + bi3 * (g-g0)^2 + ci1 + ci2 * z + ci3 * z^2
    # vmac = 3 * (aa1 + ba1*(t -t0) + ca1 * (t-t0)^2 + ba2*(g-g0) + ca2*(g-g0)^2 + ba3*z + ca3*z^2) + 2.0
    microturbulence_velocity = ai1 + ai2 * (temperature - temperature_base) + \
                               ai3 * (temperature - temperature_base) ** 2

    macroturbulence_velocity = 3 * (aa1 + ba1 * (temperature - temperature_base) + ca1
                                    * (temperature - temperature_base) ** 2)

    return microturbulence_velocity, macroturbulence_velocity


"""1st try is with Cannon, the machine learning software that gives reasonable first guesses for our parameters"""
def cannon_attempt_1(object_cannon_data):
    """
    Input:
        object_cannon_data: Data from the cannon file for starting parmeters.
    Output:
        starting_parameters: A dictionary with keys of
            effective_temperature
            gravity
            metallicity
            rotational_velocity
            microturbulence_velocity
            macroturbulence_velocity
    """

    print("Using cannon data")
    # As both GUESS And CANNON use the same function, the 2nd input tells the function which file to use. CANNON or
    # GUESS. From there it takes the starting parameters for our spectroscopy and puts it in a dictionary.
    starting_parameters = get_parameter_data(object_cannon_data, "cannon")

    # We use these variables a lot in this function, and so we set them individually before inputting them back into 
    # the dictionary.
    effective_temperature = starting_parameters['effective_temperature']
    gravity = starting_parameters['gravity']

    # These checks are for if the Cannon made a poor guess of the gravitational effect and it was off the main sequence
    # HR evolution. If that occurs we reset them to reasonable numbers.
    if effective_temperature < 5250 and gravity > (
            4.25 - (4.25 - 3.5) * (5250 - effective_temperature) / 1250):

        print("Cool dwarf (Upturn) [what's upturn] found. Adjusting log(g) and metallicity [Fe/H]")

        gravity = (4.5 + (0.2 * (5250 - effective_temperature) / 1250))
        starting_parameters['metallicity'] = 0
        starting_parameters['microturbulence_velocity'], starting_parameters['macroturbulence_velocity'] = (
            microturbulence_macroturbulence(effective_temperature, gravity))
        # Can't differ between the two with low res, so we use rotational velocity.
        starting_parameters['rotational_velocity'] = starting_parameters['macroturbulence_velocity']
        starting_parameters['macroturbulence_velocity'] = 0.0

    # Possible parameters to distinguish between giants and dwarfs
    # Estimated with Dartmouth isochrones for metallicity = -0.5dex, age = 15Gyr 4500K, 2 dex and 3650 K, 0.5 dex
    elif effective_temperature < 4500 and starting_parameters['metallicity'] < -0.75 and \
            gravity > (2 - ((2 - 0.5) * (4500 - effective_temperature) / 850)):
        print('Possible cool giant (TiO) identified, adjusting gravity and metallicity')
        # gravity = 4.5 + (0.2*(5250-effective_temperature)/1250)
        starting_parameters['metallicity'] = 0
        starting_parameters['microturbulence_velocity'], starting_parameters['macroturbulence_velocity'] \
            = (microturbulence_macroturbulence(effective_temperature, gravity))
        # Can't differ between the two with low res, so we use rotational velocity.
        starting_parameters['rotational_velocity'] = starting_parameters['macroturbulence_velocity']
        starting_parameters['macroturbulence_velocity'] = 0.0

    elif effective_temperature < 4250 and gravity < 2 and starting_parameters['metallicity'] < -0.75:
        print("Giant at end of Cannon training set identified, adjusting metallicity")
        starting_parameters['metallicity'] = 0.0

    # And we put the temperature and gravity back into the dictionary.
    starting_parameters['effective_temperature'] = effective_temperature
    starting_parameters['gravity'] = gravity
    return starting_parameters


' 2nd try GUESS. Very similar to CANNON but using GUESS data instead, and velocities are obtained in a different way.'
def guess_attempt_2(reduction_and_analysis_data):
    """
    Input:
        reduction_and_analysis_data: Data from the GUESS file for starting parmeters.
    Output:
        starting_parameters: A dictionary with keys of
            effective_temperature
            gravity
            metallicity
            rotational_velocity
            microturbulence_velocity
            macroturbulence_velocity
    """

    if (reduction_and_analysis_data['flag_guess']) != 0:
        print("Guess quality too low.")
        return
    print("Guess flag quality is at ", reduction_and_analysis_data['flag_guess'])

    # As both GUESS And CANNON use the same function, the 2ND input tells the function which file to use. CANNON or
    # GUESS. From there it takes the starting parameters for our spectroscopy and puts it in a dictionary.
    starting_parameters = get_parameter_data(reduction_and_analysis_data, "guess")
    gravity = starting_parameters['gravity']
    effective_temperature = starting_parameters['effective_temperature']

    # Modifies gravity to a more accurate level.
    if gravity > 3.5:
        gravity = gravity + 0.5
        print("NB: Offset to initial GUESS log(g) by +0.5 for log(g) > 3.5")
    # Modifies metallicity to a more accurate level.
    starting_parameters['metallicity'] = starting_parameters['metallicity'] + 0.15
    print("NB: Offset to initial GUESS [Fe/H] by +0.15")

    # Initial GUESS does not work so well on these stars so we modify them further
    if effective_temperature < 5250 and gravity > (
            4.25 - (4.25 - 3.5) * (5250 - effective_temperature) / 1250):
        print("Cool dwarf (Upturn) found. Adjusting log(g) and metallicity [Fe/H]")
        gravity = (4.5 + (0.2 * (5250 - effective_temperature) / 1250))
        starting_parameters['metallicity'] = 0

    if effective_temperature < 4250 and gravity < 2.0 and starting_parameters['metallicity'] < -0.75:
        print("Giant below 4250K identified, adjusting [Fe/H] to 0")  # Why? @@@@@@@@@@@@
        starting_parameters['metallicity'] = 0

    # We obtain velocities purely through the turbulence function with GUESS as it has no other data on those values.
    starting_parameters['microturbulence_velocity'], starting_parameters['macroturbulence_velocity'] = (
        microturbulence_macroturbulence(effective_temperature, gravity))
    starting_parameters['rotational_velocity'] = starting_parameters['macroturbulence_velocity']
    starting_parameters['macroturbulence_velocity'] = 0.0

    # Flags reduction issues if they appear
    if reduction_and_analysis_data['red_flag'] != 0:
        # bit mask for 1 to 8, summing up CCD problem, e.g. (0011) = 1+2=3 for BG
        print("Reduction issue! Red flag: ", reduction_and_analysis_data['red_flag'])
        if reduction_and_analysis_data['red_flag'] == 14:
            print("Skyflat!")

    # Add the new temperature and gravity to the starting parameters again.
    starting_parameters['effective_temperature'] = effective_temperature
    starting_parameters['gravity'] = gravity

    return starting_parameters


# Skipping for now.
def gaussian_attempt_3():
    print("Neither Cannon nor GUESS appropriate, trying Gaussian approach. Not yet implemented.")
    exit()


# Grabbing the variable data like log_g from the either GUESS or CANNON data, depending on what's input.
def get_parameter_data(object_data, guess_or_cannon):

    # We use a dictionary to kep starting parameters in.
    starting_parameters = {}
    starting_parameters['effective_temperature'] = object_data['Teff_' + guess_or_cannon]
    starting_parameters['gravity'] = object_data['Logg_' + guess_or_cannon]
    starting_parameters['metallicity'] = object_data['Feh_' + guess_or_cannon]

    # Only cannon has this data. GUESS does not.
    if str(guess_or_cannon) == 'cannon':
        starting_parameters['microturbulence_velocity'] = object_data['Vmic_' + guess_or_cannon]
        starting_parameters['rotational_velocity'] = object_data['Vsini_' + guess_or_cannon]
        starting_parameters['macroturbulence_velocity'] = 0

    return starting_parameters


# Account for special modes: LBOL, SEIS, FIXED
# If Seis, prepare asteroseismic info
# If LBOL, prepare photomettric/astrometric info
# If Lbol/Seis, update logG based on provided info.
def update_mode_info(field_end, field_for_obs, reduction_and_analysis_data):
    """
    Input:
        field_end: Type of mode. LBOL, SEIS, FIXED
        field_for_obs: Checks for if it is the sun. Requires special measures for if it is.
        reduction_and_analysis_data: Information on the star to produce the astromtric/asteroseismic information.
    Output:
        reduction_variable_dict: Contains the astromtric/asteroseismic data such as distance, parallax, magnitude of
                                k band, and more.
    """
    # Dictionary to put the variables in.
    reduction_variable_dict = {}
    # If LBOL mode AND the Sun, we use a separate update function for it being a special case.
    if 'sun' in field_for_obs and 'lbol' in field_for_obs:
        reduction_variable_dict = sun_lbol_update()

    # Elif just running in lbol mode, but not the sun we use a mnormal lbol update.
    elif 'lbol' in field_for_obs:
        # Checks to see if it exists with good information.
        if reduction_and_analysis_data['r_est'] > 0:
            reduction_variable_dict = lbol_update(reduction_and_analysis_data)
        # If it doesn't, we can't continue.
        else:
            print("Star not in Gaaia DR2, but lbol-keyword activated. Cancel.")
            exit()

    # If SEIS mode: get numax. A separate method to lbol to retrieve required astroseismic data.
    # Numax: The value of frequency at which the oscillation displays its strongest pulsation amplitude.
    if (field_end.lower()) == 'seis':
        try:
            numax = (reduction_and_analysis_data['numax'])
            e_numax = 0
            reduction_variable_dict = {'nu_max': numax[0], 'e_numax': e_numax}
        except KeyError as missingvalue:
            print("No numax value. Returning.")
            exit()

    return reduction_variable_dict

# A special lbol update for it being the sun, such as assuming distance to be 10pc
def sun_lbol_update():
    """
    Output:
        reduction_variable_dict: Astrostonmical values of the given object listed below. See the SME manual for details.
            h_mag,
            k_mag,
            ebv,
            plx,
            e_plx,
            dist,
            dist_lo,
            dist_hi,
            e_dist,
            e_k_mag,
            a_k,
            e_a_k,
            lbol
    """
    print("Chosen object is the sun, requiring a special case lbol update.")
    # Setting various required data points. See the sme manual for a description.
    k_mag = 3.28
    h_mag = 3.32
    plx = 100
    e_plx = 0.0
    dist = 10.0  # assuming 10 pc
    e_dist = 0.01
    ebv = 0.0

    e_k_mag = 0.01
    dist_lo = 9.9
    dist_hi = 10.1
    a_k = 0.00
    e_a_k = 0.03
    lbol = 0
    reduction_variable_dict = {'h_mag': h_mag, 'k_mag': k_mag, 'ebv': ebv, 'plx': plx, 'e_plx': e_plx,
                               'dist': dist, 'dist_lo': dist_lo, 'dist_hi': dist_hi, 'e_dist': e_dist,
                               'e_k_mag': e_k_mag, 'a_k': a_k, 'e_a_k': e_a_k, 'lbol': lbol}

    return reduction_variable_dict


# Take data from the reduction and analysis data (GUESS) as starting values for our astronomical LBOL data..
def lbol_update(reduction_and_analysis_data):
    """
    Input:
        reduction_and_analysis_data: GUESS data containing first guess data on stellar paramters such as distance.
    Output:
        reduction_variable_dict: Astrostonmical values of the given object listed below. See the SME manual for details.
            h_mag,
            k_mag,
            ebv,
            plx,
            e_plx,
            dist,
            dist_lo,
            dist_hi,
            e_dist,
            e_k_mag,
            a_k,
            e_a_k,
            lbol
    """
    print("Star in Gaia DR2 (And possible 2MASS and WISE. Running LBOL update.")

    # See SME manual for more information on these variables.
    j_mag = reduction_and_analysis_data['j_m']
    e_j_mag = reduction_and_analysis_data['j_msigcom']
    h_mag = reduction_and_analysis_data['h_m']
    e_h_mag = reduction_and_analysis_data['h_msigcom']
    k_mag = reduction_and_analysis_data['ks_m']
    e_k_mag = reduction_and_analysis_data['ks_msigcom']
    w2_mag = reduction_and_analysis_data['w2mpro']
    e_w2_mag = reduction_and_analysis_data['w2mpro_error']
    ebv = reduction_and_analysis_data['ebv']
    lbol = 0
    # USed in update logg. extintion of ks band?
    a_k = 0.918 * (h_mag - w2_mag - 0.08)
    e_a_k = e_h_mag ** 2 + e_w2_mag ** 2
    if a_k < 0:
        a_k = 0

    print("A_Ks from RJCE:", str(a_k), '+/-', str(e_a_k))
    print("Quality flags (2MASS+WISE)", str(reduction_and_analysis_data['ph_qual_tmass']), '+&',
          str(reduction_and_analysis_data['ph_qual_wise']))
    print("A_Ks from 0.36*E(B-V)", str(0.36 * ebv))

    #  Needs [0] first as is a list and we take the first value
    if np.isinf(float(h_mag)) or np.isinf(float(w2_mag)) or \
            str(reduction_and_analysis_data['ph_qual_tmass'][0][1]) != 'A' or \
            str(reduction_and_analysis_data['ph_qual_wise'][0][1]) != 'A':
        a_k = 0.36 * ebv

    if (0.36 * ebv) > (3 * a_k):
        ebv = 2.78 * a_k
        print("E(B-V) not trustworthy, setting to 2.78*A_Ks =", str(ebv), "instead of",
              str(reduction_and_analysis_data['ebv']))

    plx = reduction_and_analysis_data['parallax']
    e_plx = reduction_and_analysis_data['parallax_error']
    dist = reduction_and_analysis_data['r_est']
    print("Distance", str(dist), "with limits", str(reduction_and_analysis_data['r_lo']), "to",
          str(reduction_and_analysis_data['r_hi']))

    e_dist = 0.5 * (reduction_and_analysis_data['r_hi'] - reduction_and_analysis_data['r_lo'])
    if dist < 100:
        print("Star within 100pc, setting A_K and EBV to 0")
        a_k = 0
        ebv = 0

    reduction_variable_dict = {'j_mag': j_mag[0], 'e_j_mag': e_j_mag[0], 'h_mag': h_mag[0], 'e_h_mag': e_h_mag[0],
                               'k_mag': k_mag[0], 'e_k_mag': e_k_mag[0], 'w2_mag': w2_mag[0], 'e_w2_mag': e_w2_mag[0],
                               'ebv': ebv, 'a_k': a_k,
                               'plx': plx[0], 'e_plx': e_plx[0], 'dist': dist[0], 'e_dist': e_dist[0],
                               'dist_lo': reduction_and_analysis_data['r_lo'][0],
                               'dist_hi': reduction_and_analysis_data['r_hi'][0], 'e_a_k': e_a_k[0], 'lbol': lbol}
    return reduction_variable_dict


# We run the pysme_update_logg to modify gravity based on the starting parameters we have. We continiously run that
# file throughout, but with the updated parameters instead. Also updates the velocity parameters based on a new gravity.
def update_gravity_function(reduction_variable_dict, field_end, starting_parameters):
    """
    Input:
        reduction_variable_dict: Astronomical data set before from GUESS including distance and paralax and more,
                                    but not temperature, etc.
        field_end: The type of run we are performing. Lbol or Seis.
        starting_parameters: The initial parameters of temperature, gravity, metallicity, and other less important ones.
    Output:
        starting_parameters: With modified gravity and velocity parameters
    """
    # Update gravity. How we update it depends on Seis or Lbol
    gravity = \
        pysme_update_logg.optimize_logg_with_extra_input(starting_parameters, reduction_variable_dict, field_end)

    # The velocities depend on gravity as well, so we now must modify them.
    microturbulence_velocity, macroturbulence_velocity =\
        microturbulence_macroturbulence(starting_parameters['effective_temperature'], gravity)
    rotational_velocity = macroturbulence_velocity
    macroturbulence_velocity = 0

    if field_end == 'lbol':
        print("Running in LBOL mode, glob_free are TEF, FEH, VRAD, (VSINI), GRAV adjusted from TEFF, KMAG, PLX")

    elif field_end == 'seis':
        # Nu is the frequency of the oscillation.
        print("Running in SEISMIC mode, glob_free are TEF, FEH, VRAD, (VSINI), GRAV adjusted from TEFF, NU_MAX")

    # It shouldn't be able to reach here, but if it does then something unexpected occured with field_end and we stop.
    else:
        print("Error with naming convention in galah3. Field_end should be lbol or seis, but is", field_end)
        exit()

    starting_parameters['microturbulence_velocity'] = microturbulence_velocity
    starting_parameters['macroturbulence_velocity'] = macroturbulence_velocity
    starting_parameters['rotational_velocity'] = rotational_velocity
    starting_parameters['gravity'] = gravity
    return starting_parameters


"""Main part 3.3/5" 
Ensure reasonable parameters"""


# We check to see if our first guess parameters are within reasonable limits. If not, we adjust them slightly fit.
# We adjust gravity if the temperature is out of bounds.
def reasonable_parameters(atmosphere_grid_file, starting_parameters):
    """
    Input:
        atmosphere_grid_file: The name of the atmosphere we use, as each will have different limitations on their
                                predictions.
        starting_parameters: The initial parameters of temperature, gravity, metallicity, and other less important ones.
    Output:
        starting_parameters: With modified gravity, metallicity, and temperature parameters

    """
    # Brings the values to within those boundaries.
    effective_temperature = np.clip(starting_parameters['effective_temperature'], 3000, 7500)
    gravity = np.clip(starting_parameters['gravity'], 0.5, 5)
    metallicity = np.clip(starting_parameters['metallicity'], -3, 0.5)

    # Adjust logg to marcs' lower grid limits of teff. Beyond that is not accurate enough to use these atmospheres.
    if atmosphere_grid_file == 'marcs2014.sav' or atmosphere_grid_file == 'marcs2012.sav':
        # Sets grav to the maximum of either 2 or gravity (clipping with only a minimum value, no max)
        if 7500 > effective_temperature > 6500:
            gravity = np.maximum(2, gravity)
        elif 6500 > effective_temperature > 5750:
            gravity = np.maximum(1.5, gravity)
        elif 5750 > effective_temperature > 5000:
            gravity = np.maximum(1, gravity)
        elif 5000 > effective_temperature > 4000:
            gravity = np.maximum(0.5, gravity)

    # Do the same but with the stagger file limits for temperature, although still adjusting gravity according to temp.
    elif atmosphere_grid_file == 'stagger-t5havni_marcs64.sav' or \
            atmosphere_grid_file == 'stagger-t5havni_marcs.sav' or \
            atmosphere_grid_file == 'stagger-tAA.sav':

        # Sets grav to the maximum of either 2 or gravity (clipping with only a minimum value, no max)
        if 4750 > effective_temperature:
            gravity = np.maximum(1.75, gravity)
        elif 5250 > effective_temperature > 4750:
            gravity = np.maximum(2.75, gravity)
        elif 5750 > effective_temperature > 5250:
            gravity = np.maximum(3.25, gravity)
        elif 6250 > effective_temperature > 5750:
            gravity = np.maximum(3.75, gravity)
        elif effective_temperature >= 6250:
            gravity = np.maximum(4.15, gravity)

    print("\nStarting values are: \nTeff:", effective_temperature, "\nGravity (log_g):", gravity,
          "\nmetallicity:", metallicity, "\n")

    starting_parameters['effective_temperature'] = effective_temperature
    starting_parameters['metallicity'] = metallicity
    starting_parameters['gravity'] = gravity

    return starting_parameters
