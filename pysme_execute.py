"""We run pysme galah1-4, makestruct, and run_sme to run PySME. We use only data collected in Galah folders, and output
into OUTPUT/ where we house the spectra, variables, and any other files created in the mean time.
This file is mainly a collection of all other files. Everything involved is titled pysme_x

We use Galah data and, after modification, use Pysme to produce a synthetic spectra based on our variables. Then we
modify those variables -or Free Parameters- and model the spectra again. When ours matches the observed spectra to a
reasonable degree, we have accurate estimates on things such as abundance, temperature, gravity, and more."""
import pysme_galah1
import pysme_galah2
import pysme_galah3
import pysme_galah4

# We run each pysme_galah file from 1 to 3. 4 comes after.
"""
Galah 1: Input names for atmosphere files, object number, etc. Some can be user input. Specifically the 
desired stellar object.
Galah 2: Take the stellar spectra and adjust it for potential error in the atmosphere.
Galah 3: Get starting guesses for variables such as Teff for the object."""

class read_iso:
    def __init__(self, age):
        self.num_cols=4
        self.columns = ['M_Mo', 'logTeff', 'logG', 'logL_Lo']
        self.num_ages = len(age)
        self.ages = age

    def fill_chemistry(self, m_h, fe_h, alpha_fe):
        self.FeH = fe_h
        self.Z = 10**m_h*0.0152
        self.aFe = alpha_fe

    def fill_iso(self, iso_input):
        self.data = iso_input


def collect_data():
    """
    Output: Makestruct dict: a dictionary that contains most variables we need to modify our spectra and run sme

            Reduction_variable_dict: A less useful dictionary but still contains some useful variables. These
            come directly from a galah data file and are not modified.
    """

    # Initial set up. We return the object name
    obs_name, obs_file, object_for_obs, field_end, setup_for_obs, field_for_obs, iterations = \
        pysme_galah1.setup_object()

    # Setting up of which atmosphere files to use in pysme. Returns primarily strings or lists.
    atmosphere_grid_file, line_list, atomic_abundances_free_parameters, atmosphere_abundance_grid = \
        pysme_galah1.set_atmosphere()

    # Collects a few important variables. Primarily line locations, and the minimum depth for atomic lines.
    broad_lines, depthmin, line_cores = \
        pysme_galah1.sme_variables()

    """Galah 2: Read in spectra + resolution + interpolation to add resolution 
    and future wavelengths we decide + correct telluric & skyline error (atmospheric errors)"""
    # Produces interpolation to produce resolutions for different wavelengths we produce during the run
    interpolation, resolution_factor = \
        pysme_galah2.get_resolution(object_for_obs)

    # Opens galah ccd data files to produce the light spectra graphs with wavelength, flux, and error as outputs.
    total_ccd_wavelength, total_ccd_sob_flux, total_ccd_relative_flux_error = \
        pysme_galah2.get_wavelength_flux(object_for_obs)

    # Opens a large data file to index it down to our given object only, containing data such as macroturbulent velocity
    reduction_and_analysis_data = \
        pysme_galah2.data_release_index(object_for_obs)

    # We use our spectra data so far and correct for telluric and skyline error, returning information in a cleaner
    # dictionary form. We also require barycentric velocity of the object.
    ccd_data_dict = \
        pysme_galah2.error_correction(reduction_and_analysis_data['v_bary'], total_ccd_wavelength, total_ccd_sob_flux,
                                      total_ccd_relative_flux_error)

    """Part   3) Determine initial stellar parameters
              3.1) Based on old Cannon/GUESS/default
              3.2) If run with field_end LBOL, SEIS, FIXED, update initial parameters
              3.3) Ensure reasonable parameters"""

    # We find the index and then the data and quality of the cannon (machine attempt) data for the observations
    # initial guess of starting parameters. If it's not there, we use GUESS which is less accurate but more precise.
    object_cannon_data, cannon_quality = \
        pysme_galah3.cannon_index(object_for_obs)

    # We take either cannon or GUESS data for the initial first choices of starting variables for the star,
    # depending on quality and existance.
    starting_parameters = \
        pysme_galah3.cannon_guess_choice(object_cannon_data, cannon_quality, reduction_and_analysis_data)

    # Needed to modify solar data as its vrad is wrong in galah_master file
    if 'sun' in obs_name:
        starting_parameters['radial_velocity_global'] = [-16]

    # Performs a lot of small variable modifications, such as j_mag and parallax. Used primarily in pysme_update_logg.
    reduction_variable_dict = \
        pysme_galah3.update_mode_info(field_end, field_for_obs, reduction_and_analysis_data)

    # We adjust the velocities and grav. for these types of runs only.
    # Here is where we run pysme_update_logg initially and is the last update before running makestruct unless it's
    # unreasonable, in which case it'll be modified in th reasonable_parameters.
    if field_end == 'lbol' or 'seis':
        starting_parameters = \
            pysme_galah3.update_gravity_function(
                reduction_variable_dict, field_end, starting_parameters)

    # We check to see if our first guess parameters are within reasonable limits. If not, we adjust them slightly fit.
    # We adjust gravity if the temperature is out of bounds.
    starting_parameters = \
        pysme_galah3.reasonable_parameters(atmosphere_grid_file, starting_parameters)

    # gam6 in the dictionary is a global correction factor to all van der Waals damping constants. Values of
    # 1.5 to 2.5 are sometimes used for iron.

    # Setting up the dictionary we're going to input a lot of variables we want for makestruct, which in turn uses
    # them for PySME. Setting it here as it's quite important and helps readability.

    makestruct_dict = {'setup_for_obs':          setup_for_obs,          'resolution_factor':    resolution_factor,
                       'obs_name':               obs_name,               'object_for_obs':       object_for_obs,
                       'global_free_parameters': [],                     'broad_lines':          broad_lines,
                       'unnormalised_spectra':   ccd_data_dict,          'atmosphere_grid_file': atmosphere_grid_file,
                       'iterations':             iterations,             'depthmin':             depthmin,
                       'line_list':              line_list,              'balmer_run':           False,
                       'line_cores':             line_cores,             'normalise_flag':       False,
                       'field_end':              field_end,              'original_location':    'GALAH/LINELIST/',
                       'load_file':              False,                  'gam6':                 1,
                       'segment_mask':           setup_for_obs + "_Segm.dat",
                       'original_line_list':     line_list,              'nlte_abund': atmosphere_abundance_grid,
                       'interpolation':          interpolation,
                       'line_list_location':     'GALAH/LINELIST/',
                       'line_mask':              "GALAH/DATA/" + setup_for_obs + '_Sp.dat',
                       'atomic_abundances_free_parameters':               atomic_abundances_free_parameters,

                       }
    # Now we add the starting parameters of temperature, gravity, metallicity, and velocities to the makestruct input
    makestruct_dict.update(starting_parameters.items())
    print("Starting variables:"
          "\nTemperature:",     makestruct_dict['effective_temperature'],
          "\nLog_g:",           makestruct_dict['gravity'],
          "\nMetallicity:",     makestruct_dict['metallicity'],
          "\nRadial Velocity:", makestruct_dict['radial_velocity_global'])

    """Part  4) Optimisation of stellar parameters (loop with alternating 4.1 and 4.2)
             4.1) Segment selection and normalisation (fixed parameters)
             4.2) Iterative optimisation (fixed segments and normalisation)."""

    return makestruct_dict, reduction_variable_dict


# We run all code, and perform the total sme run. Outputs are stored in OUTPUT/ where we can find the produced spectra
# and produced parameters such as temperature for the star chosen.
def execute():
    # Runs collect_data to produce a dictionary full of starting values to begin our spectra creation
    makestruct_dict, reduction_variable_dict = collect_data()
    # Iteratively runs PySME to produce the spectra or paramters, and modifies the output slightly to reach a higher
    # accuracy.
    pysme_galah4.iterative_loop(makestruct_dict, reduction_variable_dict)


execute()
