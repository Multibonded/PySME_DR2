"""This section of pysme galah is basically just a set up of variables that we use later in the other scripts.
Some options such as object number are up to the user to decide. Others, like atmosphere are still available for change
but it is recommended to leave as default unless the user knows why they are changing it."""

import numpy as np

"""# 150204002101256 is a random we have
# 150427000801049 is cannon test.
# 150210005801171 is arcturus
# 150405000901378 is the sun."""
# Basic set up for the object to be able to find the object in the large data releases.
def setup_object():
    """
    Output:
    obs_name: Combination of object number, method of calculation, and the data set being used.
    obs_file: Obs_name but with the addition of the file type. We are currently using .dat
    object_for_obs: The number of the object. Primary method of indentification when indexing data releases
    field_end: Displays the method of calculation. Lbol or Seis. Takes the final 4 characters of field_for_obs
    setup_for_obs: The data set being used.
    field_for_obs: Shows the method of calculation, but also other parameters such as whether it's a test, benchmark, is
                    the sun, etc.
    iterations: How many iterations to run Pysme for (iterations[x] for x pysme_execute run) and how many times to
                run pysme_execute (len(iterations))
    """
    # Setting object number, used for indentification.
    object_for_obs = 150210005801171  # input("Object, please.")
    # The sun is a special case. Must be noted if running this one with 'sun' in the name as it requires additional
    # calculation.
    if object_for_obs == 150405000901378:
        field_for_obs = "Benchmark_sun_lbol"  # input("Field, please.")
    # Arcturus is famous and so we make it known if we are running this.
    elif object_for_obs == 150210005801171:
        field_for_obs = "Benchmark_arcturus_lbol"  # input("Field, please.")
    # Some may require 'seis' and so that should be included as a possibility.
    else:
        field_for_obs = "lbol"  # input("lbol or seis?")

    # Represents if it's an lbol or seis run. Used in many pysme files.
    field_end = field_for_obs[-4:]

    # Displays the data set being used. Used to find the files in the Galah folder.
    setup_for_obs = "DR2"  # input("Setup, please.")

    # Combines all variables so we can save files as a unique named file.
    obs_name = str(field_for_obs) + "_" + str(object_for_obs) + "_" + str(setup_for_obs) + "_Sp"
    # The object data file to find. Not in obs_name as we don't want to save as .dat
    obs_file = str(obs_name + ".dat")

    # Four SME calls : normalise, iterate, normalise, iterate
    iterations = [1, 2, 1, 20]

    return obs_name, obs_file, object_for_obs, field_end, setup_for_obs, field_for_obs, iterations

# Setting up of the atmosphere file names.
def set_atmosphere():
    """
    Output:
    atmosphere_grid_file: The name of the atmosphere file to use in pysme.
    line_list: The name of the file that contains all atomic data we use in our regular non-balmer runs.
    atomic_abundances_free_parameters: Unused at the moment but may be required in future.
    atmosphere_abundance_grid: A list containing the NLTE element grid file names.
    """
    """Choosing atmoshpere, grids, and NLTE."""
    # Atmosphere grid file to use.
    atmosphere_grid_file = 'marcs2014.sav'
    # Linelist containing atomic data
    line_list = 'galah_master_v5.1.fits'
    # Unused
    atomic_abundances_free_parameters = np.zeros(99)
    # NLTE file names. [0] must be H and [1] Fe, or change it in pysme_run_sme
    atmosphere_abundance_grid = ['marcs2012_H2018.grd', 'marcs2012_Fe2016.grd']

    return atmosphere_grid_file, line_list, atomic_abundances_free_parameters, atmosphere_abundance_grid

# Other variable we use in our files.
def sme_variables():
    """Output:
    broad_lines: The balmer lines and other important lines to note. Primarily used in balmer_runs for Halpha and beta
    depthmin: minimum depth required to include atomic lines. Too low will include too many and take too long to run
    line_cores: Important line number to note.
    """
    depthmin = 0.1
    print('Using minimum depth ', depthmin)
    # Balmer lines are the first two. Adding here gives them into alllinesindex which we use to identify desired
    # linelist lambdas.
    broad_lines = [4861.3230, 6562.7970, 5167.3216, 5172.6843, 5183.6042]  # Hbeta, Halpha, Mg Ib triplet
    line_cores = [8498.0200]
    # Imaginary number as 0 and the like are reasonable results for some things.


    return broad_lines, depthmin, line_cores


