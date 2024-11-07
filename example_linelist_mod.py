import pysme_readlines
import pickle

"""I've copied and pasted a lot of functions that were used in other files. The linelist is loaded from the file,
modified in pysme_readlines file, then extracted and modified some more in the other functions below to get it 
in the Pandas Dataframe form that's made from a dictionary like:

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

    data["error"] = error
    data["lande_lower"] = linelist['line_lulande'][:, 0]
    data["lande_upper"] = linelist['line_lulande'][:, 1]
    data["j_lo"] = linelist['line_extra'][:, 0]
    data["e_upp"] = linelist['line_extra'][:, 1]
    data["j_up"] = linelist['line_extra'][:, 2]
    data["term_lower"] = [t[10:].strip() for t in linelist['line_term_low']]
    data["term_upper"] = [t[10:].strip() for t in linelist['line_term_upp']]

The one to run is
pre_dataframe. Sorry I sound so vague, I hope the comments help!
Any suggestions of code or comments are welcome"""


# Produces the arrays with the stellar information from the master file for every wavelength. such as line_atomic etc.
# Either loads a previously modified linelist, or makes a modified linelist using the data we load from GALAH in
# load_linelist, and then modifying various parameters within it such as the species name to include ionization.
def indexed_stellar_information(makestruct_dict):
    """
    Input:
        makestruct_dict: Information on the chosen line list, and whether it's already been modified.]
        What you need is ['line_list'] = The line list name in a string , ['line_list_location'] being where you've put it and
        ['original_location'] being the same, but is used for my own code to keep track of where things are.
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
        master_line_list = fits.open(makestruct_dict['original_location'] + makestruct_dict['line_list'])[1]

        # This function modifies the linelist data and saves the output in the OUTPUT/ directory
        pysme_readlines.run_merger(master_line_list, makestruct_dict['line_list_location'],
                                   makestruct_dict['line_list'])
        # The linelist is now modified so we want to always call this new one as it has an updated depth
        makestruct_dict['line_list'] = makestruct_dict['line_list'][:-5] + '_modified.csv'

        # Now we know it exists, we can do what we were trying to before.
        sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
                                        makestruct_dict['line_list'], "rb"))


    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        sme_linelist['atomic'], sme_linelist['lande'], \
        sme_linelist['depth'], sme_linelist['lineref'], \
        sme_linelist['species'], sme_linelist['line_extra'], \
        sme_linelist['line_term_low'], sme_linelist['line_term_upp'], \
        sme_linelist['line_lulande']

    return line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande


# We produce an indexed version of the large amount of atomic data that is contained in the files used, taking only the
# atomic lines that are either in the segments only, or in the linemask depending on what we send the variable 'run' to.


# To Tom and Ella: the 2nd, 3rd and 4th variable are only used in making the atomic index (Explained below)
def produce_indexed_atomic_data(makestruct_dict, segment_begin_end=0, linemask_data=0, run="Synthesise"):
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



    print("Read this first, then delete me and the exit clause below: "
          "The below block of comment is code to select the indexes in either the "
          "segments or linemask ranges. I don't know which wavelengths you want, so I remove it. All it produces is "
          "a numpy array for the indexes of the line_atomic that you're interested in running in pysme.")
    # This should be the indexes you want
    desired_atomic_lines_index = 0
    exit()
    """# If run is Solve, we want the indexes of wavelengths in segments we already have,
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
    print(nselect, "unique spectral lines are selected within wavelength segments out of", len(line_atomic))"""

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



def pre_dataframe(makestruct_dict):

    """      Input:
        makestruct_dict: Information on the chosen line list, and whether it's already been modified.]
        What you need is ['line_list'] = The line list name in a string , ['line_list_location'] being where you've put it and
        ['original_location'] being the same, but is used for my own code to keep track of where things are.
 """


    # Information on atomic lines such as wavelength, species, ionization. Primarily used during the PySME run itself.
    # During solve runs we take only atomic data in the segment mask, but that still appears to be the majority.
    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, lower_level, upper_level, lu_lande = \
        produce_indexed_atomic_data(makestruct_dict, segment_begin_end, linemask_data, run=run)




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


    print("@Ella/Tom: This is what we use to load it into pysme")
    from pysme.linelist.linelist import LineList
    # either read the file created by idl_conversion or load the input linelist from makestruct.
    # long is nlte, short is lte
    sme.linelist = LineList(linedata=linelist, lineformat="long", medium="vac")


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
