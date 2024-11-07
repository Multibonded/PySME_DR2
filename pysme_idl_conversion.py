import numpy as np
import pickle
import pandas as pd
"""Converting the linelist from our IDL version to the PySME version it requires. Basically just some name changes"""

# Currently unused as it makes no difference.
def parse_line_error(error_flags, values):

    """Transform Line Error flags into relative error values

    Parameters
    ----------
    error_flags : list(str)
        Error Flags as defined by various references
    values : float
        depths of the lines, to convert absolute to relative errors

    Returns
    -------
    errors : list(float)
        relative errors for each line
    """
    nist = {
        "AAA": 0.003,
        "AA": 0.01,
        "A+": 0.02,
        "A": 0.03,
        "B+": 0.07,
        "B": 0.1,
        "C+": 0.18,
        "C": 0.25,
        "C-": 0.3,
        "D+": 0.4,
        "D": 0.5,
        "D-": 0.6,
        "E": 0.7,
    }
    error = np.ones(len(error_flags), dtype=float)
    for i, (flag, _) in enumerate(zip(error_flags, values)):
        print(i, flag, _)
        if flag[0] in [" ", "_", "P"]:
            # undefined or predicted
            error[i] = 0.5
        elif flag[0] == "E":
            # absolute error in dex
            # TODO absolute?
            error[i] = 10 ** float(flag[1:])
        elif flag[0] == "C":
            # Cancellation Factor, i.e. relative error
            error[i] = abs(float(flag[1:]))
        elif flag[0] == "N":
            # NIST quality class
            flag = flag[1:5].strip()
            error[i] = nist[flag]
    return error

def dataframe_the_linelist(linelist):

    # If there is only one line, it is 1D in the IDL structure, but we expect 2D

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
        #,"atomic": linelist['atomic'] # Because apparently we need an atomic part to the linelsit, despite not being specified.
    }

    # we're always long format
    lineformat = "long"
    # PySME doesn't import lineref as an array, and so we do something to combine all values in the array each line to allow us to modify it in the same way
    # Pysme tries to do. I think. Unsure about this as it's all blank in my example code/linref.
    linerefjoined = np.array(([''.join(array) for array in linelist['lineref']]))

    # PySME strips it, but we have a big ol array of spaces so we lose all the data and end with empty string!
    # Not to mention, parse line error has a section for if the value IS a space! That is impossible with strip.
    """error = [s[0:11].strip() for s in linerefjoined]"""
    # so instead we're going to remove "strip" and hope for the best. This is where an error may be thrown if we're wrong.
    # Bsically it's gonna change the error and assume there's no value inside (giving a default of 0.5). We can
    # correct this potential issue by adding some kind of check to say "If there are characters in here, strip. If not, don't".
    # #################################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Furthermore, it attempts to turn flag into a float (from 1:), but it's a string in what we have! I believe it may be
    # meaning to take the VALUE from depth( which does exist and is called on in the same loop) as the new error value.
    # furthermore, why are several letters (such as K07) ignored and given the default error value?
    # I have replaced the float flag[1:] as _ (the depth value)
    # Even ones like NZL don't fit how PYSME calls them. This is very very strange. I'm setting them all to the default 0.5. for now
    """
    Hi Jack,
    The errors in the VALD linelist are given as strings in various formats. (e.g. “N D”, “E 0.05”). Parse_line_error takes those and puts them in the same format, I.e. relative errors.
    I have no idea what your flags represent, but I would guess that they are not the error, but the source of the data. See https://www.astro.uu.se/valdwiki/VALD3linelists for what they could mean.
     
    Now at the moment at least, the relative errors on the line data, do not influence the calculations in any way. So if you can’t find them in your data, it won’t be a problem.
     
    Regards
    Ansgar
    """
    error = [s[0:11] for s in linerefjoined] #.strip removed
    #error = parse_line_error(error, linelist['depth'])
    error = np.ones(len(error), dtype=float)
    error.fill(0.5)
    data["error"] = error
    data["lande_lower"] = linelist['line_lulande'][:, 0]
    data["lande_upper"] = linelist['line_lulande'][:, 1]
    data["j_lo"] = linelist['line_extra'][:, 0]
    data["e_upp"] = linelist['line_extra'][:, 1]
    data["j_up"] = linelist['line_extra'][:, 2]
    data["term_lower"] = [t[10:].strip() for t in linelist['line_term_low']]
    data["term_upper"] = [t[10:].strip() for t in linelist['line_term_upp']]

    # We have an issue with lineref being 2d. needs to be 1 d I believe. But it contains all 7 arrays from IDL?
    # So instead I just input the previously mentioned linerefjoined to conglomorate all the arrays into one row per row.
    data['reference'] = linerefjoined
    linedata = pd.DataFrame.from_dict(data)
    linedata.to_pickle("OUTPUT/LINELIST/pysme_linelist_dataframe")
    return linedata