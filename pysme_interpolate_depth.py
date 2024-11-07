"""This files contain a few seperate functions to etrapolate depth information from GALAH files by findining one
with the closest parameters to ours. It takes into account the fractions of the two closest files each and uses that
to interpolate a more accurate result."""
import numpy as np
import pandas as pd
import glob
import pickle as pk
from astropy.table import Table
from astropy.io import fits


# Searches the depth files in GALAH to find one that has the correct teff, metallicity, and logg. These are represented
# by the file names themselves.
def reduce_depth(makestruct_dict):
    """
    Input:
        makestruct_dict: Parameter information
    Output:
        indexed_depth: Correct depth for a star with the given parameters (Teff, Feh, Logg)
    """
    # The header of the depth files before the teff, etc, info.
    depth_header = "ges_master_v5_"
    depth_extention = '.fits'
    depth_location = r"GALAH/DEPTH/"
    # Gets the file names in the directory
    files = (glob.glob(depth_location + depth_header + "*.fits"))
    # Splits the file names up to incldue only teff etc info and extention, relies on the files all being named the same
    # other than the information we are looking at.
    files = [file_name.split(depth_header)[-1] for file_name in files]
    # removes extention of the file (depth extention)
    files = [file_name.split(depth_extention)[0] for file_name in files]
    # Splits up the variables individually to teff, logg, and metallicity.
    variables = [file_name.split("_") for file_name in files]
    files = np.asarray(files)
    # We take the temp, grav, and metallicity out from the file names and put them into arrays each
    teff = np.asarray([teff[0] for teff in variables]).astype(float)
    logg = np.asarray([logg[1] for logg in variables]).astype(float)
    fe_h = np.asarray([fe_h[2] for fe_h in variables]).astype(float)
    # Get t1, t2, tfrac of temp metallicity and grav, essentially the closest values we could find in the file names.
    temperature_dict, metal_dict, grav_dict = collect_file_info(makestruct_dict, teff, logg, fe_h)
    # We now open those files to take their depth information.
    indexed_depth = open_files(teff, fe_h, logg, temperature_dict, metal_dict,
                               grav_dict, depth_header, depth_extention, depth_location, files, makestruct_dict)
    # The depth is cut down to 1/3 of its size as we only need some hard coded index values of the depth.
    # The code we have been unning was to find the right depth FILE.
    return indexed_depth


# this is a copy of bc_interpolation otuside of the main function to be able to call it without the other inputs
# to adjust linefull.depth as it says in idl.
def collect_file_info(makestruct_dict, teff, logg, fe_h):
    """
    This function interpolates the depth(filter) from Casagrande+14

    The tables from Casagrande+14 usually come for all different values of E(B-V).
    We have already selected values of E(B-V) = [0.00, 0.12, 0.24, 0.36, 0.48] and first select the appropriate one
    For this, we have a FITS file with precomputed depth values for several filters.

    We do a tri-linear interpolation with teff, logg, and fe_h
    and return the values of depth in a dictionary

    For this, we find the 2 closest entries in teff, then fe_h, and then logg
    with this set of 8 points, we can perform a trilinear interpolation

    INPUT:
    teff         : SME.teff
    current_logg : the current logg value (will be given back in case 1 and 2)
    fe_h         : SME.feh or an even better estimate of [Fe/H]
    ebv          : color reddening E(B-V)
    filter       : filter for which the bolometric correction will be applied
                   Options: V, B, J, H, K

    OUTPUT:
    depth           : dictionary with interpolated depth from Casagrande+14 with teff, init_logg, fe_h as input

    """
    # 1) Select appropriate Teff
    tg = np.unique(teff)
    temperature_dict = {'t1': find_variable(makestruct_dict, tg, 'effective_temperature')[0],
                        't2': find_variable(makestruct_dict, tg, 'effective_temperature')[1],
                        'tfrac': find_variable(makestruct_dict, tg, 'effective_temperature')[2]}

    # 2) Select appropriate [Fe/H]
    has_correct_teff = np.where((teff == temperature_dict['t1']) | (teff == temperature_dict['t2']))
    fg = np.unique(fe_h[has_correct_teff])
    metal_dict = {'f1': find_variable(makestruct_dict, fg, 'metallicity')[0],
                  'f2': find_variable(makestruct_dict, fg, 'metallicity')[1],
                  'ffrac': find_variable(makestruct_dict, fg, 'metallicity')[2]}

    # 3) Select appropriate logg
    has_correct_teff_and_feh = np.where(
        ((teff == temperature_dict['t1']) & (fe_h == metal_dict['f1'])) |
        ((teff == temperature_dict['t1']) & (fe_h == metal_dict['f2'])) |
        ((teff == temperature_dict['t2']) & (fe_h == metal_dict['f1'])) |
        ((teff == temperature_dict['t2']) & (fe_h == metal_dict['f2']))
    )
    gg = np.unique(logg[has_correct_teff_and_feh])
    grav_dict = {'g1': find_variable(makestruct_dict, gg, 'gravity')[0],
                 'g2': find_variable(makestruct_dict, gg, 'gravity')[1],
                 'gfrac': find_variable(makestruct_dict, gg, 'gravity')[2]}

    return temperature_dict, metal_dict, grav_dict


# We have the parameter information, so we open the files that have the same values as we want.
def open_files(teff, fe_h, logg, temperature_dict, metal_dict, grav_dict,
               depth_header, depth_extention, depth_location, files, makestruct_dict):
    t1, t2, tfrac = temperature_dict['t1'], temperature_dict['t2'], temperature_dict['tfrac']
    g1, g2, gfrac = grav_dict['g1'], grav_dict['g2'], grav_dict['gfrac']
    f1, f2, ffrac = metal_dict['f1'], metal_dict['f2'], metal_dict['ffrac']
    # Now get  the files for the 8 points with either teff, logg, and fe_h value and then the depth values
    df111 = files[np.where(((teff == t1) & (fe_h == f1) & (logg == g1)))]
    df112 = files[np.where(((teff == t1) & (fe_h == f1) & (logg == g2)))]
    df121 = files[np.where(((teff == t1) & (fe_h == f2) & (logg == g1)))]
    df122 = files[np.where(((teff == t1) & (fe_h == f2) & (logg == g2)))]
    df211 = files[np.where(((teff == t2) & (fe_h == f1) & (logg == g1)))]
    df212 = files[np.where(((teff == t2) & (fe_h == f1) & (logg == g2)))]
    df221 = files[np.where(((teff == t2) & (fe_h == f2) & (logg == g1)))]
    df222 = files[np.where(((teff == t2) & (fe_h == f2) & (logg == g2)))]

    # Get the full file name again using these variables.
    full_file_111 = fits.open(depth_location + depth_header + str(df111[0]) + depth_extention)[0].data
    full_file_112 = fits.open(depth_location + depth_header + str(df112[0]) + depth_extention)[0].data
    full_file_121 = fits.open(depth_location + depth_header + str(df121[0]) + depth_extention)[0].data
    full_file_122 = fits.open(depth_location + depth_header + str(df122[0]) + depth_extention)[0].data
    full_file_211 = fits.open(depth_location + depth_header + str(df211[0]) + depth_extention)[0].data
    full_file_212 = fits.open(depth_location + depth_header + str(df212[0]) + depth_extention)[0].data
    full_file_221 = fits.open(depth_location + depth_header + str(df221[0]) + depth_extention)[0].data
    full_file_222 = fits.open(depth_location + depth_header + str(df222[0]) + depth_extention)[0].data

    # Combine them, weighted by the fractions, first the logg difference, then the teff difference,
    # then the feh difference
    d11 = (1 - gfrac) * full_file_111 + gfrac * full_file_112
    d12 = (1 - gfrac) * full_file_121 + gfrac * full_file_122
    d21 = (1 - gfrac) * full_file_211 + gfrac * full_file_212
    d22 = (1 - gfrac) * full_file_221 + gfrac * full_file_222

    d1 = (1 - ffrac) * d11 + ffrac * d12
    d2 = (1 - ffrac) * d21 + ffrac * d22

    trilinear_depth = (1 - tfrac) * d1 + tfrac * d2
    # Hard coded values of the ones we care aboiut for sme. End numbers increased by 1 from idl due to python
    # not including stopping values in the slice.
    if "ges_master_v5.1" in makestruct_dict['line_list']:
        indexed_depth = trilinear_depth
    else:
        indexed_depth = np.concatenate((trilinear_depth[100471:162357], trilinear_depth[318432:435590],
                                    trilinear_depth[652638:748076], trilinear_depth[851743:884263]))
    return indexed_depth


def find_variable(makestruct_dict, input_variable, var_type):
    gg = input_variable
    # Test whether requested log(g) is in grid
    gdiff = abs(gg - makestruct_dict[var_type])
    gmin = np.where(gdiff == min(gdiff))
    gmin = gmin[0]
    g1 = gg[gmin]
    g1 = g1[0]
    if min(gg - makestruct_dict[var_type]) >= 0.0:
        # too low -> choose lowest log(g)
        gfrac = 0.0
        g2 = g1
    elif max(gg - makestruct_dict[var_type]) <= 0.0:
        # too high -> choose highest log(g)
        gfrac = 0.0
        g2 = g1
    else:
        # get ratio of two entries with smallest difference
        gsort = np.sort(gdiff)
        g2min = np.where(gsort[1] == gdiff)
        g2min = g2min[-1]
        g2 = gg[g2min]
        g2 = g2[0]
        gfrac = abs((makestruct_dict[var_type] - g1) / (g2 - g1))

    return g1, g2, gfrac

