"""The purpose of this code is to update the gravity -primarily- after we change the variables in a Solve run.
How we do this depends on the parameters of metallicity, teff, and starting log_g. 
We also use the values rfom the reduction dictionry finally, especially as we use more for Lbol runs which we seem 
to be doing more than seis runs. """# coding: utf-8
# Sven's code.
# In[ ]:


import numpy as np
import os
import sys
from astropy.table import Table
import pickle
import pysme_age_mass_guess_parsec
# Is needed to load parsec isochrones.
from pysme_age_mass_guess_parsec import read_iso


# Changed d to dist. a_ks comes from a_k from galah sp3
def optimize_logg_with_extra_input(
    starting_parameters, reduction_variable_dict, field_end, glob_free=None,
    elli_dir=r'GALAH/ELLI',
    debug=False, show_todo=True):
    for x in reduction_variable_dict:
        print("X", x)
        if np.isnan(reduction_variable_dict[x]):
            reduction_variable_dict[x] = 0
        print(x, reduction_variable_dict[x], type(reduction_variable_dict[x]))
    """
    This function estimates the best LOGG from extra input (if provided).
    
    There are 4 ways to calculate LOGG:
    
    1) estimated it from the spectrum
        This is the case, when LOGG is part of glob_free.
    2) estimated from extra asteroseismic data (e.g. nu_max)
        This is the case, when '_seis' is part of the 'field_end' name
    3) estimated from extra data like astrometric and photometric data (e.g. k_mag == 2MASS K_S apparent magnitude)
        This is the case, when '_lbol' is part of the 'field_end' name.
        In this case, we estimate LOGG ~ log10(MASS) + 4*log10(TEFF) - log10(LBOL)
        The MASS we get from isochrone interpolation with a code called ELLI.
        TEFF is part of glob_free and thus coming from sme.teff
        LBOL we get from the astrometric+photometric information and an estimate of the bolometric correction (k_mag, 
                                                                                                    dist, A_KS, BC_KS)
        We have to update logg iteratively, because of the dependence of BC, MASS, and LOGG
    4) kept fixed 
        This is the case, when glob_free does not contain LOGG 
        and neither _lbol or _seis are part of the 'field_end' name.

    NEEDED INPUT:
    current_logg : the current logg value (will be given back in case 1 and 2)
    teff         : SME.teff
    fe_h         : SME.feh or an even better estimate of [Fe/H]
    field_end        : name of the field end, representing lbol or seis
    glob_free    : SME.glob_free array of the global parameters that are free (logg can be part of it as GRAV)
    elli_dir     : directory, where the ELLI code is found
    OPTIONAL INPUT:
    1) Needed if 'lbol' pipeline should be used
    k_mag         : 2MASS Ks band magnitude
    e_k_mag   : Uncertainty 2MASS Ks band magnitude
    dist            : Best distance estimate [pc]
    dist_lo         : Low percentile distance estimate [pc]
    dist_hi         : High percentile distance estimate [pc]
    a_ks         : A(Ks) extinction
    e_a_ks.      : Uncertainty of A(Ks)
    ebv          : color reddening E(B-V)
    2) Needed if 'seis' pipeline should be used
    nu_max       : float; frequency of strongest observed pulsation amplitude for a solar-like oscillator
    3) Debug
    debug        : True/False if printout should be generated

    OUTPUT:
    
    an updated value of logg following one of the 4 cases of pipeline setups

    """

    # setting the variables from the dict. Early here to be able to modify them when debugging.
    teff = starting_parameters['effective_temperature']
    current_logg = starting_parameters['gravity']
    fe_h = starting_parameters['metallicity']

    print("Using variables for log_g update:", reduction_variable_dict)
    if glob_free is None:
        glob_free = []

    """
    I prefered removng them as inputs.
    # Loop through arguments to replace any None values with ones found in reduction and analysis data. Doing it like
    # this and keeping them as inputs does allow for easier variable changes if desired.
    for function_argument in locals():
        print(function_argument)"""

    # Initialise reference values from the Sun. These are IAU standard values
    teff_sun = 5772.
    logg_sun = 4.438
    m_bol_sun = 4.7554
    nu_max_sun = 3090.0
    
    # We try to import the age/mass interpolation code here,
    # which is used later for the function mass_interpolation()
    #age_mass_guess_parsec = elli_dir+'age_mass_guess_parsec'
    #sys.path.insert(0, r'C:\Users\jama2357\Documents\Galafiles\GALAH\ELLI')
    try:
        # In addition, we load the Parsec isochrone data into the variable 'y'
        # Requires th read_iso class, that we currenty get from pysme_age_mass_guess_parsec.
        y = np.load(r'GALAH/ELLI/Parsec_isochrones.npy',allow_pickle=True,encoding='latin1')
        # and safe their [Fe/H] values into feh_iso
        feh_iso = [i.FeH for i in y]

    except ModuleNotFoundError:
        print('Could not find ELLI code at ', elli_dir)
        raise

    def logg_numax(teff, nu_max, teff_sun=teff_sun, nu_max_sun = nu_max_sun, logg_sun = logg_sun, debug=False):
        """
        Returns logg as esimated from teff and nu_max
        See e.g. Bedding et al. (2010): ApJ, 713, 176 
        for further information on the scaling relation
        of solar-like oscillators
        
        INPIUT:
        teff   : sme.teff
        nu_max : float value
        
        OUTPUT:
        logg
        
        """
        return(np.log10(nu_max/nu_max_sun*np.sqrt(teff/teff_sun)*10**logg_sun))
    
    def logg_parallax(mass, teff, l_bol, teff_sun=teff_sun, logg_sun = logg_sun, debug=False):
        """
        return logg from mass, teff, l_bol
        """
        return (np.log10(mass)+4.*np.log10(teff/teff_sun)-np.log10(l_bol)+logg_sun)
    

    def lbol_function(k_mag, e_k_mag, dist, dist_lo, dist_hi, a_ks, e_a_ks, bc_ks, e_bc_ks = 0.05, m_bol_sun = m_bol_sun, debug=False, show_todo=True):
        """
        This function computes the bolometric luminosity from the Ks magnitude
        
        INPUT:
        k_mag       : 2MASS magnitude in Ks band
        e_k_mag :
        d          : distance (to go from apparent to absolute)
        dist_lo
        dist_hi
        a_ks       : attenuation in Ks band
        e_a_ks     : 
        bc_ks      : bolometric correction (from absolute to bolometric))
        e_bc_ks    : uncertainty of bc_ks
        
        OUTPUT:
        lbol   : bolometric magnitude
        e_lbol : uncertainty of bolometric magnitude
        """
        
        # Best guess l_bol
        l_bol = 10.**(-0.4*(k_mag - 5.*np.log10(dist/10.) + bc_ks - a_ks - m_bol_sun))
        
        # Estimating worst case (all values 1 sigma in increasing/decreasing direction respectively)
        l_bol_lo = 10.**(-0.4*(k_mag+e_k_mag - 5.*np.log10(dist_lo/10.) + (bc_ks + e_bc_ks) - np.max([0,(a_ks - e_a_ks)]) - m_bol_sun))
        l_bol_hi = 10.**(-0.4*(k_mag-e_k_mag - 5.*np.log10(dist_hi/10.) + (bc_ks - e_bc_ks) - (a_ks + e_a_ks) - m_bol_sun))

        # Estimating average of deviations from best guess
        e_l_bol = 0.5*(l_bol_hi - l_bol_lo)

        if debug:
            print('////////')
            print('L_BOL estimation:')
            print('Best guess l_bol, average deviation from best guess, lowest 1sigma l_bol, highest 1sigma l_bol')
            print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(l_bol, e_l_bol, l_bol_lo, l_bol_hi))
            print('////////')
        if show_todo:
            print('ToDo: Implement MonteCarlo Sampling of uncertainties?')
            
        return(l_bol, e_l_bol)
    
    def mass_interpolation(teff, current_logg, fe_h, l_bol, e_lbol, e_teff=100, e_logg=0.5, e_fe_h=0.2, debug=False, show_todo=False):
        """
        This function estimates the mass from the given input values by calling the ELLI code
        
        Because we are fitting the parameters, we do not know their uncertainties.
        The routine needs large enough uncertainties to explore enough isochrone points.
        We are thus typically using typical uncertainties (see below)
        
        INPUT:
        teff         : sme.teff
        current_logg : current best logg estimate (will be updated iteratively)
        fe_h         : sme.fe_h
        l_bol        : bolometric luminosity
        e_lbol       : uncertainty of l_bol, with uncertainties
        e_teff       : 100 K
        e_logg       : 0.5 dex (to give it less weight as the l_bol)
        e_fe_h       : 0.2 dex (to have at least 2 isochrone [Fe/H] points)
        
        debug        : True/False
        show_todo    : True/False
        
        OUTPUT:
        mass : either isochrone-interpolated mass (to be written) or keep_mass
        """
        age, mass = pysme_age_mass_guess_parsec.do_age_mass_guess(
            np.array([teff, current_logg, l_bol, fe_h+0.1]), 
            np.array([e_teff, e_logg, e_lbol, e_fe_h]), 
            y,feh_iso
            )
        if debug:
            print('////////')
            print('MASS estimation:')
            print('teff, current_logg, fe_h+0.1, l_bol, e_lbol, e_teff,e_logg,e_fe_h')
            print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(teff, current_logg, fe_h+0.1, l_bol, e_lbol, e_teff,e_logg,e_fe_h))
            print('NB: Currently setting fe_h to fe_h+0.1 because of a known bias in the pipeline, underestimating [Fe/H] by ~0.1 dex')
            print('Mass: ', mass)
            print('Age: ', age)
            print('////////')

        return (mass)

    def bc_interpolation(teff, current_logg, fe_h, ebv, filter = 'K', debug=False, show_todo=False):
        """
        This function interpolates the BC(filter) from Casagrande+14

        The tables from Casagrande+14 usually come for all different values of E(B-V).
        We have already selected values of E(B-V) = [0.00, 0.12, 0.24, 0.36, 0.48] and first select the appropriate one
        For this, we have a FITS file with precomputed BC values for several filters.

        We do a tri-linear interpolation with teff, logg, and fe_h
        and return the values of BC in a dictionary

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
        bc           : dictionary with interpolated bc from Casagrande+14 with teff, init_logg, fe_h as input

        """

        # First find which of the precomputed ebv grids is closest to the actual ebv and load in the associated BC Table
        ebv_grid = np.array(['00','12','24','36','48'])
        ebv_grid_difference = np.array([abs(int(ebv_grid_entry)/100.-ebv) for ebv_grid_entry in ebv_grid])
        closest_ebv = np.where(np.min(ebv_grid_difference)==ebv_grid_difference)[0][0]
        BCTable = Table.read('GALAH/DATA/Casagrande2014_BC_EBV_'+ebv_grid[closest_ebv]+'.fits',1)

        if show_todo:
            print('There are better ways to implement the EBV... In the best case, one could just adjust Lucas fortran interpolation routine to work in python...')
        
        # Now prepare 

        # 1) Select appropriate Teff 
        tg = np.unique(BCTable['teff'])
        tdiff = abs(tg-teff)
        tmin = np.where(tdiff == min(tdiff))[0]
        tmin = tmin[0]
        t1 = tg[tmin]
        if min(tg-teff) >= 0.0:
            # too low -> choose lowest Teff                                                                                                                                                                                        
            tfrac=0.0
            t2 = t1
        elif max(tg-teff) <= 0.0:
            # too high -> choose highest Teff                                                                                                                                                                                   
            tfrac=0.0
            t2 = t1
        else:
            # get ratio of two entries with smallest difference                                                                                                                                                                 
            tsort=np.sort(tdiff)
            t2min = np.where(tsort[1] == tdiff); t2min = t2min[-1]
            t2 = tg[t2min]; t2=t2[0]
            tfrac = abs((teff - t1)/(t2-t1))

        # 2) Select appropriate [Fe/H]
        has_correct_teff = np.where((BCTable['teff'] == t1) | (BCTable['teff'] == t2))
        fg    =  np.unique(BCTable['feh'][has_correct_teff])
        fdiff = abs(fg-fe_h)
        fmin  = np.where(fdiff == min(fdiff)); fmin = fmin[0]
        f1 = fg[fmin]; f1=f1[0]
        if min(fg-fe_h) >= 0.0:
            # too low -> choose lowest [Fe/H]                                                                                                                                                                                      
            ffrac=0.0
            f2 = f1
        elif max(fg-fe_h) <= 0.0:
            # too high -> choose highest [Fe/H]                                                                                                                                                                                 
            ffrac=0.0
            f2 = f1
        else:
            # get ratio of two entries with smallest difference                                                                                                                                                                 
            fsort=np.sort(fdiff)
            f2min = np.where(fsort[1]== fdiff); f2min = f2min[-1]
            f2 = fg[f2min]; f2=f2[0]
            ffrac = abs((fe_h - f1)/(f2-f1))

        # 1) Select appropriate logg
        has_correct_teff_and_feh = np.where(
            ((BCTable['teff'] == t1) & (BCTable['feh']==f1)) | 
            ((BCTable['teff'] == t1) & (BCTable['feh']==f2)) | 
            ((BCTable['teff'] == t2) & (BCTable['feh']==f1)) | 
            ((BCTable['teff'] == t2) & (BCTable['feh']==f2))
            )
        gg  =  np.unique(BCTable['logg'][has_correct_teff_and_feh])
        # Test wether requested log(g) is in grid                                                                                                                                                                                 
        gdiff = abs(gg-current_logg)
        gmin  = np.where(gdiff == min(gdiff)); gmin = gmin[0]
        g1 = gg[gmin]; g1=g1[0]
        if min(gg-current_logg) >= 0.0:
            # too low -> choose lowest log(g)                                                                                                                                                                                      
            gfrac=0.0
            g2 = g1
        elif max(gg-current_logg) <= 0.0:
            # too high -> choose highest log(g)                                                                                                                                                                                 
            gfrac=0.0
            g2 = g1
        else:
            # get ratio of two entries with smallest difference                                                                                                                                                                 
            gsort=np.sort(gdiff)
            g2min = np.where(gsort[1] == gdiff); g2min = g2min[-1]
            g2 = gg[g2min]; g2=g2[0]
            gfrac = abs((current_logg - g1)/(g2-g1))

        # Now get the BC values for the 8 points with either teff, logg, and fe_h value
        df111 = BCTable['BC_'+filter][(((BCTable['teff'] == t1) & (BCTable['feh'] == f1) & (BCTable['logg'] == g1)))]
        df112 = BCTable['BC_'+filter][(((BCTable['teff'] == t1) & (BCTable['feh'] == f1) & (BCTable['logg'] == g2)))]
        df121 = BCTable['BC_'+filter][(((BCTable['teff'] == t1) & (BCTable['feh'] == f2) & (BCTable['logg'] == g1)))]
        df211 = BCTable['BC_'+filter][(((BCTable['teff'] == t2) & (BCTable['feh'] == f1) & (BCTable['logg'] == g1)))]
        df122 = BCTable['BC_'+filter][(((BCTable['teff'] == t1) & (BCTable['feh'] == f2) & (BCTable['logg'] == g2)))]
        df212 = BCTable['BC_'+filter][(((BCTable['teff'] == t2) & (BCTable['feh'] == f1) & (BCTable['logg'] == g2)))]
        df221 = BCTable['BC_'+filter][(((BCTable['teff'] == t2) & (BCTable['feh'] == f2) & (BCTable['logg'] == g1)))]
        df222 = BCTable['BC_'+filter][(((BCTable['teff'] == t2) & (BCTable['feh'] == f2) & (BCTable['logg'] == g2)))]

        # Combine them, weighted by the fractions, first the logg difference, then the teff difference, then the feh difference
        d11=(1-gfrac)*df111+gfrac*df112
        d12=(1-gfrac)*df121+gfrac*df122
        d21=(1-gfrac)*df211+gfrac*df212
        d22=(1-gfrac)*df221+gfrac*df222

        d1=(1-ffrac)*d11+ffrac*d12
        d2=(1-ffrac)*d21+ffrac*d22

        bc = (1-tfrac)*d1+tfrac*d2
        
        if debug:
            print('////////')
            print('BC estimation: Teff/logg/feh values, weights estimated from the closest grid points')
            print(teff, tfrac,t1,t2)
            print(fe_h, ffrac,f1,f2)
            print(current_logg, gfrac,g1,g2)
            print('BC: ',float(bc))
            print('////////')

        return(float(bc))
                                          
    def update_logg_parallax(teff, current_logg, fe_h, k_mag, e_k_mag, dist, dist_lo, dist_hi, a_ks, e_a_ks, ebv, debug=debug):
        """
        Updates logg as a function off the input labels via the physical relation
        
        Because of the dependence of the functions,
        we first update BC, then l_bol, then mass

        """


        # Calculate bolometric correction BC for the given parameters
        bc_ks = bc_interpolation(teff, current_logg, fe_h, ebv, debug=debug)
        print(1, bc_ks)

        # Estimate bolometric luminiosity l_bol
        l_bol, e_lbol = lbol_function(k_mag, e_k_mag, dist, dist_lo, dist_hi, a_ks, e_a_ks, bc_ks, debug=debug, show_todo=show_todo)
        print(2, l_bol, e_lbol)

        # Estimate mass via ELLI's isochrone interpolation
        mass  = mass_interpolation(teff, current_logg, fe_h, l_bol, e_lbol, debug=debug, show_todo=show_todo)
        print(3, mass)
        #exit()
        logg = (logg_parallax(mass, teff, l_bol, debug=debug))
        print(4, logg)
        #exit()

        # Use mass, teff, and l_bol to calculate logg
        return logg

    """
    Check which of the 4 cases is actually true and run the function accordingly    
    """

    # Here we check which of the 4 setups is actually used
    # if logg is fitted from the spectrum, we will not update it as part of this routine
    if 'GRAV' in glob_free:
        # Case 1
        if debug:
            print('LOGG/GRAV is a free parameter, returning input value')

        return current_logg
    else:
        if field_end == 'seis':
            # Case 2

            if debug:
                print('Seis pipeline activated')
                
            # check if input value can be used
            if isinstance(reduction_variable_dict['nu_max'], float):
                if reduction_variable_dict['nu_max']>=0.:
                    return(logg_numax(teff, reduction_variable_dict['nu_max'], debug=debug))
                else:
                    raise ValueError("nu_max has to be positive and finite")
            else:
                raise ValueError("nu_max has to be float")

        elif field_end == 'lbol':

            # Case 3
            if debug:
                print('Lbol pipeline activated')

            input_values = np.array([reduction_variable_dict['k_mag'], reduction_variable_dict['e_k_mag'], reduction_variable_dict['dist'],
                                     reduction_variable_dict['dist_lo'], reduction_variable_dict['dist_hi'], reduction_variable_dict['a_k'],
                                     reduction_variable_dict['e_a_k'], reduction_variable_dict['ebv']])
            input_names  = np.array(['k_mag', 'e_k_mag', 'd', 'dist_lo', 'dist_hi', 'a_ks', 'e_a_k', 'ebv'])

            check_if_float = np.where(np.array([isinstance(param, float) for param in input_values])==False)[0]
            if len(check_if_float) != 0:

                raise ValueError("Has to be float: "+", ".join(input_names[check_if_float]))
            else:
                check_if_finite = np.where(np.array([np.isfinite(param) for param in input_values])==False)[0]
                if len(check_if_finite)!=0:
                    raise ValueError("Has to be finite: "+", ".join(input_names[check_if_finite]))
                else:
                    # We will have to iteratively update the logg value until it is consistent
                    # We have to do this iteratively, because of the dependence of BC, MASS, and LOGG
                    # So we start with the current value and update it,
                    # until update of logg does not change significantly anymore, i.e. less than 0.001 dex
                    updated_logg = update_logg_parallax(teff, current_logg, fe_h, reduction_variable_dict['k_mag'],
                                                        reduction_variable_dict['e_k_mag'], reduction_variable_dict['dist'],
                                                        reduction_variable_dict['dist_lo'], reduction_variable_dict['dist_hi'],
                                                        reduction_variable_dict['a_k'], reduction_variable_dict['e_a_k'],
                                                        reduction_variable_dict['ebv'], debug=debug)
                    print(10, updated_logg)
                    #exit()
                    if debug:
                        it = 1
                    while np.abs(updated_logg - current_logg) > 0.001:

                        if debug:
                            print('Iteration: ', it, current_logg, updated_logg)
                            it+=1
                        current_logg = updated_logg

                        updated_logg = update_logg_parallax(teff, current_logg, fe_h, reduction_variable_dict['k_mag'],
                                                            reduction_variable_dict['e_k_mag'], reduction_variable_dict['dist'],
                                                            reduction_variable_dict['dist_lo'], reduction_variable_dict['dist_hi'],
                                                            reduction_variable_dict['a_k'], reduction_variable_dict['e_a_k'],
                                                            reduction_variable_dict['ebv'], debug=debug)


                    return(updated_logg)
        else:
            # Case 4
            if debug:
                print('LOGG/GRAV fixed, return input value')


            return(current_logg)

# In[ ]:


def debug_run():
    current_logg = 4.438
    teff = 5772.
    fe_h = 0.00
    k_mag = None
    e_k_mag = None
    dist = None
    dist_lo = None
    dist_hi = None
    a_ks = None
    e_a_ks = None
    ebv = None
    nu_max = None

    # Testing case 1:
    # field_end = 'test'
    # glob_free = ['TEFF','GRAV','FEH','VMIC','VSINI','VRAD']

    # Testing case 2:
    field_end = 'test_lbol'
    k_mag = 3.28
    e_k_mag = 0.01
    dist = 10.0
    dist_lo = 9.9
    dist_hi = 10.1
    a_ks = 0.00
    e_a_ks = 0.03
    ebv = 0.00
    glob_free = ['TEFF','FEH','VMIC','VSINI','VRAD']

    # Testing case 3:
    # field_end = 'test_seis'
    # nu_max = 3090.
    # glob_free = ['TEFF','FEH','VMIC','VSINI','VRAD']

    # Testing case 4:
    # field_end = 'test'
    # glob_free = ['TEFF','FEH','VMIC','VSINI','VRAD']

    optimize_logg_with_extra_input(
        current_logg, teff, fe_h, field_end, glob_free,
        k_mag=k_mag, e_k_mag=e_k_mag, dist=dist, dist_lo=dist_lo, dist_hi=dist_hi, a_ks=a_ks, e_a_ks=e_a_ks, ebv=ebv,
        nu_max=nu_max,
        debug=True, show_todo=False)


    used_vars = ['k_mag', 'e_k_mag', 'dist', 'dist_lo', 'dist_hi', 'a_k', 'e_a_k', 'ebv', 'nu_max']
    # old vars were ['k_mag', 'e_k_mag', 'd', 'd_lo', 'd_hi', 'a_ks', 'e_a_ks', 'ebv', 'nu_max']

