""" Runs pysme either as solve or syntehsize, then output the results in an OUTPUT folder. Synthesize changes the
spectra, and solve varies the parameters.
"""

import tqdm


# Removes the progress bar. Mut be before imports.
class _TQDM(tqdm.tqdm):
    def __init__(self, *argv, **kwargs):
        kwargs['disable'] = True
        if kwargs.get('disable_override', 'def') != 'def':
            kwargs['disable'] = kwargs['disable_override']
        super().__init__(*argv, **kwargs)

    def _time(x):
        pass


tqdm.tqdm = _TQDM
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, r"/media/sf_SME-master/src")
from pysme.gui import plot_plotly
from pysme import sme as SME
from pysme import util
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum
from pysme.abund import Abund
from pysme.linelist.linelist import LineList
import pickle
import numpy as np


# Only function in the file, runs sme after setting up its input parameters from our input.
#               Currently supported file formats:
#             * ".npy": Numpy save file of an SME_Struct
#             * ".sav", ".inp", ".out": IDL save file with an sme structure
#             * ".ech": Echelle file from (Py)REDUCE
# So we cannot load with SME_structure.load(), we must instead set each parameter individually.
def start_sme(temp_sme, linelist, run):
    """
        Input:
            temp_sme: Makestruct_dict in the other files, this contains all the information required to run Pysme.
                        It is given in dictionary form so must be set manually.
            linelist: Pysme requires the linelist both in the regular naming convention and also a seperate class
            run:      Used to determine whether we run Synth or Solve
        Output:
            None, although we save the result as seperate files in the OUTPUT folder.
    """
    obs_name = temp_sme['obs_name']

    # Load your existing SME structure or create your own. We create our own.
    sme = SME.SME_Structure()
    pickle.dump(temp_sme, open("Latest input to sme.pkl", "wb"))

    # Set all sme class properties as the names of the input dict, skipping those that don't exist or
    # are set later.
    for key in temp_sme:
        print(key)
        try:
            # We set those in deeper class properties
            if key != 'abund' and key != 'atmo' and key != 'maxiter':
                setattr(sme, key, temp_sme[key])
        except AttributeError:
            print(AttributeError, key)

    # We set abundance out of the loop due to SME class differences
    sme.abund = Abund(sme.monh, temp_sme['abundances'])
    sme.atmo.source = temp_sme["atmo"]["source"]
    sme.atmo.method = temp_sme["atmo"]["method"]
    sme.atmo.depth = temp_sme["atmo"]["depth"]
    sme.atmo.interp = temp_sme["atmo"]["interp"]
    sme.atmo.geom = temp_sme["atmo"]["geom"]


    print("temp sme cscale type is", temp_sme['cscale_type'])
    print("temp sme cscale flag is", temp_sme['cscale_flag'])

    sme.nlte.set_nlte("H", temp_sme['nlte_abund'][0])
    sme.nlte.set_nlte("Fe", temp_sme['nlte_abund'][1])

    # either read the file created by idl_conversion or load the input linelist from makestruct.
    # long is nlte, short is lte
    sme.linelist = LineList(linedata=linelist, lineformat="long", medium="vac")

    sme.fitresults.maxiter = temp_sme['maxiter']
    sme.fitparameters = temp_sme["fitparameters"]

    print("\n\n\nmetallicity and grav and teff BEFORE sme", sme.monh, sme.logg, sme.teff, "\n\n\n")
    sme.h2broad = True
    sme.specific_intensities_only = False
    sme.normalize_by_continuum = True

    # Start the logging to the file

    # Start SME solver
    print("Running in ", run, "mode")
    sme.save("smeinput", sme)
    pickle.dump(sme, open("Starting sme.pkl", "wb"))
    util.start_logging(temp_sme['obs_name']+'log')

    if run == "Synthesise":
        print("Starting sme_synth")
        sme = synthesize_spectrum(sme)
    elif run == "Solve":
        print("Starting sme_solve")
        # Fitparameters (globfree) come fom the sme structure.
        sme = solve(sme, param_names=sme.fitparameters)
    else:
        print("Run is neither Synthesise or Solve, that's a big ERROR. Exiting.")
        exit()


    # Applies the fix that we perform on synth to spec, the observed spectra.
    for seg in range(len(sme.wave)):
        x = sme.wave[seg] - sme.wave[seg][0]
        cont = np.polyval(sme.cscale[seg], x)
        sme.spec[seg] = sme.spec[seg] / cont

    print(sme.citation())
    print("Cscale after a run is:", sme.cscale)
    print("Sme accwi and accrt:", sme.accwi, sme.accrt)
    print("AFter a", run," run, sme vsini, nmu, mu is ", sme.vsini, sme.nmu, sme.mu)
    print("len of synth right after this run (", temp_sme['balmer_run'], "is balmer?", len(sme.synth))
    print("Finished current sme loop, starting afresh.")
    print("VRAD AFTER SME IS", sme.vrad)
    print("sme ipres is:", sme.ipres)
    print("Fit parameters are still", sme.fitparameters)
    print("\n\n\nmetallicity and grav and teff AFTER sme", sme.monh, sme.logg, sme.teff, "\n\n\n")
    # We create dicts to save out as mentioned below. Only some are saved, the ones that are unused and set
    # independently each run are not bothered with.
    # Only happens when not running a balmer line run with a diff fit file.
    print("After, fitresults are", sme.fitresults)
    print("and maxiter is ", sme.fitresults.maxiter)
    print("iptype is ", sme.iptype)

    spectra_dict = {"wave": sme.wave,   'flux': sme.spec,
                    "error": sme.uncs,  'mask': sme.mask,
                    "synth": sme.synth}

    variables_dict = {"effective_temperature": sme.teff,    "gravity": sme.logg,
                      "metallicity": sme.monh,               "radial_velocity_global": sme.vrad,
                      "abundances": sme.abund,              "microturbulence_velocity": sme.vmic,
                      "macroturbulence_velocity": sme.vmac, "rotational_velocity": sme.vsini,
                      'field_end': temp_sme['field_end'],   "load_file": True,
                      'segment_begin_end': sme.wran,        "ipres": sme.ipres,
                      'vsini': sme.vsini,                   "vmac": sme.vmac,
                      }

    print("Balmer run is", temp_sme['balmer_run'])
    if not temp_sme['balmer_run']:
        # Convert to pickle to open elsewhere, because opening SME classes outside of linux is v hard.
        # This could be changed once all code is on the same platform perhaps. But I like using non-niche stuff.
        # OVerwrites the previous sme input, which in turn is recreated and overwrites this output.
        # Save results
        pickle.dump(variables_dict, open(r"OUTPUT/VARIABLES/" + obs_name + "_SME_variables.pkl", "wb"))
        pickle.dump(spectra_dict, open(r"OUTPUT/SPECTRA/" + obs_name + "_SME_spectra.pkl", "wb"))

        print("Saved SME output in OUTPUT/SPECTRA and OUTPUT/VARIABLES")
        # For testing in one big file, we currently don't use it.
        np.save(r"OUTPUT/FullOutput/SME_Output", sme)

    # IF it is a balmer run we don't ovewrite our good data, and we don't update logg.
    elif temp_sme['balmer_run']:
        print("Finishing balmer run, saving separate files.")

        pickle.dump(variables_dict, open(r"OUTPUT/VARIABLES/" + obs_name + "_Balmer_variables.pkl", "wb"))
        pickle.dump(spectra_dict, open(r"OUTPUT/SPECTRA/"+obs_name+"_Balmer_spectra.pkl", "wb"))
        a = pickle.load(open(r"OUTPUT/SPECTRA/" + obs_name + "_Balmer_spectra.pkl", "rb"))

        np.save(r"OUTPUT/FullOutput/SME_Balmer_Output", sme)
