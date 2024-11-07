"""This file is used to read the linelist and modify it according to the requirements for PySME, such as condensing the
species name from a list to a single object, and adding ionization to it. Because this is a huge file and we do the same
thing to it each run, instead of taking 10 minutes per run, we save this as a modified line list that no longer needs
to be edited, and instead can be indexed directly in future runs."""

import numpy as np
from scipy.stats import norm
from numpy import random as rand
import matplotlib.pyplot as plt
import time
from astropy.io import fits
import pysme_galah1
import pickle

# The class representing the linelist data, with each different type of variable or array such as lineatomic or depth
# represented as an object in LineMerging. By running the Class itself, it runs the needed functions to perform such
# modificational acts.
class LineMerging(object):

    def __init__(self, master_file, line_list, linelist_location):
        # names in idl, from isotope mix
        self.element_list = np.array(
            ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
             'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
             'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
             'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
             'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
             'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
             'Cf', 'Es'))
        self.master_line_list = master_file
        # The important arrays that this class was created to produce. Pysme reads in a different linelist than
        # IDL wants so we adapt to that now.
        self.line_atomic, self.lande_mean, self.depth, self.data_reference_array, self.species, self.j_e_array, \
            self.lower_level, self.upper_level, self.lu_lande = self.readline(self.master_line_list)
        # Saving the data produced here, as it is NOT reliant on anything but the master file,
        # so no point recreating it every run. We instead apply the data_index desired AFTER by loading the new line
        # list and not saving over it with the indexed version.

        IDL_structure = {'atomic': self.line_atomic,        'lande': self.lande_mean,
                         'depth': self.depth,               'lineref': self.data_reference_array,
                         'species': self.species,           'line_extra': self.j_e_array,
                         "line_term_low": self.lower_level, "line_term_upp": self.upper_level,
                         "line_lulande": self.lu_lande,     "short_line_format": 2}

        pickle.dump(IDL_structure, open(linelist_location + line_list[:-5] + '_modified.csv', "wb"))

    # this is isotopic_mix.pro. Its intetion is to create an array containing the fraction of each element.
    # After, it joins the species list together into a single string.
    # Then, it checks to see if the species is in our element/isotope list and takes its index to update its
    # log_gf value. Only run if we have isotopes in the list.
    def isotopic_mix(self, species, master_line_list, rows_in_line_list):
        """
            Input:
                self: The class
                species: The species we have in the atomic lines, specifies whether it's an isotope or not as well.
                master_line_list: the atomic data with the Loggf information
                rows_in_line_list: The number of rows to loop through to adjust
            Output:
                species: The adjusted species names, put together instead of individual elements of a list (['Ti', 'O']
                master_line_list: Now with updated log gf
        """
        # isotope number and isotope fraction
        isotope_number = np.zeros((len(self.element_list), 10))
        isotope_fraction = np.zeros((len(self.element_list), 10))

        # This could be a dictionary with the names of the elements
        # associated with the values
        # Li
        isotope_number[3 - 1, [0, 1]] = [6, 7]
        isotope_fraction[3 - 1, [0, 1]] = [7.59, 92.41]
        # Light element in stars is actgually not very abundant so we replace it with 0%
        isotope_fraction[3 - 1, [0, 1]] = [0, 100.]
        # C
        isotope_number[6 - 1, [0, 1]] = [12, 13]
        isotope_fraction[6 - 1, [0, 1]] = [98.8938, 1.1062]
        # N
        isotope_number[7 - 1, [0, 1]] = [14, 15]
        isotope_fraction[7 - 1, [0, 1]] = [99.771, 0.229]
        # O
        isotope_number[8 - 1, [0, 1, 2]] = [16, 17, 18]
        isotope_fraction[8 - 1, [0, 1, 2]] = [99.7621, 0.0379, 0.2000]
        # Mg
        isotope_number[12 - 1, [0, 1, 2]] = [24, 25, 26]
        isotope_fraction[12 - 1, [0, 1, 2]] = [78.99, 10.00, 11.01]
        # Si
        isotope_number[14 - 1, [0, 1, 2]] = [28, 29, 30]
        isotope_fraction[14 - 1, [0, 1, 2]] = [92.2297, 4.6832, 3.0872]
        # Ti
        isotope_number[22 - 1, [0, 1, 2, 3, 4]] = [46, 47, 48, 49, 50]
        isotope_fraction[22 - 1, [0, 1, 2, 3, 4]] = [8.25, 7.44, 73.72, 5.41, 5.18]
        # Cu
        isotope_number[29 - 1, [0, 1]] = [63, 65]
        isotope_fraction[29 - 1, [0, 1]] = [69.17, 30.83]
        # Zr
        isotope_number[40 - 1, [0, 1, 2, 3, 4]] = [90, 91, 92, 94, 96]
        isotope_fraction[40 - 1, [0, 1, 2, 3, 4]] = [51.45, 11.22, 17.15, 17.38, 2.80]
        # Ba
        isotope_number[56 - 1, [0, 1, 2, 3, 4, 5, 6]] = [130, 132, 134, 135, 136, 137, 138]
        isotope_fraction[56 - 1, [0, 1, 2, 3, 4, 5, 6]] = [0.106, 0.101, 2.417, 6.592, 7.854, 11.232, 71.698]
        # La
        isotope_number[57 - 1, [0, 1]] = [138, 139]
        isotope_fraction[57 - 1, [0, 1]] = [0.091, 99.909]
        # Pr 2

        isotope_number[59 - 1, [0]] = [141]
        isotope_fraction[59 - 1, [0]] = [100.]
        # Nd 2

        isotope_number[60 - 1, [0, 1, 2, 3, 4, 5, 6]] = [142, 143, 144, 145, 146, 148, 150]
        isotope_fraction[60 - 1, [0, 1, 2, 3, 4, 5, 6]] = [27.044, 12.023, 23.729, 8.763, 17.130, 5.716, 5.596]
        # Sm 2

        isotope_number[62 - 1, [0, 1, 2, 3, 4, 5, 6]] = [144, 147, 148, 149, 150, 152, 154]
        isotope_fraction[62 - 1, [0, 1, 2, 3, 4, 5, 6]] = [3.07, 14.99, 11.24, 13.82, 7.38, 26.75, 22.75]
        # Eu 2

        isotope_number[63 - 1, [0, 1]] = [151, 153]
        isotope_fraction[63 - 1, [0, 1]] = [47.81, 52.19]

        # A list to recreate the compound species in one word (TiO, SiH etc)
        # we repeat for each row in the line list galah file.
        for master_list_row in range(0, rows_in_line_list):
            # A list to recreate the compound species in one word (TiO, SiH etc). Faster than for i in.
            #print(repr(master_line_list.header))

            species_ionised = "".join(master_line_list.data['name'][master_list_row]) + \
                            " " + \
                            str(master_line_list.data['ion'][master_list_row])

            species.append(species_ionised)
            #species.append("".join(master_line_list.data['name'][master_list_row]))

            # the index of the elements in the element list, which we use to index through isotope_number/fraction
            # to get their values in the next line. Still think a dictionary would be better. @@@@@@@@@
            element_index = np.isin(self.element_list, master_line_list.data['name'][master_list_row])
            # The fits file (master line list) contains the isotopes that are used for this row. So it will contain '16'
            # if we have O16 rather than O17, we use this index to find its fraction percentage that exists.
            isotope_index = np.isin(isotope_number[element_index], master_line_list.data[master_list_row]['isotope'])
            """
            debugging 
            print("element list with element index", self.element_list[element_index])
            print("isotope numbers",isotope_number[element_index])
            print("isotope frac",isotope_fraction[element_index])
    
            print("master line list isotope values", master_line_list.data[i]['isotope'])
            print("Isotope index?", isotope_index)
            print("Isotope fractions of those indices?", isotope_fraction[element_index][isotope_index])
    
            print(master_line_list.data[i]['log_gf'])
            """
            # The isotope fractions appear to have identical effects on the log gf so we loop through them and apply
            # them to the log gf of that line.
            # for each fraction in the isotope fraction index that we've compared with the fits file
            for fraction in isotope_fraction[element_index][isotope_index]:
                if fraction:
                    # Updates log gf according to fraction.
                    master_line_list.data[master_list_row]['log_gf'] = np.log10(10**(master_line_list.data[
                        master_list_row]['log_gf'])*fraction/1E2)

        return species, master_line_list

    # eleparse in idl. It takes in the names of the molecules/elements from the galah file from 'species' as names and
    # if it's a molecule it's assigned a value of 100 and ionisation of 1, if there's more than
    # one species it's separated. For PySME if it's an ion we set it to 1 later and attach it directly to species.
    def element_parsing(self, species_input, atom_index, master_line_list):
        """
        Input:
            self: The class
            species_input: An array of the atomic species to check for molecules/ions/etc
            atom_index: Indexes of atoms to compare to species indexes
            master_line_list: Name and ion data
        Output:
            atom_number: atomic numbers of the atoms
            ion_stage: Ionization of the atoms.
        """
        atom_number = np.zeros((len(species_input)))
        ion_stage = np.zeros((len(species_input)))
        """ We set the atomic number and ion value of each atom in master file, or to 100 and 0 if it's a molecule.
        Definitely feel like we could combine with iso mix, and it would benefit from a dictionary. Although I guess the 
        indexing itself is basically like a list of 0-100
        
        For PySME, the ionization is actually only read from the species name, and molecules have 1 ionization."""
        for i in range(0, len(species_input)):
            # It's a molecule, so we set to default setting of 0 and 0. The [0] is to take the actual indexes
            if i not in atom_index[0]:
                atom_number[i] = 0
                ion_stage[i] = 0
                species_input[i] = species_input[i][:-1]
                species_input[i] += str(1)
            else:
                # Set the element name to the first value (as it's an atom ,so only one element in the list exists)
                element_name = master_line_list.data[i]['name'][0]
                atomic_value_index = np.where(element_name == self.element_list)
                # If the element doesn't exist in our list we flag it as a bad one with -1
                if atomic_value_index[0].size == 0:
                    atom_number[i] = -1
                    ion_stage[i] = -1
                else:
                    # We take the index value and add one, as indexing starts at
                    # 0 but Hydrogen of course has an atomic number of 1
                    atom_number[i] = atomic_value_index[0][0] + 1
                    ion_stage[i] = master_line_list.data[i]['ion']
        return atom_number, ion_stage

    # produces a sepcies list then runs through the other two functions to produce an array with all information on
    # each species we find.
    def readline(self, master_line_list):
        rows_in_line_list = len(master_line_list.data['lambda'])

        species = []
        # If we have any isotopes we run the isotopic mix function to modify their names and values
        if np.any(master_line_list.data['isotope']) > 0:

            # Return the species list
            species, master_line_list = self.isotopic_mix(species, master_line_list, rows_in_line_list)
            species = np.asarray(species)
        # If we aren't doing that function as we have no isotopes,
        # we make the species list here, it only takes a few seconds.
        else:

            # For each list of elements involved in the line species, we join them together ([C, N] becomes CN)
            for elements in master_line_list.data['name']:
                species.append("".join(elements))
            species = np.asarray(species)

        # Replace CC with C2 1 denomination. 1 for being a molecule, which Pysme recognises.
        replacing = 'CC'
        # Finds the indexes where CC is in the species value, doesn't have to be exact due to the ionisation value.
        # We'd expect it to be 1 but this is safer.
        ccindex = [i for i, v in enumerate(species) if replacing in v]

        species[ccindex] = 'C2 1'

        # Finding indexes that contain only one species.
        # Np.asarray required as the fits files are given in chararray which
        # is outdated and only there for backwards compatibility.
        # 'not' means is empty so there are no other species beyond
        # the first. Could put this in the other masterlinelist loop? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # We use it to input values in atomnumb array in element parsing.
        atom_index = np.where(np.asarray([not x[1] for x in master_line_list.data['name']]))
        # It seems to think some of the element list might have isotopes,
        # so I'm using an input for if we want to use a list
        # that is variable like that
        # We want to find the atomic number and ionization of the species we have.
        atom_number, ion_stage = self.element_parsing(species, atom_index, master_line_list)
        # We just plug the data from the galah file into columns, basically. Necesary input for sme
        # data array is atomic.

        data_array = np.transpose(np.array((atom_number, ion_stage, master_line_list.data['lambda'],
                                            master_line_list.data['e_low'], master_line_list.data['log_gf'],
                                            master_line_list.data['rad_damp'], master_line_list.data['stark_damp'],
                                            master_line_list.data['vdw_damp'])))

        lande_mean = np.transpose(master_line_list.data['lande_mean'])
        depth = np.transpose(master_line_list.data['depth'])
        lu_lande = np.transpose(np.array((master_line_list.data['lande_low'], master_line_list.data['lande_up'])))
        j_e_array = np.transpose(
            np.array((master_line_list.data['j_low'], master_line_list.data['e_up'], master_line_list.data['j_up'])))
        lower_level = np.transpose(np.array((master_line_list.data['label_low']).replace('\'', '').strip()))
        upper_level = np.transpose(np.array((master_line_list.data['label_up']).replace('\'', '').strip()))
        data_reference_array = np.transpose(
            np.array([master_line_list.data['lambda_ref'], master_line_list.data['e_low_ref'],
                      master_line_list.data['log_gf_ref'], master_line_list.data['rad_damp_ref'],
                      master_line_list.data['stark_damp_ref'],
                      master_line_list.data['vdw_damp_ref'],
                      master_line_list.data['lande_ref']]))
        # Not really totally sure wheret he 0.3 comes from,
        # or why we use data_array at all when we could just call on lambda.

        bad_element_index = np.where(atom_number < 0)
        bad_ion_index = np.where(ion_stage < 0)
        if bad_element_index[0].size > 0:
            bad_element_list = species[bad_element_index]
            print("Unkown species:", bad_element_list)
        if bad_ion_index[0].size > 0:
            bad_ion_list = species[bad_ion_index]
            print("Unkown ion charges:", bad_ion_list)
        # And then we just trim off the lande and depth columns from data!
        # what was the point in any of that?! @@@@@@@@@@@@@@
        # Oh but dataref still has the lande ref! @@@@@@@@@@@@@@@

        return data_array, lande_mean, depth, data_reference_array, \
               species, j_e_array, lower_level, upper_level, lu_lande


"""at this point there's a lot of looping through each file to add on their own data array etc to the array. It doesn't
make alot of sense if we only have one file so I'm ignoring that (oh and it just calls abuff without defining it. 
I think it must be data array).
Also more info on if not short etc. but we always have short set to 2 so that's useless.
Also a lot of transposing and reforming which appears to not do anything for us. So linemerge is a useless file that 
just calls readline."""

# Idl sets range = reform(fltarr(2, nfile),2,nfile). Short is always 2 so not sure why we'd call it but.
# line_atomic, line_species, line_lande, line_depth, line_ref,term_low=line_term_low,term_upp=line_term_upp,short_
# format=short_format,extra=line_extra,lulande=line_lulande, line j e is extra, wavelength range is not used


def run_merger(master_line_list, linelist_location, line_list=0):
    # In case we have no input linelist, we won't throw an error. If they don't have galah1 EITHER, then that's an issue
    if not line_list:
        _, line_list, _ = pysme_galah1.set_atmosphere()
        print("No linelist found in Makestruct. Depth is probably wrong. Get some help, this shouldn't happen.")
        master_line_list = fits.open("GALAH/LINELIST/"+line_list)[1]

    LineMerging(master_line_list, line_list, linelist_location)