"""We use the abundances and equations given to us by the marc_abund.pro file. We normalise/log a lot of things, but then
un-log Hydrogen because apparently SME doesn't want loggedH. Still confused about how that's ok. Maybe it logs H itself.
Only run before pysme, thus using sp3 metallicity"""


import numpy as np
from scipy.interpolate import interp1d
# metallicity taken from the file that gets it from the cannon fits file
# It also calls for enoh12 and alpha keyword? But it produces eonh12, but alpha is still a mystery...

"""Skipping
if n_params() lt 1 then begin
  print, 'syntax: solar_abund, abund ,[,elem ,atwe]'
  return
endif"""
def create_abundances(metallicity):
    #;Logarithmic abundances relative to hydrogen with hydrogen set to 12.
    #;Reference: Grevesse, Asplund & Sauval 2007, Space Sciences Review 130, 105
    #;Unknown value: -8.00

                            #;   "H",  "He",  "Li",  "Be",   "B",   "C",   "N",   "O",   "F",  "Ne",
    element_abund1 = np.array([12.00, 10.93,  1.05,  1.38,  2.70,  8.39,  7.78,  8.66,  4.56,  7.84,
    #;  "Na",  "Mg",  "Al",  "Si",   "P",   "S",  "Cl",  "Ar",   "K",  "Ca",
       6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.50,  6.18,  5.08,  6.31,
    #;  "Sc",  "Ti",   "V",  "Cr",  "Mn",  "Fe",  "Co",  "Ni",  "Cu",  "Zn",
       3.17,  4.90,  4.00,  5.64,  5.39,  7.45,  4.92,  6.23,  4.21,  4.60,
    #;  "Ga",  "Ge",  "As",  "Se",  "Br",  "Kr",  "Rb",  "Sr",   "Y",  "Zr",
       2.88,  3.58,  2.29,  3.33,  2.56,  3.25,  2.60,  2.92,  2.21,  2.58,
    #;  "Nb",  "Mo",  "Tc",  "Ru",  "Rh",  "Pd",  "Ag",  "Cd",  "In",  "Sn",
       1.42,  1.92, -8.00,  1.84,  1.12,  1.66,  0.94,  1.77,  1.60,  2.00])
                            #;   "Sb",  "Te",   "I",  "Xe",  "Cs",  "Ba",  "La",  "Ce",  "Pr",  "Nd",
    element_abund2=np.array([ 1.00,  2.19,  1.51,  2.24,  1.07,  2.17,  1.13,  1.70,  0.58,  1.45,
    #;  "Pm",  "Sm",  "Eu",  "Gd",  "Tb",  "Dy",  "Ho",  "Er",  "Tm",  "Yb",
      -8.00,  1.00,  0.52,  1.11,  0.28,  1.14,  0.51,  0.93,  0.00,  1.08,
    #;  "Lu",  "Hf",  "Ta",   "W",  "Re",  "Os",  "Ir",  "Pt",  "Au",  "Hg",
       0.06,  0.88, -0.17,  1.11,  0.23,  1.25,  1.38,  1.64,  1.01,  1.13,
    #;  "Tl",  "Pb",  "Bi",  "Po",  "At",  "Rn",  "Fr",  "Ra",  "Ac",  "Th",
       0.90,  2.00,  0.65, -8.00, -8.00, -8.00, -8.00, -8.00, -8.00,  0.06,                     #''there is no tio'
    #  "Pa",   "U",  "Np",  "Pu",  "Am",  "Cm",  "Bk",  "Cf(but Cs in the code, typo)",  "Es", "TiO"
       8.00, -0.52, -8.00, -8.00, -8.00, -8.00, -8.00,   -8.00,                         -8.00])
    # What is the difference here? ~~~@@@@@@
    eonh12   = [element_abund1, element_abund2]
    eonh     = [element_abund1, element_abund2]

    # Why the hell are we just minusing things by 12? Shouldn't we be LOGGING IT? or is it because it is used later to
    # actually produce t he log? @@@@@@@@@@@@@@@@@@@@@@@@ (yes)
    log_eonh = np.concatenate([(element_abund1)-12, (element_abund2) - 12])


    # Adding alpha enhancement. What does that mean? @@@@@@@@@@@@@@@@@@@@@@@@@
    # What is this list for? What is 'i', and why are we taking 1 away immediately why not just do the right numbers? @@@@@@@@@
    alpha_list = np.array([8,10,12,14,16,18,20,22]) - 1
    # Now we wanted to check for the existance of the metallicity keyword, but we import it directly so how can
    # Same for non-existence of alpha
    # we test if it exists or not? when doesn't it? when does it? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Interpolates for the point on this graph for metallicity. Not sure if we need to change these values ever? @@@@@@@@@@@
    # Swapped order from IDL as it goes y, x and we want x, y
    interpolation_line = interp1d([-10.,-1.,0.,10.], [0.4,0.4,0.0,0.0], kind="linear")
    # Check values with SVen. Basically finds the data for the metallicity data point. (Just 1? ) @@@@@@@@@@@
    alpha_enhancement = interpolation_line(int(metallicity))
    print("Alpha enhancement:", alpha_enhancement)

    # adjust for alpha for each element that we specified earlier to modify
    for element in alpha_list: log_eonh[element] = log_eonh[element] + alpha_enhancement

    # Convert abundances relative to total number of nuclei
    eonh = 10**log_eonh
    renorm = sum(eonh)
    eontot = eonh/renorm
    # SME wants non-logarithmic Hydrogen abundance.
    log_eontot = (np.log10(eontot))
    abundances = log_eontot
    # what the hell. How does this work, everygting else is in log, but we change hydrogen back to % or something? @@@@@@@@
    abundances[0] = eonh[0]/renorm

    # Element name list

    element_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
            'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
            'Bk', 'Cf', 'Es']

    """;Build array of atomic weights.
    ;From Loss, R. D. 2003, Pure Appl. Chem., 75, 1107, "Atomic Weights
    ; of the Elements 2001". The isotopic composition of the Sun and stars
    ; may differ from the terrestrial values listed here!
    ;For radionuclides that appear only in Table 3, we have adopted the
    ; atomic mass of the longest lived isotope, rounding to at most two
    ; digits after the decimal point. These elements are: 43_Tc_98, 
    ; 61_Pm_145, 85_At_210, 86_Rn_222, 87_Fr_223, 88_Ra_226, 89_Ac_227,
    ; 93_Np_237, 94_Pu_244, 95_Am_243, 96_Cm_247, 97_Bk_247, 98_Cf_251,
    ; 99_Es_252"""
    # Should we do dict with name: weight? atwe doesn't actually seem to be used anywhere? @@@@@@@@@@@@@@@@@@@@@@
    atomic_weights=[1.00794 ,   4.002602,   6.941,   9.012182,  10.811,  12.0107,  14.0067,  15.9994,  18.9984032, 20.1797
           ,  22.989770,  24.3050   ,  26.981538,  28.0855   ,  30.973761
           ,  32.065   ,  35.453    ,  39.948   ,  39.0983   ,  40.078
           ,  44.955910,  47.867    ,  50.9415  ,  51.9961   ,  54.938049
           ,  55.845   ,  58.933200 ,  58.6934  ,  63.546    ,  65.409
           ,  69.723   ,  72.64     ,  74.92160 ,  78.96     ,  79.904
           ,  83.798   ,  85.4678   ,  87.62    ,  88.90585  ,  91.224
           ,  92.90638 ,  95.94     ,  97.91    ,  95.94     , 101.07
           , 102.90550 , 106.42     , 107.8682  , 112.411    , 114.818
           , 118.710   , 121.760    , 127.60    , 126.90447  , 131.293
           , 132.90545 , 137.327    , 138.9055  , 140.116    , 140.90765
           , 144.24    , 144.91     , 150.36    , 151.964    , 157.25
           , 158.92534 , 162.500    , 164.93032 , 167.259    , 168.93421
           , 173.04    , 174.967    , 178.49    , 180.9479   , 183.84
           , 186.207   , 190.23     , 192.217   , 195.078    , 196.966
           , 200.59    , 204.3833   , 207.2     , 208.98038  , 209.99
           , 222.02    , 223.02     , 226.03    , 227.03     , 232.0381
           , 231.03588 , 238.02891  , 237.05    , 244.06     , 243.06
           , 247.07    , 247.07     , 251.08    , 252.08     ]
    return abundances

