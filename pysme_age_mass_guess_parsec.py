"""Unsure on the details, I think it gets a mass and age of the star estimate based on our input
Used in Update_logg"""
# coding: utf-8

# # Fast Mass/Age estimation with Parsec isochrones
# 
# ##  Sven Buder based on age_mass_guess by Jane Lin

# In[1]:

import numpy as np
import glob
import sys
from scipy.interpolate import interp1d


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


# In[2]:
def set_y():

    try:
        ELLI_directory = r'GALAH/ELLI'
    except FileNotFoundError as bigerror:
        print("Can't find the ELLI folder in /GALAH/ELLI")
        raise bigerror
    """if len(glob.glob(ELLI_directory)) != 1:
        ELLI_directory = '/avatar/buder/trunk/GALAH/ELLI'
        if len(glob.glob(ELLI_directory)) != 1:
            ELLI_directory = '/shared-storage/buder/svn-repos/trunk/GALAH/ELLI'
            if len(glob.glob(ELLI_directory)) != 1:
                ELLI_directory = '/home/k/klind/pfs/trunk/GALAH/ELLI'
                if len(glob.glob(ELLI_directory)) != 1:
                    ELLI_directory = '/Users/buder/trunk/GALAH/ELLI'
                    if len(glob.glob(ELLI_directory)) != 1:
                        sys.exit('You are not Sven working on Avatar, Gemini2, Abisko or Svens computer, please adjust ELLI directory')
    """
# If the isochrones have already been downloaded, they are simply read in.
#
# Otherwise, download the Parsec isochrones for the following set of [Fe/H] and ages.
#
# Because Parsec isochrones are based on metallicity $Z$ (and $Z_\odot = 0.0152$), we compute them based [Fe/H] and the standard alpha-enhancement.
#
# To do so, we assume that metallicity [M/H] scales with [Fe/H] and [alpha/Fe] as found by Salaris & Cassisi (2005)
#
# With the given age and $z$, we can now download the isochrones from the parsec website and saved as 'Parsec_isochrones.npy'.
    # In[12]:
    if len(glob.glob(ELLI_directory+'/Parsec_isochrones.npy'))==0:

        try:
            # A python package that allows you to download PADOVA isochrones directly from their website
            from ezpadova import cmd
        except:
            print("No EZPADOVA")
            sys.exit('no EZPADOVA available')

        fe_h = np.arange(-2.4,0.65,0.1)
        age = np.arange(0.5,13.51,0.5)

        alpha_fe              = -0.4*fe_h
        alpha_fe[fe_h < -1.0] = 0.4
        alpha_fe[fe_h >= 0.0] = 0.

        m_h = fe_h + np.log10(0.694*10**alpha_fe + 0.306)

        z = 10**(m_h) * 0.0152

        y = []

        for each_m_h, each_fe_h, each_alpha_fe in zip(m_h, fe_h, alpha_fe):

            iso_1 = read_iso(age)
            iso_1.fill_chemistry(each_m_h, each_fe_h, each_alpha_fe)
            iso_data = []

            for each_age in age:
                iso = cmd.get_one_isochrone(each_age * 10**9, metal=10**each_m_h*0.0152);

                iso_data.append(dict(
                    M_Mo = iso['M_act'],
                    logTeff = iso['logTe'],
                    logG = iso['logG'],
                    logL_Lo = iso['logLLo']
                ))

            iso_1.fill_iso(np.array(iso_data))
            y.append(iso_1)

        np.save(ELLI_directory+'/'+'Parsec_isochrones.npy', y)
    else:

        y = np.load(ELLI_directory+'/'+'Parsec_isochrones.npy', allow_pickle=True, encoding='latin1')

    feh_iso = [i.FeH for i in y]


    # Now we choose which isochrone information to use

    # In[4]:

    include_teff=1
    include_logg=1
    include_lbol=1
    include_feh=1
    mask=np.array([include_teff,include_logg,include_lbol,include_feh])

    twopi4=pow(2*np.pi,4) #(2*pi)^4

    # Added function to stop naked code always running and throwing errors
    return y, feh_iso, mask, twopi4

def interp(x,y):
    return interp1d(x=x,y=y,kind='linear')


# In[5]:

def age_mass_guess(iso_set, value, sigma):

    _, _, mask, twopi4 = set_y()

    def lnProb_guess(value,sigma,model):
        norm_guess=np.log(np.sqrt( twopi4 * np.prod(pow(sigma[np.nonzero(mask)],2))))
        return -sum(mask*  pow( (value-model)/sigma, 2)) - norm_guess

    #empty storage
    lnP_guess=[]
    for isoc in iso_set.data:
        #for each point in the isochrone, calculate ln(P)
        model=np.array([pow(10,isoc['logTeff']), isoc['logG'], pow(10,isoc['logL_Lo']), iso_set.FeH])
        lnP_guess.append(lnProb_guess(value,sigma,model))

    lnP_guess=np.array(lnP_guess)
    mass_guess=0.0
    age_guess=0.0
    norm_guess=0.0

    #now do a weighted sum of mass and age
    for i, iso_age in enumerate(iso_set.ages):

        p=np.exp(lnP_guess[i])
        age_guess += iso_age * sum(p)
        mass_guess += sum(iso_set.data[i]['M_Mo'] * p)
        norm_guess += sum(p)

    mass_guess = mass_guess/norm_guess
    age_guess = age_guess/norm_guess
    return age_guess, mass_guess


# In[6]:

def do_age_mass_guess(value,sigma,y,feh):
    #value[3] = 0.8421765754780444  # interpolation error
    #value = np.array((2.50000000e+03, 3.00000000e+00, 2.24160976e-01, 0.6))
    #sigma = np.array((1.00000000e+02, 5.00000000e-01, 2.02703993e-02, 2.00000000e-01))



    # We set FeH to be within the boundaries of th isochrone metallicity information. wE still use value 3 later tho
    FeH=value[3]
    if FeH>=feh[-1]:
        FeH=feh[-2]
    if FeH<=feh[0]:
        FeH=feh[0]
    #print("y0[0].FeH, FeH, y[-1].FeH", y[0].FeH, FeH, y[-1].FeH)
    # It always should be with our prev. if statements. (within the metallicity of isochrones)
    if y[0].FeH <= FeH <= y[-1].FeH:
        # For every isochrone in the file
        for i in range(len(y)):
            # which isochrone is our metallicity at? If trimmed before it will be at the min or max-1 one.
            if y[i].FeH <= FeH < y[i+1].FeH:
                # index of the isochrone closest to our metallicity. Always rounds down?
                iy=i

                break
        ageg0=[]; massg0=[]
        # takes fewer values if nearer the maximum FeH of 0.5 or 0.6. Is that an issue?
        for x in y[iy:iy+2]:
            ageg1,massg1=age_mass_guess(x, value, sigma)
            if np.isnan(ageg1) ==False and np.isnan(massg1)==False:
                ageg0.append(ageg1)
                massg0.append(massg1)
        if len(ageg0)>1:
            #print("The interp errorvars", ([y[iy].FeH,y[iy+1].FeH],[ageg0[0],ageg0[1]]),(value[3]))
            #y[iy].FeH = 0.5000000000000027  # interpolation error
            #y[iy+1].FeH = 0.6000000000000028
            #ageg0[0] = 6.623680208254429
            #ageg0[1]=6.872902583343335
            #value[3] = 0.8421765754780444

            ageg=interp([y[iy].FeH,y[iy+1].FeH],[ageg0[0],ageg0[1]])(value[3])
            massg=interp([y[iy].FeH,y[iy+1].FeH],[massg0[0],massg0[1]])(value[3])
            return(ageg,massg)
        if len(ageg0)==0:
            print('no starting guesses found, returning 5Gyr & 1M_o')
            return(np.isnan,np.isnan)
        else:
            return(ageg0[0],massg0[0])


# In[7]:

def print_result(sys_input, y, feh_iso, ELLI_directory):

    sys.argv = [sys_input[-1],str(sys_input[0]),str(sys_input[1]),str(sys_input[2]),str(sys_input[4]),
                str(sys_input[3]),'100','0.2',str(sys_input[4]/10.),'0.2']

    if sys.argv[2]<=50:
        print('Please feed in TEFF on linear scale')
    if sys.argv[4]<=0:
        print('Please feed in LBOL on linear scale')

    age,mass = do_age_mass_guess(np.array([float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),
                                           min([max([float(sys.argv[5]),-2.4]),0.6])]),
                                 np.array([float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8]),
                                           float(sys.argv[9])]),y,feh_iso)

    age=float(age)
    mass=float(mass)

    np.savetxt(ELLI_directory+'/'+sys.argv[1]+'_mass.txt',[[age,mass]],fmt='%s')

    print("{sobject_id:s}: Age: {age:10.8f},  Mass: {mass:10.8f}, dMass: {dmass:10.8f}, SeisMass: "
          "{seis_mass:10.8f}".format(sobject_id=sys.argv[1], age=age, dmass=mass-sys.argv[0], seis_mass=sys.argv[0],
                                     mass=mass))
    return mass


# In[11]:
def save_mass(y, feh_iso, ELLI_directory):

    age,mass = do_age_mass_guess(np.array([float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),
                                           min([max([float(sys.argv[5]),-2.4]),0.6])]),
                                 np.array([float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8]),float(sys.argv[9])])
                                 , y, feh_iso)

    age=float(age)
    mass=float(mass)

    np.savetxt(ELLI_directory+'/'+sys.argv[1]+'_mass.txt',[[age,mass]],fmt='%s')



# In[10]:

# sys_input = [123456789,5772.,4.438,0.00,1.00,1.00]
# print_result(sys_input)

