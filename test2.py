from astropy.table import Table
from astropy.io import fits
import pysme_interpolate_depth

a = fits.open("D:/PhD/Pysme/ges_master_v5_4250_4.50_-2.50.fits")
print(repr(a[0].header))
makestruct_dict = {}
makestruct_dict['gravity'] = 4.50
makestruct_dict['effective_temperature'] = 4250
makestruct_dict['metallicity'] = -2.5
makestruct_dict['line_list'] = "ges_master_v5.1"
b = pysme_interpolate_depth.reduce_depth(makestruct_dict)
print(len(b))