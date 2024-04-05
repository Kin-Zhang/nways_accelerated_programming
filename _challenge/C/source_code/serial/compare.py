
import numpy as np
 
# Original output data file
orig_file = "orig_velocity.dat"
new_file = "new_velocity.dat" 

orig_data = np.loadtxt(orig_file,delimiter = ' ')
new_data = np.loadtxt(new_file,delimiter = ' ')


diff_data = new_data - orig_data

print("shape of orig_data:",orig_data.shape)
print("shape of new_data:",new_data.shape)
print("shape of diff_data:",diff_data.shape)

maxError = np.amax(diff_data)

print("shape of maxError:",maxError.shape)

maxError_exp = "{:e}".format(maxError)
 
print('Max Error is : ', maxError_exp)
