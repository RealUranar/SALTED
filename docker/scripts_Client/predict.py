import sys,os
import time
import numpy as np
from ase.io import read
from salted import init_pred 
from salted import salted_prediction 

from salted.sys_utils import ParseConfig
inp = ParseConfig().parse_input()
start = time.perf_counter()
#Create directorys
dirpath = os.path.join(inp.salted.saltedpath, "prediction")
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Initialize SALTED prediction
lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals = init_pred.build()

# do prediction for the given structure    
frames = read(inp.prediction.filename,":")
for i in range(len(frames)):
    structure = frames[i]
    coefs = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,structure) 
    np.save(os.path.join(dirpath,"COEFFS-"+str(i+1)+".npy"),coefs)

end = time.perf_counter()
print(f"Time: {end-start}")