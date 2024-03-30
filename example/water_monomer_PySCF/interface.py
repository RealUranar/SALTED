import sys
import time
import numpy as np
from ase.io import read
from salted import init_pred 
from salted import salted_prediction 

sys.path.insert(0, './')
import inp

# Initialize SALTED prediction
lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals = init_pred.build()

# do prediction for the given structure    
frames = read(inp.filename,":")
for i in range(len(frames)):
    structure = frames[i]
    coefs = salted_prediction.build(lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals,structure) 
    np.savetxt("COEFFS-"+str(i+1)+".dat",coefs)
