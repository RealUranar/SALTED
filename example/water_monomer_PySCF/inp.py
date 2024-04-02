# system definition 
# -----------------
filename = "water_monomers_1k.xyz" # XYZ file
species = ["H","O"] # ordered list of species
qmcode = 'pyscf'
average = True
parallel = False 
field = False

# Rascaline atomic environment parameters
# ---------------------------------------
rep1 = 'rho'
rcut1 = 4.0
nrad1 = 4
nang1 = 5
sig1 = 0.3
rep2 = 'rho'
rcut2 = 4.0
nrad2 = 4
nang2 = 5
sig2 = 0.3
neighspe1 = ["H","O"] # ordered list of species
neighspe2 = ["H","O"] # ordered list of species

# Feature sparsification parameters
# ---------------------------------
sparsify = False 
nsamples = 100 # Number of structures to use for feature sparsification
ncut = 0 # Set ncut = 0 to skip feature sparisification

# paths to data
# -------------
saltedpath = './'
saltedname = 'test'
path2qm = "qmdata/" # path to the raw PySCF output

# PySCF variables 
# --------------
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis
dfbasis = "RI-cc-pvqz" # auxiliary basis

# ML variables  
# ------------
z = 2.0           # kernel exponent 
Menv = 10        # number of FPS environments
Ntrain = 800       # number of training structures
trainfrac = 1.0   # training set fraction
regul = 1e-10      # regularisation parameter
eigcut = 1e-10    # eigenvalues cutoff

# Parameters for direct minimization
#-----------------------------------
gradtol = 1e-5    # convergence parameter
restart = False   # restart minimization

# Parameters if performing matrix inversion
#------------------------------------------
blocksize = 0
trainsel = 'random'

# Prediction Paths
# ------------
predname = 'prediction'
predict_filename = "water_pred.xyz"