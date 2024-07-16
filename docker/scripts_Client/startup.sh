#!/bin/bash

# Default variable values
genData=false
train=false
predict=false
refernce=false
nProc=16

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help              Display this help message"
 echo " -c, --clear             Delete temp directory"
 echo " -g  --genTrainData      Run PySCF to generate training data from provided file"
 echo " -t, --trainModel        Run the training steps"
 echo " -p, --predict           Predict the coefficents for a provided structure"
 echo " -r, --refernce          Do a refernce PySCF calculation on the provided Structure"
 echo " -n, --nprocesses        Number of processes to use for MPI"
}

while getopts :hcgtprn: flag 2>/dev/null
do
    case "${flag}" in
        h) usage && exit 0;;
        c) rm -R \temp;;
        g) genData=true;;
        t) train=true;;
        p) predict=true;;
        r) refernce=true;;
        n) nProc=${OPTARG};;
        :) echo "Option -$OPTARG requires an argument." >&2 && exit 1;;
        ?) usage && exit 1;;
    esac
done


export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mkdir -p /temp
mkdir -p /temp/qmdata
cd /temp

ntasks=$nProc
echo "Number of processes: $ntasks"

# Perform the desired actions based on the provided flags and arguments
if [ "$genData" = true ]; then
    echo "Running PySCF:" 
	python /scripts/moveData.py genData 
    python -m salted.pyscf.run_pyscf
	python -m salted.get_basis_info 
	python -m salted.pyscf.dm2df_pyscf
	python /scripts/moveData.py genDataDone
fi

if [ "$train" = true ]; then
    echo "Training SALTED ML model:" 
	python /scripts/moveData.py buildModel
	python  -m salted.get_basis_info 
	python  -m salted.initialize
	python  -m salted.sparse_selection
	mpirun -np $ntasks python -m salted.sparse_descriptor
	python -m salted.rkhs_projector
	mpirun -np $ntasks  python -m salted.rkhs_vector
	mpirun -np $ntasks 	python 	-m salted.minimize_loss 
	mpirun -np $ntasks 	python 	-m salted.validation 

	python /scripts/moveData.py buildModelDone
fi

if [ "$predict" = true ]; then
	python /scripts/moveData.py predictStructure 
	echo "Predicting Coeffs.:"
    	python -m salted.get_basis_info
	python /scripts/predict.py
	python /scripts/moveData.py predictStructureDone
fi

if [ "$refernce" = true ]; then
    echo "Calculating reference:"
	python /scripts/moveData.py calcReference
	python -m salted.pyscf.calc_reference
	python /scripts/moveData.py calcReferenceDone
fi


cd /src && rm -R __pycache__