#!/bin/bash

# Default variable values
genData=false
train=false
predict=false
refernce=false

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
}


# Function to handle options and arguments
while getopts :hcgtprn: flag 2>/dev/null
do
    case "${flag}" in
        h) usage && exit 0;;
        c) rm -R \temp;;
        g) genData=true;;
        t) train=true;;
        p) predict=true;;
        r) refernce=true;;
        :) echo "Option -$OPTARG requires an argument." >&2 && exit 1;;
        ?) usage && exit 1;;
    esac
done

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

mkdir -p qmdata
python  -m salted.get_basis_info 

# Perform the desired actions based on the provided flags and arguments
if [ "$genData" = true ]; then
    echo "Running PySCF:" 
	python /scripts/submitJobs.py PySCF
fi

if [ "$train" = true ]; then
    echo "Training SALTED ML model:" 
 	python /scripts/submitJobs.py Training
fi

if [ "$predict" = true ]; then
	echo "Predicting Coeffs.:"
	python /scripts/predict.py
fi

if [ "$refernce" = true ]; then
	echo "Calculating reference:"
	python -m salted.pyscf.calc_reference
fi
