import glob, os, sys
import numpy as np
from salted.sys_utils import ParseConfig, read_system


species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
inp = ParseConfig().parse_input()

PARTITON_TO_MEM = {
    "c23ml": 10600,
    "c23mm": 5300,
    "c23ms": 2600
}

def write_missing_indexes():
    files = glob.glob(os.path.join(inp.qm.path2qm,"overlaps","*.npy"))
    indexes = np.sort([int(file.split("conf")[-1].replace(".npy", "")) for file in files])
    missing = np.setdiff1d(np.arange(ndata), indexes)
    np.savetxt(os.path.join(inp.salted.saltedpath,"missing_indexes.txt"), missing, fmt="%i")
    if len(missing) == 0:
        print("No missing indexes")
        exit()
    elif len(missing) >= 1000:
        return 999
    return len(missing)

JOBTYPES = {
    "PySCF" : {
        "job-name": "PySCF",
        "output": "output.%A.%a.txt",
        "time": "0-00:20:00",
        "cpus-per-task": 8,
        "ntasks": 1,
        "nodes": 1,
        "partition": "c23ms",
        "account": "thes1689",
        "array": "0-999",
    },
    "Training" : {
        "job-name": "Training",
        "output": "output.%A.txt",
        "time": "0-04:00:00",
        "cpus-per-task": 1,
        "ntasks": 30,
        "nodes": 1,
        "partition": "c23ms",
        "account": "thes1689"
    },
}

def writeSubmitScripts(jobtype):
    """
    Submit jobs to the cluster
    """
    if jobtype not in JOBTYPES:
        raise ValueError(f"Jobtype {jobtype} not supported")
    
    script = "#!/usr/bin/zsh\n\n########## start of batch directives #######\n"
    job = JOBTYPES[jobtype]
    for key, value in job.items():
        if key == "array" and jobtype == "PySCF":
            lenMissing = write_missing_indexes()
            if lenMissing == 1:
                value = "0"
            else:
                value = f"0-{lenMissing-1}"
        script += f"#SBATCH --{key}={value}\n"
        
    script += "\n########## end of batch directives #########\n"
    script += 'echo "Starting job $SLURM_JOB_ID array ID $SLURM_ARRAY_TASK_ID"\n'
    if jobtype == "PySCF":
        script += "apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif python -m salted.pyscf.run_pyscf -i ${SLURM_ARRAY_TASK_ID}\n"
        script += "apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif python -m salted.pyscf.dm2df_pyscf -i ${SLURM_ARRAY_TASK_ID}\n"
        script += "cat output.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}.txt >> output.${SLURM_ARRAY_JOB_ID}.txt\n"
        script += "rm output.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}.txt\n"
    elif jobtype == "Training":
        nproc = JOBTYPES[jobtype]["ntasks"]
        script += "apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif python  -m salted.initialize\n"
        script += "apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif python  -m salted.sparse_selection\n"
        script += f"apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif mpirun -np $SLURM_NTASKS  python -m salted.sparse_descriptor\n"
        script += f"apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif python -m salted.rkhs_projector\n"
        script += f"apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif mpirun -np $SLURM_NTASKS  python -m salted.rkhs_vector\n"
        script += f"apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif mpirun -np $SLURM_NTASKS 	python 	-m salted.minimize_loss\n" 
        script += f"apptainer exec --bind=$HPCWORK --no-home $HPCWORK/apptainers/salted.sif mpirun -np $SLURM_NTASKS 	python 	-m salted.validation"

    with open(f"submit_{jobtype}.sh", "w") as f:
        f.write(script)
    os.chmod(f"submit_{jobtype}.sh", 0o755)

if __name__ == "__main__":
    #Check if argument is either PySCF or Training
    jobtype = sys.argv[1]
    if jobtype not in JOBTYPES:
        raise ValueError(f"Jobtype {jobtype} not supported")
    
    if jobtype == "PySCF":
        """ prepare directories to store data """
        for data_dname in ("coefficients", "projections", "overlaps", "density_matrices"):
            if not os.path.exists(dpath := os.path.join(inp.qm.path2qm, data_dname)):
                os.makedirs(dpath)
    writeSubmitScripts(jobtype)
    print(f"Submit script for {jobtype} written")
    
