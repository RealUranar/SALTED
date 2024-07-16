import glob
import os, sys
import shutil

sys.path.insert(0, "/src")
from salted.sys_utils import ParseConfig
inp = ParseConfig().parse_input()

tempDirectory = "."
arg = sys.argv[1]
    
#Always copy inp.py
shutil.copy2(os.path.join("/src/","inp.yaml"), tempDirectory)

if arg == "genData":
    print("Moving data for PySCF")
    shutil.copy2(os.path.join("/src/",inp.system.filename), tempDirectory)
elif arg == "genDataDone":
    print("Saving PySCF data to Host")
    os.system(f"tar -cf qmdata.tar {os.path.join(inp.qm.path2qm,  'coefficients')} \
                                    {os.path.join(inp.qm.path2qm, 'overlaps')}\
                                    {os.path.join(inp.qm.path2qm, 'projections')}\
                                    && cp qmdata.tar /src")

elif arg == "buildModel":
    print("Moving data to build Model")
    shutil.copy2(os.path.join("/src/",inp.filename), tempDirectory)
    if not os.path.exists("/temp/qmdata"):
    	os.system(f"cp -u /src/qmdata.tar . \
                	&& tar -xf qmdata.tar")
     
elif arg == "buildModelDone":
    with open("inputs.txt", "w") as f:
        out = f"average = {inp.system.average}\n"
        out += f"field = {inp.system.field}\n"
        out += f"sparsify = {inp.descriptor.sparsify.ncut > 0}\n"
        out += f"ncut = {inp.descriptor.sparsify.ncut}\n"
        out += f"species = {inp.system.species}\n"
        out += f"rcut1 = {inp.descriptor.rep1.rcut}\n"
        out += f"rcut2 = {inp.descriptor.rep2.rcut}\n"
        out += f"nang1 = {inp.descriptor.rep1.nang}\n"
        out += f"nang2 = {inp.descriptor.rep2.nang}\n"
        out += f"nrad1 = {inp.descriptor.rep1.nrad}\n"
        out += f"nrad2 = {inp.descriptor.rep2.nrad}\n"
        out += f"sig1 = {inp.descriptor.rep1.sig}\n"
        out += f"sig2 = {inp.descriptor.rep2.sig}\n"
        out += f"neighspe1 = {inp.descriptor.rep1.neighspe}\n"
        out += f"neighspe2 = {inp.descriptor.rep2.neighspe}\n"
        out += f"zeta = {inp.gpr.z}\n"
        out += f"Menv = {inp.gpr.Menv}\n"
        out += f"Ntrain = {inp.gpr.Ntrain}\n"
        out += f"trainfrac = {inp.gpr.trainfrac}\n"
        out += f"dfbasis = {inp.qm.dfbasis}\n"
        f.write(out)

    cmd = (
        f"tar -cf model.tar "
        f"--transform 's|^|Model/|' inputs.txt averages "
        f"--transform 's|equirepr_{inp.salted.saltedname}/|GPR_data/|' "
        f"$(find equirepr_{inp.salted.saltedname}/ -type f -name 'fps{inp.descriptor.sparsify.ncut}-*.npy') "
        f"$(find equirepr_{inp.salted.saltedname}/ -type f -name 'FEAT_M-{inp.gpr.Menv}.h5') "
        f"$(find equirepr_{inp.salted.saltedname}/ -type f -name 'projector_M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}.h5') "
        f"--transform 's|regrdir_{inp.salted.saltedname}/M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}/|GPR_data/|' "
        f"$(find regrdir_{inp.salted.saltedname}/M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}/ -type f -name 'weights_N{int(inp.gpr.Ntrain*inp.gpr.trainfrac)}_reg*') "
        # f"--transform 's|wigners/|Model/wigners/|' "
        f"$(find wigners/ -type f -name 'wigner_lam-*_lmax1-{inp.descriptor.rep1.nang}_lmax2-{inp.descriptor.rep2.nang}.dat')"
    )

    os.system(cmd)
        
elif arg == "predictStructure":
    print("Moving files to predict Structure")
    shutil.copy2(os.path.join("/src/", inp.prediction.filename), tempDirectory)
    if not os.path.exists("/temp/wigners"):
    	os.system(f"cp -u /src/model.tar . \
                	&& tar -xf model.tar")
elif arg == "predictStructureDone":
    os.system('mkdir -p /src/prediction')
    os.system('find -wholename "*predictions*/*pred*.npy" -exec mv -t /src/prediction {} +')

elif arg == "calcReference":
    shutil.copy2(os.path.join("/src/", inp.prediction.filename), tempDirectory)
elif arg == "calcReferenceDone":
    os.system(f'cp -u -R reference /src')