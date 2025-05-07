import glob
import os, sys
import shutil

sys.path.insert(0, "/src")
from salted.sys_utils import ParseConfig
from salted import pack_model
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
    pack_model.build()
    os.system(f"cp *.salted /src")
        
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