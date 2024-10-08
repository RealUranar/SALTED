import h5py
import glob, os
import numpy as np

from salted.sys_utils import ParseConfig
   
def pack_averages(h5File, path):
    grp = h5File.create_group('averages')
    files = glob.glob(os.path.join(path,"averages",'av*.npy'))
    for file in files:
        data = np.load(file)
        grp.create_dataset(os.path.basename(file).split('.')[0].split("_")[-1], data=data)

def pack_wigners(h5File, path, inp):
    grp = h5File.create_group('wigners')
    files = glob.glob(os.path.join(path,"wigners",f'wigner_lam-*_lmax1-{inp.descriptor.rep1.nang}_lmax2-{inp.descriptor.rep2.nang}.dat'))
    for file in files:
        data = np.loadtxt(file)
        grp.create_dataset(os.path.basename(file).split('_')[1], data=data)

def pack_fps(h5File, path, inp):
    grp = h5File.create_group('fps')
    files = glob.glob(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'fps{inp.descriptor.sparsify.ncut}-*.npy'))
    for file in files:
        data = np.load(file)
        lam = os.path.basename(file).split('-')[-1].split('.')[0]
        grp.create_dataset(f"lam-{lam}", data=data)

def pack_FEAT_projectors(h5File, path, inp):
    files = glob.glob(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'FEAT_M-{inp.gpr.Menv}.h5')) + glob.glob(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'projector_M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}.h5'))
    for file in files:
        print(file)
        with h5py.File(file, 'r') as temp:
            for key in temp.keys():
                h5File.copy(temp[key], key)

def pack_weights(h5File, path, inp):
    file = glob.glob(os.path.join(path, f"regrdir_{inp.salted.saltedname}", f"M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}", f'weights_N{int(inp.gpr.Ntrain*inp.gpr.trainfrac)}_reg*'))[0]
    data = np.load(file)
    h5File.create_dataset("weights", data=data)

def pack_model_info(h5File, inp):
    inputs = {
        "averages": inp.system.average,
        "field": inp.system.field,
        "sparsify": inp.descriptor.sparsify.ncut > 0,
        "ncut": inp.descriptor.sparsify.ncut,
        "species": inp.system.species,
        "rcut1": inp.descriptor.rep1.rcut,
        "rcut2": inp.descriptor.rep2.rcut,
        "nang1": inp.descriptor.rep1.nang,
        "nang2": inp.descriptor.rep2.nang,
        "nrad1": inp.descriptor.rep1.nrad,
        "nrad2": inp.descriptor.rep2.nrad,
        "sig1": inp.descriptor.rep1.sig,
        "sig2": inp.descriptor.rep2.sig,
        "neighspe1": inp.descriptor.rep1.neighspe,
        "neighspe2": inp.descriptor.rep2.neighspe,
        "zeta": inp.gpr.z,
        "Menv": inp.gpr.Menv,
        "Ntrain": inp.gpr.Ntrain,
        "trainfrac": inp.gpr.trainfrac,
        "dfbasis": inp.qm.dfbasis
    }
    grp = h5File.create_group('input')
    for key, value in inputs.items():
        grp.create_dataset(key, data=value)
    
def build():
    inp = ParseConfig().parse_input()
    path = inp.salted.saltedpath
    
    with h5py.File(f'{inp.salted.saltedname}.h5', 'w') as f:
        pack_averages(f, path)
        pack_wigners(f, path, inp)
        pack_fps(f, path, inp)
        pack_FEAT_projectors(f, path, inp)
        pack_weights(f, path, inp)
        pack_model_info(f, inp)

if __name__ == "__main__":
    build()