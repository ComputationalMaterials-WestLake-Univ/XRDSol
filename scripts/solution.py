import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
import copy
import numpy as np

from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list
import torch.nn as nn

from pymatgen.io.cif import CifWriter
import chemparse
from p_tqdm import p_map
import os
import json
import pandas as pd
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from monty.json import MontyEncoder
from pymatgen.core import Element

from phasemapy.parser import ICDDEntry
from xrdsol.common.data_utils import get_voigt_xrd
import re

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

class SampleDataset(Dataset):

    def __init__(self, entries):
        super().__init__()
        self.entries = entries
        self.num_evals = len(entries)


    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        entry = self.entries[index]
        z = entry.z
        self.composition = entry.composition*z
        chem_list = []
        for elem in self.composition:          
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem.symbol)] * num_int)
        self.chem_list = chem_list
        wavelength = 1.54056  # Å
        initial_alphagamma = 0.03
        wider_x = np.arange(0, 90, 0.02)
        amp = get_voigt_xrd(wider_x, entry.data['xrd'][0], entry.data['xrd'][1], initial_alphagamma, wavelength)
        xrd_data = [wider_x, amp]
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
            lengths=torch.Tensor([entry.a, entry.b, entry.c]).view(1,-1),
            angles=torch.Tensor([entry.alpha, entry.beta, entry.gamma]).view(1,-1),
            xrd=torch.Tensor(xrd_data[1]),
        ) 


def diffusion(loader, model, num_evals, step_lr = 1e-5):
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()

        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        for eval_idx in range(num_evals):
            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr = step_lr)
            solution_batch = {key: value.cpu().tolist() for key, value in outputs.items()}
            with open (f'./solution_icdd_mutiple_full_occ/{idx}_{eval_idx}.json', 'w') as f:
                json.dump(solution_batch, f)

    return (
            batch_frac_coords  
    )



def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=True)

    if torch.cuda.is_available():
        model.to('cuda')
    print('Load the ICDD entries!')
    file_path = './data/unsolved_entries'
    pdfs_icdd = glob(f'{file_path}/*.xml')
    print('Cleaning the ICDD entries!')
    icdd_entries = [ICDDEntry.from_icdd_xml(pdf) for pdf in pdfs_icdd]
    entries = [d for d in icdd_entries if d.structure is None and d.z is not None and
                 d.z * d.composition.num_atoms <= 20]   
    entries = [d for d in entries if not re.findall(r'\d+\.\d+', (d.composition*d.z).hill_formula) and d.z != 0]
    
    print('Create dataset!')
    print(len(entries))
    test_set = SampleDataset(entries)
    print('Create DataLoader!')
    test_loader = DataLoader(test_set, batch_size=128,shuffle=False,num_workers=8,pin_memory=True)
    print('Evaluate the diffusion model.')


    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi'][args.dataset]


    start_time = time.time()
    _ = diffusion(test_loader, model, args.num_evals, step_lr)
    if args.label == '':
        diff_out_name = 'eval_diff.pt'
    else:
        diff_out_name = f'eval_diff_{args.label}.pt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', default='mp_20')
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--label', default='icdd')
    args = parser.parse_args()
    main(args)
