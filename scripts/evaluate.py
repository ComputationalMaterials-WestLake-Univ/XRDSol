import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

import copy

import numpy as np


def diffusion(loader, model, num_evals, step_lr = 1e-5):
    import json
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        for eval_idx in range(num_evals):
            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, _ = model.sample(batch, step_lr = step_lr)
            solution_batch = {key: value.cpu().tolist() for key, value in outputs.items()}
            with open (f'./solution_mp20_25runs/{idx}_{eval_idx}.json', 'w') as f:
                json.dump(solution_batch, f)
    return (batch_frac_coords)



def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=True)

    if torch.cuda.is_available():
        model.to('cuda')


    print('Evaluate the diffusion model.')

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi'][args.dataset]


    start_time = time.time()

    _ = diffusion( test_loader, model, args.num_evals, step_lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    main(args)
