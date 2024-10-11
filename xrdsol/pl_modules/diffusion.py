import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from xrdsol.common.utils import PROJECT_ROOT
from xrdsol.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc, get_voigt_xrd)

from xrdsol.pl_modules.diff_utils import d_log_p_wrapped_normal

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from scripts.eval_utils import lattices_to_params_shape, get_crystals_list
from p_tqdm import p_map
from concurrent.futures import ThreadPoolExecutor

MAX_ATOMIC_NUM=100

def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None

def process_structure(structure):
    if structure:
        try:
            xrdcal = XRDCalculator()
            xrd = xrdcal.get_pattern(structure, scaled=False)
            wavelength = 1.54056  # Å
            initial_alphagamma = 0.03
            wider_x = np.arange(0, 90, 0.02)
            amp = get_voigt_xrd(wider_x, xrd.x, xrd.y, initial_alphagamma, wavelength)
            return amp
        except Exception as e:
            return np.zeros(4500).tolist()
    else:
        return np.zeros(4500).tolist()


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        ori_xrd = batch.xrd

        rand_x = torch.randn_like(frac_coords)
        scaled_factor = 1/(batch.lengths/batch.lengths.min(dim=1, keepdim=True)[0])
        scaled_factor_per_atom = scaled_factor.repeat_interleave(batch.num_atoms, dim=0)        
        rand_x = rand_x * scaled_factor_per_atom

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        if self.keep_coords:
            input_frac_coords = frac_coords

        input_lattice = lattices

        pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch, batch.xrd)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        crystal_list = get_crystals_list(pred_x, batch.atom_types, batch.lengths, batch.angles, batch.num_atoms)
        structure_list = []
        for s in crystal_list:
            strut = get_pymatgen(s)
            structure_list.append(strut)
        simu_xrd = p_map(process_structure,structure_list )




        simu_xrd = torch.tensor(simu_xrd, dtype=torch.float32, device=torch.device('cuda'))
        simu_xrd = simu_xrd.view(1, -1)
        simu_xrd = simu_xrd.squeeze()
        loss_xrd = F.mse_loss(simu_xrd/torch.max(simu_xrd), ori_xrd/torch.max(ori_xrd))*1000
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = ( self.hparams.cost_coord * loss_coord +
            loss_xrd)
        return {
            'loss' : loss,
            'loss_coord' : loss_coord,
            'loss_xrd': loss_xrd
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        scaled_factor = 1/(batch.lengths/batch.lengths.min(dim=1, keepdim=True)[0])
        scaled_factor_per_atom = scaled_factor.repeat_interleave(batch.num_atoms, dim=0)
        x_T = x_T * scaled_factor_per_atom
        if self.keep_coords:
            x_T = batch.frac_coords

        l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']


            if self.keep_coords:
                x_t = x_T

            l_t = l_T
            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_x = rand_x * scaled_factor_per_atom

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch, batch.xrd)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            # Predictor

            l_t_minus_05 = l_T
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_x = rand_x * scaled_factor_per_atom
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, batch.xrd)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = l_t
          
            

            traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices': l_t_minus_1
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_xrd  = output_dict['loss_xrd']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'xrd_loss': loss_xrd,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_xrd  = output_dict['loss_xrd']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_xrd_loss': loss_xrd,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss

    
