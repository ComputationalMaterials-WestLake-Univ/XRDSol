 
# Equivariant Diffusion Crystal Structure Solution from Powder X-ray Diffraction (XRDSol)

The XRDSol is a equivariant diffusion model to solve crystalline structures by iteratively refining atom coordinates from powder XRD. XRDSol has solved thousands of entries that were previously unresolved or unlabeled. Compared to prior methods, the solving efficiency of XRDSol (0.6 seconds per solution) is more than 10,000 ‒100,000 times. Furthermore, we have addressed critical challenges in crystal structure solutions, such as light-element materials, natural minerals, and disordered structures. Additionally, XRDSol has corrected 39 long-standing erroneous crystal structures from the past several decades. 

Its main functionalities:

- correctly solve the crystal structures from PXRD.
- identify the incorrect crystal structure solutions.
- solve the disordered materials by "ordered approximations".


<p align="center">
  <img src="figures\overview of xrdsol.jpg" /> 
</p>

<p align="center">
  <img src="figures\solution_demo.gif" width="600">
</p>


## Table of Contents

- [Dependencies](#Dependencies)
- [Datasets](#datasets)
- [Usage](#usage)
- [Citation](#citation)
- [Contact](#contact)

## Dependencies

```
python==3.8.11
scipy == 1.10.1
wandb==0.10.33
torch==1.9.0
torch-geometric==1.7.2
pytorch_lightning==1.3.8
hydra-core==1.1.0
pymatgen==2023.5.10
```
Rename the .env.template file into .env and add your file path.
```
PROJECT_ROOT: your project' file path
HYDRA_JOBS: your hydra' file path
WABDB_DIR: your wandb' file path
```

## Datasets

We employed the MP-20 dataset (45231 structures) to train our equivariant diffusion model.


## Usage
### train:

To tain the MP-20 dataset, run the following command:

```
python xrdsol/run.py data=mp_20 expname=mp_20
```
### evalution:

To evalute the MP-20 test dataset, run the following command:

```
python scripts/evaluate.py --model_path<model_path> --dataset mp_20 --num_evals <run_times>
```
### solution:

To solve the crystal structures from PXRD, run the following command:

```
python scripts/solution.py --model_path<model_path> --dataset <dataset> --num_evals <run_times>
```


## Citation

Please consider citing the following paper if you find our code & data useful.

```
@article{XXX,
  title={Equivariant Diffusion Crystal Structure Solution from Powder X-ray Diffraction},
  author={Dongfang Yu, Zhewen Zhu, Yizhou Zhu},
  journal={XXX},
  year={2024}
}
```

## Contact

Please leave an issue or reach out to Yizhou Zhu (zhuyizhou AT westlake DOT edu DOT cn ) if you have any questions.
