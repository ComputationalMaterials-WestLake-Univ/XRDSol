 
# Equivariant Diffusion Crystal Structure Solution from Powder X-ray Diffraction (XRDSol)

The AutoMapper is a equivariant diffusion model to solve crystalline structures by iteratively refining atom coordinates from powder XRD. XRDSol has solved thousands of entries that were previously unresolved or unlabeled. Compared to prior methods, the solving efficiency of XRDSol (0.6 seconds per solution) is more than 104‒105 times. Furthermore, we have addressed critical challenges in crystal structure solutions, such as light-element materials, natural minerals, and disordered structures. Additionally, XRDSol has corrected 39 long-standing unplausible crystal structures from the past several decades. 

Its main functionalities:

- correctly solve the crystal structures from PXRD.
- identify the incorrect crystal structure solutions.
- solve the disordered materials by "ordered approximations".


<p align="center">
  <img src="phasemapy/Overview of AutoMapper.svg" /> 
</p>



## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Citation](#citation)
- [Contact](#contact)

## Installation


Run the following command to install the environment:
```
pip install -r requirements.txt
```


## Datasets

We employed the MP-20 dataset (45231 structures) to train our equivariant diffusion model.

Find more about these datasets by going to our [Datasets_mp_20]("data/mp_20) page.

## Usage
### evalution:

To solve the V–Nb–Mn dataset, run the following command:

```
python phasemapy/scripts_V_Nb_Mn_O/solver_V-Nb-Mn.py
```
### solution:

To solve the Bi-Cu-V dataset, run the following command:

```
python phasemapy/scripts_Bi_Cu_V_O/solver_Bi_Cu_V.py
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
