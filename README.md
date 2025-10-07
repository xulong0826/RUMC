# RUMC

A PyTorch implementation of "RUMC: Reward-Guided Monte Carlo Sampling with Uncertainty for De Novo Molecular Generation."
The paper has been accepted by NTCI 2025. 

# Acknowledgements
We thank the authors of TenGAN: Pure Transformer Encoders Make an Efficient Discrete GAN for De Novo Molecular Generation for releasing their code. The code in this repository is based on their source code release [TenGAN](https://github.com/naruto7283/TenGAN). If you find this code useful, please consider citing their work.

# Framework
![Overview of RUMC](https://github.com/xulong0826/RUMC/blob/main/RUMC_overview.png)

## Installation
Execute the following commands:
```
$ conda env create -n rumc -f env.yml
$ source activate rumc
```

## File Description

  - **dataset:** contains the training datasets. Each dataset contains only one column of SMILES strings.
	  - QM9.csv
	  - ZINC.csv
   
  - **res:** all generated datasets, saved models, and experimental results are saved in this folder.
	- save_models: all training results, pre-trained and trained filler and discriminator models are saved in this folder.

	- main.py: definite all hyper-parameters, pretraining of the generator, pretraining of the discriminator, adversarial training of the TenGAN and Ten(W)GAN.
		
	- mol_metrics.py: definite the vocabulary, tokenization of SMILES strings, and all the objective functions of the chemical properties.	

	- data_iter.py: load data for the generator and discriminator.

	- generator.py: definite the generator.

	- discriminator.py: definite the discriminator.

	- rollout.py: definite the Monte Carlo method.

	- utils.py: definite the performance evaluation methods of the generated molecules, such as the validity, uniqueness, novelty, and diversity. 

## Available Chemical Properties at Present:
	- solubility
	- druglikeness
	- synthesizability
 
## Experimental Reproduction

```
  $ python 1qm9.py

  $ python 1zinc.py

  $ python 1ablation.py
```
