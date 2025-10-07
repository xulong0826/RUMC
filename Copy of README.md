# TenGAN

A PyTorch implementation of "TenGAN: Pure Transformer Encoders Make an Efficient Discrete GAN for De Novo Molecular Generation."
The paper has been accepted by [AISTATS 2024](https://). ![Overview of TenGAN](https://github.com/naruto7283/TenGAN/blob/main/tengan_overview.png)

## Installation
Execute the following commands:
```
$ conda env create -n tengan_env -f env.yml
$ source activate tengan_env
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

  - RUMC on the ZINC dataset with drug-likeness as the optimized property:
  ```
parser.add_argument('--properties', type=str, default='druglikeness', help='the chemical property for molecular generation (druglikeness, solubility, or synthesizability)')

  $ python 1qm9.py

  $ python 1zinc.py

  $ python 1ablation.py
  ```
