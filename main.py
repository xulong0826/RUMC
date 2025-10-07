import os
import csv
import copy
import glob
import time
import torch
import heapq
import pickle
import argparse
import warnings
import numpy as np 
import pandas as pd
from utils import *
from rdkit import Chem
import pytorch_lightning
from rdkit import rdBase
from mol_metrics import *
from rdkit.Chem import Draw
from rollout import Rollout, OwnModel
from discriminator import DiscriminatorModel
from generator import GeneratorModel, GenSampler
from data_iter import GenDataLoader, DisDataLoader
import datetime
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

# ===========================
# Default settings
parser = argparse.ArgumentParser()
# General hyperparameters
parser.add_argument('--dataset_name', type=str, default='ZINC', help='use QM9 or ZINC dataset')
parser.add_argument('--max_len', type=int, default=70, help='the max length of SMILES data: 60 for QM9, 70 for ZINC')
parser.add_argument('--batch_size', type=int, default=64, help='the batch size for both generator and discriminator')

# ===========================
# Generator
parser.add_argument('--gen_pretrain', action='store_true', help='whether pretrain the dataset')
parser.add_argument('--generated_num', type=int, default=10000, help='generate size of the negative file: 5000 for QM9, 10000 for ZINC')
parser.add_argument('--gen_train_size', type=int, default=9600, help='the size of training data: 4800 for QM9, 9600 for ZINC')
parser.add_argument('--gen_num_encoder_layers', type=int, default=4, help='the number of transformer encoder layers')
parser.add_argument('--gen_d_model', type=int, default=128, help='the dimension of the embedding')
parser.add_argument('--gen_dim_feedforward', type=int, default=1024, help='the dimension of the feedforward layer')
parser.add_argument('--gen_num_heads', type=int, default=4, help='the number of heads')
parser.add_argument('--gen_max_lr', type=float, default=8e-4, help='the maximum learning rate')
parser.add_argument('--gen_dropout', type=float, default=0.1, help='the dropout probability')
parser.add_argument('--gen_epochs', type=int, default=150, help='the pretraining epochs')

# ===========================
# Discriminator
parser.add_argument('--dis_pretrain', action='store_true', help='whether pretraining the discriminator')
parser.add_argument('--dis_wgan', action='store_true', help='whether applying the WGAN')
parser.add_argument('--dis_minibatch', action='store_true', help='whether applying the mini-batch discrimination')
parser.add_argument('--dis_num_encoder_layers', type=int, default=4, help='the number of transformer encoder layers of the discriminator')
parser.add_argument('--dis_d_model', type=int, default=100, help='the dimension of the embedding of the discriminator')
parser.add_argument('--dis_num_heads', type=int, default=5, help='the number of heads of the discriminator')
parser.add_argument('--dis_epochs', type=int, default=10, help='the pretraining epochs for the discriminator')
parser.add_argument('--dis_feed_forward', type=int, default=200, help='the dimension of the feedforward layer')
parser.add_argument('--dis_dropout', type=float, default=0.25, help='the dropout probability')

# ===========================
# Adversarial training
parser.add_argument('--adversarial_train', action='store_true', help='whether adversarial trian TenGAN or Ten(W)GAN')
parser.add_argument('--update_rate', type=float, default=0.8, help='the update rate')
parser.add_argument('--properties', type=str, default='druglikeness', help='the chemical property for molecular generation (druglikeness, solubility, or synthesizability)')
parser.add_argument('--dis_lambda', type=float, default=0.5, help='the tradeoff between RL and GAN. If 0: NAIVE elif 1: SeqGAN else: TenGAN')
parser.add_argument('--adv_lr', type=float, default=8e-5, help='the learning rate for the GAN or WGAN')
parser.add_argument('--save_name', type=int, default=66, help='the name of the loaded model')
parser.add_argument('--roll_num', type=int, default=8, help='the rollout times for Monte Carlo tree search: 16 for QM9, 8 for ZINC')
parser.add_argument('--adv_epochs', type=int, default=100, help='the adverarial training epochs for TenGAN or Ten(W)GAN')

# 添加这些命令行参数
parser.add_argument('--use_mc_dropout', action='store_true', help='whether to use MC Dropout for uncertainty estimation')
parser.add_argument('--use_reward_norm', action='store_true', help='whether to use adaptive reward normalization')
parser.add_argument('--mc_samples', type=int, default=8, help='number of MC samples for uncertainty estimation')
parser.add_argument('--buffer_capacity', type=int, default=10000, help='buffer_capacity: 10000 for QM9, 30000 for ZINC')
parser.add_argument('--replay_prob', type=float, default=0.1, help='the probability of using replay buffer for generator training')

args = parser.parse_known_args()[0]

# ===========================
# Other model paths
POSITIVE_FILE = 'dataset/' + args.dataset_name + '.csv' # Save the real / original SMILES data
NEGATIVE_FILE = 'res/generated_smiles_' + args.dataset_name + '.csv' # Save the generated SMILES data

if args.dis_lambda == 0:
    MODEL_NAME = 'Naive'
elif args.dis_lambda == 1:
    MODEL_NAME = 'SeqGAN'
else:
    MODEL_NAME = 'TenGAN_' + str(args.dis_lambda)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # cuda:0,1,2,3
GPUS = [1]

if args.dis_wgan:
    DIS_MAX_LR = 8e-4
else:
    DIS_MAX_LR = 8e-7

G_STEP = 1
if args.dis_wgan:
    D_STEP = 3
else:
    D_STEP = 1

# File paths
# PATHS = 'res/save_models/' + args.dataset_name + '/' + MODEL_NAME + '/rollout_' + str(args.roll_num) + '/' + '/batch_' + str(args.batch_size) + '/' + args.properties
# if not os.path.exists(PATHS):
#     os.makedirs(PATHS)

# # Save the pre-trained generator and discriminator
# G_PRETRAINED_MODEL = PATHS + '/g_pretrained.pkl' 
# D_PRETRAINED_MODEL = PATHS + '/d_pretrained.pkl'
# PROPERTY_FILE = PATHS + '/trained_results.csv'
exp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
PATHS = 'res/save_models/' + args.dataset_name + '/' + MODEL_NAME + '/rollout_' + str(args.roll_num) + '/' + '/batch_' + str(args.batch_size) + '/' + args.properties + '/' + exp_tag
if not os.path.exists(PATHS):
    os.makedirs(PATHS)

G_PRETRAINED_MODEL = PATHS + '/g_pretrained.pkl' 
D_PRETRAINED_MODEL = PATHS + '/d_pretrained.pkl'
PROPERTY_FILE = PATHS + '/trained_results.csv'
NEGATIVE_FILE = PATHS + '/generated_smiles_' + args.dataset_name + '.csv'

# Save adversarial training information
if args.adversarial_train:
    with open(PROPERTY_FILE, 'a+') as wf:
        wf.truncate(0)
        wf.write('Epoch,QED_Mean,QED_Std,QED_Min,QED_Max,SA_Mean,SA_Std,SA_Min,SA_Max,LogP_Mean,LogP_Std,LogP_Min,LogP_Max,Validity,Uniqueness,Novelty,Diversity,Time\n')
TenGAN_G_MODEL = PATHS + '/Epoch_' + str(args.save_name) + '_gen.pkl'
TenGAN_D_MODEL = PATHS + '/Epoch_' + str(args.save_name) + '_dis.pkl'

args = parser.parse_args()

# ===========================
print('\n\n\nVocabulary Information:')
print('==================================================================') 
tokenizer = Tokenizer()
tokenizer.build_vocab()
print(tokenizer.char_to_int)

# Save all hyperparameters
with open(PATHS+'/hyperparameters.csv', 'a+') as hp:
    # Clean the hyperparameters file
    hp.truncate(0)

    params = {}
    print('\n\nParameter Information:')
    print('==================================================================')
    params['POSITIVE_FILE'] = POSITIVE_FILE
    params['NEGATIVE_FILE'] = NEGATIVE_FILE
    params['G_PRETRAINED_MODEL'] = G_PRETRAINED_MODEL
    params['D_PRETRAINED_MODEL'] = D_PRETRAINED_MODEL
    params['PROPERTY_FILE'] = PROPERTY_FILE
    params['BATCH_SIZE'] = args.batch_size
    params['MAX_LEN'] = args.max_len
    params['VOCAB_SIZE'] = len(tokenizer.char_to_int)
    params['DEVICE'] = DEVICE
    params['GPUS'] = GPUS
    for param in params:
        string = param + ' ' * (25 - len(param))
        print('{}:   {}'.format(string, params[param]))
        hp.write('{}\t{}\n'.format(str(param), str(params[param])))
    print('\n')

    params = {}
    params['GEN_PRETRAIN'] = args.gen_pretrain
    params['GENERATED_NUM'] = args.generated_num
    params['GEN_TRAIN_SIZE'] = args.gen_train_size
    params['GEN_NUM_ENCODER_LAYERS'] = args.gen_num_encoder_layers
    params['GEN_DIM_FEEDFORWARD'] = args.gen_dim_feedforward
    params['GEN_D_MODEL'] = args.gen_d_model
    params['GEN_NUM_HEADS'] = args.gen_num_heads
    params['GEN_MAX_LR'] = args.gen_max_lr
    params['GEN_DROPOUT'] = args.gen_dropout
    params['GEN_EPOCHS'] = args.gen_epochs
    for param in params:
        string = param + ' ' * (25 - len(param))
        print('{}:   {}'.format(string, params[param]))
        hp.write('{}\t{}\n'.format(str(param), str(params[param])))
    print('\n')

    params = {}
    params['DIS_PRETRAIN'] = args.dis_pretrain
    params['DIS_WGAN'] = args.dis_wgan
    params['DIS_MINIBATCH'] = args.dis_minibatch
    params['DIS_NUM_ENCODER_LAYERS'] = args.dis_num_encoder_layers
    params['DIS_D_MODEL'] = args.dis_d_model
    params['DIS_NUM_HEADS'] = args.dis_num_heads
    params['DIS_MAX_LR'] = DIS_MAX_LR
    params['DIS_EPOCHS'] = args.dis_epochs
    params['DIS_FEED_FORWARD'] = args.dis_feed_forward
    params['DIS_DROPOUT'] = args.dis_dropout
    for param in params:
        string = param + ' ' * (25 - len(param))
        print('{}:   {}'.format(string, params[param]))
        hp.write('{}\t{}\n'.format(str(param), str(params[param])))
    print('\n')

    params = {}
    params['ADVERSARIAL_TRAIN'] = args.adversarial_train
    params['PROPERTIES'] = args.properties
    params['DIS_LAMBDA'] = args.dis_lambda
    params['MODEL_NAME'] = MODEL_NAME
    params['UPDATE_RATE'] = args.update_rate
    params['ADV_LR'] = args.adv_lr
    params['G_STEP'] = G_STEP
    params['D_STEP'] = D_STEP
    params['ADV_EPOCHS'] = args.adv_epochs
    # params['ROLL_NUM'] = args.roll_num
    params['ROLL_NUM'] = args.mc_samples if args.use_mc_dropout else args.roll_num
    for param in params:
        string = param + ' ' * (25 - len(param))
        print('{}:   {}'.format(string, params[param]))
        hp.write('{}\t{}\n'.format(str(param), str(params[param])))
    print('==================================================================')

# ============================================================================
def evaluation(generated_smiles, gen_data_loader, time=None, epoch=None):
    generated_mols = np.array([Chem.MolFromSmiles(s) for s in generated_smiles if len(s.strip())])
    if len(generated_mols) == 0:
        print('No SMILES data is generated, please pre-train the generator again!')
        return
    else:
        valid_smiles = []
        for mol in generated_mols:
            if mol != None and mol.GetNumAtoms() > 1 and Chem.MolToSmiles(mol) != ' ':
                valid_smiles.append(Chem.MolToSmiles(mol))   
        unique_smiles = list(set(valid_smiles))
        novel_smiles = []
        # Keep the unique order for the POSITIVE dataset
        train_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(sm)) for sm in gen_data_loader.train_data]
        for smile in unique_smiles:
            if smile not in train_smiles:
                novel_smiles.append(smile)

        validity = len(valid_smiles)/len(generated_mols)
        if len(valid_smiles) == 0:
            valid_smiles.append('c1ccccc1')
        if len(unique_smiles) == 0:
            unique_smiles.append('c1ccccc1')
        uniqueness = len(unique_smiles)/len(valid_smiles)
        novelty = len(novel_smiles)/len(unique_smiles)
        # Diversity results
        diversity = batch_diversity(novel_smiles)

        print('\nResults Report:')
        print('*'*80)
        print("Total Mols:   {}".format(len(generated_mols)))
        print("Validity:     {}    ({:.2f}%)".format(len(valid_smiles), validity*100))
        print("Uniqueness:   {}    ({:.2f}%)".format(len(unique_smiles), uniqueness*100))
        print("Novelty:      {}    ({:.2f}%)".format(len(novel_smiles), novelty*100))
        print("Diversity:    {:.2f}".format(diversity))
        print('\n')
        print('Samples of Novel SMILES:')
        if len(novel_smiles) >= 5:
            for i in range(5):
                print(novel_smiles[i])
        else:
            for i in range(len(novel_smiles)):
                print(novel_smiles[i])
        print('\n')  
        # Compute the property scores of novel smiles   
        if len(novel_smiles):
            # 计算所有属性
            qed_vals = reward_fn('druglikeness', novel_smiles)
            sa_vals = reward_fn('synthesizability', novel_smiles)
            logp_vals = reward_fn('solubility', novel_smiles)
            # 统计
            qed_mean, qed_std, qed_min, qed_max = np.mean(qed_vals), np.std(qed_vals), np.min(qed_vals), np.max(qed_vals)
            sa_mean, sa_std, sa_min, sa_max = np.mean(sa_vals), np.std(sa_vals), np.min(sa_vals), np.max(sa_vals)
            logp_mean, logp_std, logp_min, logp_max = np.mean(logp_vals), np.std(logp_vals), np.min(logp_vals), np.max(logp_vals)
            print('[QED]: [Mean: {:.3f} STD: {:.3f} MIN: {:.3f} MAX: {:.3f}]'.format(qed_mean, qed_std, qed_min, qed_max))
            print('[SA]:  [Mean: {:.3f} STD: {:.3f} MIN: {:.3f} MAX: {:.3f}]'.format(sa_mean, sa_std, sa_min, sa_max))
            print('[LogP]:[Mean: {:.3f} STD: {:.3f} MIN: {:.3f} MAX: {:.3f}]'.format(logp_mean, logp_std, logp_min, logp_max))
            # 保存所有属性到文件
            if epoch is not None and time is not None:
                with open(PROPERTY_FILE, 'a+') as wf:
                    wf.write('{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.5f},{:.1f}\n'.format(
                        epoch+1,
                        qed_mean, qed_std, qed_min, qed_max,
                        sa_mean, sa_std, sa_min, sa_max,
                        logp_mean, logp_std, logp_min, logp_max,
                        validity, uniqueness, novelty, diversity, time
                    ))
        else:
            print('No novel SMILES generated!')
        print('*'*80)
        print('\n')
    return validity, uniqueness, novelty, diversity
        
def pg_loss(probs, targets, rewards):
    """
    probs: [batch_size * max_len, vocab_size], logP of the gen output
    targets: [batch_size * max_len], integers
    rewards: [batch_size, seq_len]
    """
    one_hot = torch.zeros(probs.size(), dtype=torch.bool).to(DEVICE) # [batch_size * max_len, vocab_size] with all 'False'
    # Set 1 for the token (vocab_size) of each row of one_hot
    one_hot.scatter_(1, targets.data.view(-1, 1), 1) # [batch_size * max_len, vocab_size] 
    # Select the values in probs according to one_hot
    loss = torch.masked_select(probs, one_hot) # [batch_size * max_len]
    loss = loss * rewards.contiguous().view(-1) # [batch_size * max_len]
    loss = - torch.sum(loss)
    return loss

# ============================================================================
def main():
    # ===========================
    # For reproducing experiments
    start_time = time.time()
    print('\n\n\nStart time is {}\n\n\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # Apply the seed to reproduct the results
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # ===========================
    # Generator objects definition
    gen_data_loader = GenDataLoader(POSITIVE_FILE, args.gen_train_size, args.batch_size)
    gen_data_loader.setup()
    train_loader = gen_data_loader.train_dataloader()
    steps_per_epoch = len(train_loader)
    gen = GeneratorModel(
        n_tokens = gen_data_loader.tokenizer.n_tokens, 
        num_encoder_layers = args.gen_num_encoder_layers, 
        dim_feedforward = args.gen_dim_feedforward,
        d_model = args.gen_d_model, 
        nhead = args.gen_num_heads, 
        dropout = args.gen_dropout,
        epochs = args.gen_epochs, 
        max_lr = args.gen_max_lr,
        steps_per_epoch = steps_per_epoch)
    gen_trainer = pytorch_lightning.Trainer(
        max_epochs = args.gen_epochs,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = GPUS if torch.cuda.is_available() else 1,
        gradient_clip_val = 5.0,
        gradient_clip_algorithm = 'norm')

    # Pre-train the generator
    if args.gen_pretrain:
        print("\n\nPre-train Generator...")
        gen_trainer.fit(gen, gen_data_loader)
        # Pre-train time cost
        print('Generator Pre-train Time:\033[1;35m {:.2f}\033[0m hours'.format((time.time() - start_time) / 3600.))
        # Save the pre-trained generator model into file
        torch.save(gen.state_dict(), G_PRETRAINED_MODEL)
    else:
        # Load the pre-trained generator
        print("\n\nLoad Pre-trained Generator.")
        gen.load_state_dict(torch.load(G_PRETRAINED_MODEL))
    gen.to(DEVICE)
    # Sample the generated data
    print('Generating {} samples...'.format(args.generated_num))
    sampler = GenSampler(
        gen, 
        gen_data_loader.tokenizer, 
        args.batch_size, 
        args.max_len, 
        reward_fn=reward_fn, 
        buffer_capacity=args.buffer_capacity,
        replay_prob=args.replay_prob  # 新增参数
    )
    # generated_smiles = sampler.sample_multi(args.generated_num, NEGATIVE_FILE)
    generated_smiles = sampler.sample_multi(args.generated_num, NEGATIVE_FILE, 
                                       use_mc_dropout=args.use_mc_dropout, use_replay=(args.use_reward_norm))
    validity, uniqueness, novelty, diversity = evaluation(generated_smiles, gen_data_loader)
    
    # ===========================
    # Discriminator objects definition
    dis_data_loader = DisDataLoader(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
    dis_data_loader.setup()
    dis_train_loader = dis_data_loader.train_dataloader()
    dis_steps_per_epoch = len(dis_train_loader)
    dis = DiscriminatorModel(
        n_tokens = dis_data_loader.tokenizer.n_tokens, 
        d_model = args.dis_d_model,
        nhead = args.dis_num_heads, 
        num_encoder_layers = args.dis_num_encoder_layers,
        dim_feedforward = args.dis_feed_forward,
        dropout = args.dis_dropout,  
        max_lr = DIS_MAX_LR,
        epochs = args.dis_epochs,
        pad_token = tokenizer.char_to_int[tokenizer.pad],
        dis_wgan = args.dis_wgan,
        steps_per_epoch = dis_steps_per_epoch,
        minibatch = args.dis_minibatch)
    dis_trainer = pytorch_lightning.Trainer(
        max_epochs = args.dis_epochs,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = GPUS if torch.cuda.is_available() else 1,
        gradient_clip_val = 1.0,
        gradient_clip_algorithm = 'value')

    # Pre-train the discriminator
    if args.dis_pretrain:
        print("\n\nPre-train Discriminator...")
        dis_trainer.fit(dis, dis_data_loader)
        # Pre-train time cost
        print('Discriminator Pre-train Time:\033[1;35m {:.2f}\033[0m hours'.format((time.time() - start_time) / 3600.))
        # Save the pre-trained discriminator model into file
        torch.save(dis.state_dict(), D_PRETRAINED_MODEL)
    else:
        # Load the pre-trained discirminator
        print("\n\nLoad Pre-trained Discriminator.")
        dis.load_state_dict(torch.load(D_PRETRAINED_MODEL))
    dis.to(DEVICE)

    # ===========================
    # Adversarial objects definition
    roll_own_model = OwnModel( # Create deepcopy gen model for rollout sampling
        n_tokens = gen_data_loader.tokenizer.n_tokens,
        d_model = args.gen_d_model, 
        nhead = args.gen_num_heads, 
        num_encoder_layers = args.gen_num_encoder_layers, 
        dim_feedforward = args.gen_dim_feedforward,
        dropout = args.gen_dropout) 
    dic = {}
    # Deep copy gen to roll_own_model
    gen.requires_grad = False
    for name, param in gen.named_parameters():
        dic[name] = copy.deepcopy(param.data) # # Deepcopy: if model's parameter changes, parameters in own_model do not change 
    for name, param in roll_own_model.named_parameters():
        param.data = dic[name] 
    roll_own_model.to(DEVICE)
    gen.requires_grad = True
    adv_trainer = pytorch_lightning.Trainer(
        max_epochs = D_STEP,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = GPUS if torch.cuda.is_available() else 1)
    pg_optimizer = torch.optim.Adam(params = gen.parameters(), lr=args.adv_lr)
    rollout = Rollout(gen, roll_own_model, tokenizer, args.update_rate, DEVICE)

    # Adversarial training
    if args.adversarial_train:
        print("\n\nAdversarial Training...")
        for epoch in range(args.adv_epochs): 
            rollsampler = GenSampler(rollout.own_model, gen_data_loader.tokenizer, args.batch_size, args.max_len, reward_fn=reward_fn, buffer_capacity=args.buffer_capacity, replay_prob=args.replay_prob)
            for g_step in range(G_STEP):
                # Sampling a batch of samples
                samples = sampler.sample()
                encoded = [torch.tensor(tokenizer.encode(s)) for s in samples] # Within the start and end token
                encoded = torch.nn.utils.rnn.pad_sequence(encoded).squeeze().to(DEVICE) # [max_len, batch_size]
                gen_pred = gen.forward(encoded[:-1]) # [max_len, batch_size, vocab_size]
                gen_pred = gen_pred.transpose(0, 1) # [batch_size, max_len, vocab_size]
                gen_pred = gen_pred.contiguous().view(-1, gen_pred.size(-1)) # [batch_size * max_len, vocab_size]
                gen_pred = torch.nn.functional.log_softmax(gen_pred, dim=1) 
                targets = encoded[1:].transpose(0, 1).contiguous().view(-1,) # [batch_size * max_len]
                # Calculate the rewards of each token in a batch  
                rewards = rollout.get_reward(samples, rollsampler, args.roll_num, dis, args.dis_lambda, args.properties) # [batch_size, seq_len-init]
                rewards = torch.tensor(rewards).to(DEVICE)
                # Compute policy gradient loss
                loss = pg_loss(gen_pred, targets, rewards)  
                print('\n\n\n\033[1;35mEpoch {}\033[0m / {}, G_STEP {} / {}, PG_Loss: {:.3f}'.format(epoch+1, args.adv_epochs, g_step+1, G_STEP, loss))  
                pg_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 5, norm_type=2)
                pg_optimizer.step() 
            # Update models
            rollout.update_params()
            # Save models
            torch.save(gen.state_dict(), PATHS + '/Epoch_' + str(epoch+1) + '_gen.pkl')
            if args.dis_lambda:
                torch.save(dis.state_dict(), PATHS + '/Epoch_' + str(epoch+1) + '_dis.pkl')
            # Generate Samples
            print('Generating {} samples...'.format(args.generated_num))
            sampler = GenSampler(gen, gen_data_loader.tokenizer, args.batch_size, args.max_len, reward_fn=reward_fn, buffer_capacity=args.buffer_capacity, replay_prob=args.replay_prob)
            generated_smiles = sampler.sample_multi(args.generated_num, NEGATIVE_FILE)
            current_time = (time.time() - start_time) / 3600.
            print('\nTotal Computational Time: \033[1;35m {:.2f} \033[0m hours.'.format(current_time))
            validity, uniqueness, novelty, diversity = evaluation(generated_smiles, gen_data_loader, current_time, epoch)
            # Train discriminator
            if args.dis_lambda:
                for i in range(D_STEP):
                    # Update the dis_data_loader
                    dis_data_loader.setup() 
                    adv_trainer.fit(dis, dis_data_loader)
    else:
        # Load the trained TenGAN
        if not os.path.exists(TenGAN_G_MODEL):
            print('\n\nTenGAN Generator path does NOT exist: ' + TenGAN_G_MODEL)
            return
        else:
            print("\n\nLoad TenGAN Generator: {}".format(TenGAN_G_MODEL))
            gen.load_state_dict(torch.load(TenGAN_G_MODEL))
            gen.to(DEVICE)
            print('\n\nGenerating {} samples...'.format(args.generated_num))  
            sampler = GenSampler(gen, gen_data_loader.tokenizer, args.batch_size, args.max_len, reward_fn=reward_fn, buffer_capacity=args.buffer_capacity, replay_prob=args.replay_prob)
            generated_smiles = sampler.sample_multi(args.generated_num, NEGATIVE_FILE)
            validity, uniqueness, novelty, diversity = evaluation(generated_smiles, gen_data_loader)

        if args.dis_lambda:
            if not os.path.exists(TenGAN_D_MODEL):
                print('\n\nTenGAN Discriminator path does NOT exist: ' + TenGAN_D_MODEL)
                return
            else:
                print("Load TenGAN Discriminator: {}\n\n".format(TenGAN_D_MODEL))
                dis.load_state_dict(torch.load(TenGAN_D_MODEL))
                dis.to(DEVICE)

    # Show Top-12 molecules
    if not os.path.isfile(NEGATIVE_FILE):
        print('Generated dataset does NOT exist!\n')
    else:
        print('Top-12 Molecules of [{}]:'.format(args.properties))
        top_mols = top_mols_show(NEGATIVE_FILE, args.properties)
        img = Draw.MolsToGridImage(top_mols[:], molsPerRow = 3, subImgSize = (1000, 1000), legends = ['' for x in top_mols])
        if args.dis_wgan:
            img.save('res/top_12_w.pdf')
        else:
            img.save('res/top_12.pdf')
        print('*'*80)

    # Figure out distributions
    all_files = glob.glob('res/*.csv')
    print('\n\nFile names for drawing distributions:', all_files)
    # Notice: Add '_w.csv' to the name of WGAN file 
    if len(all_files) == 2: 
        distribution(POSITIVE_FILE, all_files[0], all_files[1])
    else:
        print('Distributions are not generated.')
    print('*'*80)
        

#============================================================================
if __name__ == '__main__':
    main()




























































































