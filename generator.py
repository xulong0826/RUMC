import math
import torch
from tqdm import tqdm
import pytorch_lightning
from pytorch_lightning import LightningModule
import numpy as np  # 添加这行
import random      # 添加这行


# ============================================================================
# PositionalEncoding needs to be definited manually, even if the transformer model is called from torch.nn
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
            d_model: the embedded dimension  
            max_len: the maximum length of sequences
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # PosEncoder(pos, 2i) = sin(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # PosEncoder(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: the sequence fed to the positional encoder model with the shape [sequence length, batch size, embedded dim].
        """
        x = x + self.pe[:x.size(0), :] # [max_len, batch_size, d_model] + [max_len, 1, d_model]
        return self.dropout(x)


# ============================================================================
# Definition of the Generator model
class GeneratorModel(LightningModule):
    
    def __init__(
        self,
        n_tokens, # vocabulary size
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        max_length = 1000,
        max_lr = 1e-3,
        epochs = 50,
        steps_per_epoch = None,
    ):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.max_length = max_length
        self.max_lr = max_lr
        self.epochs = epochs
        self.setup_layers()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr, 
            # total_steps=None, 
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs, 
            pct_start=6/self.epochs, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85, 
            max_momentum=0.95,
            div_factor=1e3, 
            final_div_factor=1e3, 
            last_epoch=-1)
        
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        return [optimizer], [scheduler]
    
    def setup_layers(self):
        self.embedding = torch.nn.Embedding(self.n_tokens, self.d_model)
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation)
        encoder_norm = torch.nn.LayerNorm(self.d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)
        self.fc_out = torch.nn.Linear(self.d_model, self.n_tokens)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # Define lower triangular square matrix with dim=sz
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, features): # features.shape[0] = max_len
        mask = self._generate_square_subsequent_mask(features.shape[0]).to(self.device) # mask: [max_len, max_len]
        embedded = self.embedding(features)
        positional_encoded = self.positional_encoder(embedded)
        encoded = self.encoder(positional_encoded, mask=mask)
        out_2= self.fc_out(encoded) # [max_len, batch_size, vocab_size]
        return out_2
    
    def step(self, batch):
        batch = batch.to(self.device)
        prediction = self.forward(batch[:-1]) # Skipping the last char
        loss = torch.nn.functional.cross_entropy(prediction.transpose(0,1).transpose(1,2), batch[1:].transpose(0,1)) # Skipping the first char
        return loss
    
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.step(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

# ============================================================================
# 新增：奖励归一化与经验回放缓冲区
class RewardBuffer:
    def __init__(self, capacity=10000, device='cpu'):
        self.buffer = []
        self.capacity = capacity
        self.device = device
        self.rewards = torch.tensor([], dtype=torch.float32, device=self.device)
        self.mean_reward = torch.tensor(0.0, device=self.device)
        self.std_reward = torch.tensor(1.0, device=self.device)

    def add(self, smiles, reward):
        if not isinstance(smiles, list):
            smiles = [smiles]
        if not isinstance(reward, list):
            reward = [reward]
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.rewards = torch.cat([self.rewards, reward_tensor])
        self.mean_reward = torch.mean(self.rewards)
        self.std_reward = torch.std(self.rewards) + 1e-8
        normalized_rewards = self.normalize_reward(reward_tensor)
        # 新增：去重，只保留独特分子
        existing_smiles = set(s for s, _ in self.buffer)
        for s, nr in zip(smiles, normalized_rewards):
            if s in existing_smiles:
                continue  # 跳过重复分子
            if len(self.buffer) >= self.capacity:
                buffer_rewards = torch.tensor([r for _, r in self.buffer], device=self.device)
                min_idx = torch.argmin(buffer_rewards)
                if nr > buffer_rewards[min_idx]:
                    self.buffer[min_idx] = (s, nr.item())
            else:
                self.buffer.append((s, nr.item()))
                existing_smiles.add(s)

    def normalize_reward(self, reward):
        std = max(self.std_reward.item(), 1e-8)
        mean = self.mean_reward.item()
        return (reward - mean) / std

    def sample(self, batch_size):
        if not self.buffer:
            return []
        buffer_rewards = torch.tensor([r for _, r in self.buffer], device=self.device)
        probs = torch.softmax(buffer_rewards, dim=0)
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)
        return [self.buffer[i][0] for i in indices.tolist()]

    def get_high_reward_samples(self, top_k=10):
        if not self.buffer:
            return []
        sorted_buffer = sorted(self.buffer, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_buffer[:min(top_k, len(sorted_buffer))]]


# ============================================================================
# Sampling "n" likely-SMILES from the GeneratorModel
class GenSampler():
    def __init__(self, model, tokenizer, batch_size, max_len, reward_fn=None, buffer_capacity=10000, replay_prob=0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.reward_fn = reward_fn
        self.device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_buffer = RewardBuffer(capacity=buffer_capacity, device=self.device)
        self.replay_prob = replay_prob  # 新增参数

    # Sampling a batch of samples by the trained generator
    def sample(self, data=None, use_replay=False):
        """
        采样一批分子
        data: 初始序列
        use_replay: 是否使用经验回放中的样本进行条件生成
        """
        self.model.eval()
        
        # 如果使用经验回放，从buffer中采样作为条件生成的起点
        if use_replay and random.random() < self.replay_prob:  # 使用参数控制概率
            replay_samples = self.reward_buffer.sample(self.batch_size)
            if replay_samples:
                # 这里简化处理，实际中可能需要将SMILES转回tensor并截断
                return replay_samples
        
        finished = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        sample_tensor = torch.zeros((self.max_len, self.batch_size), dtype=torch.long, device=self.device)
        sample_tensor[0] = self.tokenizer.char_to_int[self.tokenizer.start]
        
        with torch.no_grad():
            if data is None:
                init = 1
            else:
                sample_tensor[:len(data)] = data.to(self.device)
                init = len(data)

            for i in range(init, self.max_len):
                tensor = sample_tensor[:i]
                logits = self.model.forward(tensor)[-1]
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                sampled_char = torch.multinomial(probabilities, 1)
                sampled_char = sampled_char.squeeze(-1)
                sampled_char[finished] = self.tokenizer.char_to_int[self.tokenizer.end]
                finished |= (sampled_char == self.tokenizer.char_to_int[self.tokenizer.end])
                sample_tensor[i] = sampled_char
                if finished.all():
                    break

        smiles = ["".join(self.tokenizer.decode(sample_tensor[:, i].cpu().numpy())).strip("^$ ") for i in range(self.batch_size)]

        if self.reward_fn is not None:
            # 如果 reward_fn 支持 batch，建议 reward_fn('druglikeness', smiles) 返回一维 list
            rewards = self.reward_fn('druglikeness', smiles)
            # 如果 rewards 是 numpy array 或 torch tensor，转为 list
            if isinstance(rewards, np.ndarray) or torch.is_tensor(rewards):
                rewards = rewards.tolist()
            # 保证长度一致
            assert len(smiles) == len(rewards), f"smiles和rewards长度不一致: {len(smiles)} vs {len(rewards)}"
            self.reward_buffer.add(smiles, rewards)

        return smiles

    # 使用MC Dropout进行不确定性估计的采样
    def sample_with_uncertainty(self, n_samples=5, data=None):
        """
        使用MC Dropout估计生成不确定性
        n_samples: 为每个位置生成的样本数
        data: 初始序列
        """
        self.model.train()  # 启用dropout
        all_samples = []
        all_rewards = []

        for _ in range(n_samples):
            batch_samples = self.sample(data)
            all_samples.extend(batch_samples)
            if self.reward_fn is not None:
                # 推荐 reward_fn 支持 batch 输入
                rewards = self.reward_fn('druglikeness', batch_samples)
                if isinstance(rewards, (np.ndarray, torch.Tensor)):
                    rewards = np.squeeze(rewards).tolist()
                batch_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
                all_rewards.append(batch_rewards)
        
        self.model.eval()
        
        if self.reward_fn is not None:
            all_rewards = torch.cat(all_rewards)  # [n_samples * batch_size]
            # 分组统计
            smiles_tensor = np.array(all_samples)
            unique_smiles, inverse_indices = np.unique(smiles_tensor, return_inverse=True)
            stats = []
            for idx, smiles in enumerate(unique_smiles):
                group_mask = (inverse_indices == idx)
                group_rewards = all_rewards[group_mask]
                mean_reward = group_rewards.mean().item()
                std_reward = group_rewards.std().item()
                count = group_rewards.numel()
                ucb_score = mean_reward + 1.0 * std_reward / (count ** 0.5)
                stats.append((smiles, ucb_score))
            # 按 UCB 排序
            stats.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in stats[:self.batch_size]]
        else:
            return all_samples[:self.batch_size]

    # Sampling "n" samples by the trained generator
    def sample_multi(self, n, filename=None, use_mc_dropout=False, use_replay=False):
        samples = []
        for _ in tqdm(range(int(n / self.batch_size))):
            if use_mc_dropout:
                # tqdm.write("Sampled with MC Dropout")
                batch_sample = self.sample_with_uncertainty()
            else:
                # 使用外部传入的use_replay参数，而不是仅基于buffer是否为空
                use_buffer_replay = use_replay and (len(self.reward_buffer.buffer) > 0)
                batch_sample = self.sample(use_replay=use_buffer_replay)
            samples.extend(batch_sample)
        
        # 高奖励样本添加也应该受到use_replay控制
        if self.reward_fn is not None and use_replay:
            high_reward_samples = self.reward_buffer.get_high_reward_samples(top_k=min(n//10, 100))
            samples = high_reward_samples + samples[:n-len(high_reward_samples)]
        
        # 写入文件
        if filename:
            with open(filename, 'w') as fout:
                for s in samples[:n]:
                    fout.write('{}\n'.format(s))
        return samples[:n]