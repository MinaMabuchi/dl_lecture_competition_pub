import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
import torch.optim as optim
from torchmetrics import Accuracy
from tqdm import tqdm
import os
import numpy as np



#CLIP
class TransformerBrainToCLIP(nn.Module):
    def __init__(self, input_dim, seq_len, num_layers, nhead, dim_feedforward, output_dim, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, dim_feedforward)

        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout, max_len=seq_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.output_proj = nn.Linear(dim_feedforward, output_dim)

    def forward(self, src):
        # src shape: (batch_size, input_dim, seq_len)

        src = src.permute(0, 2, 1)  # (batch_size, seq_len, input_dim)
        src = self.input_proj(src)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, dim_feedforward)

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        # Global average pooling
        output = output.mean(dim=0)

        return self.output_proj(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def save_checkpoint_clip(epoch, model, optimizer, train_loss, val_loss, batch_idx, checkpoint_dir, filename="checkpoint.pt"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint_clip(model, optimizer, checkpoint_dir, filename="checkpoint.pt"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        print(f"Checkpoint loaded: {checkpoint_path}")
        return epoch, batch_idx, train_loss, val_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0, 0.0, float('inf')

def train_epoch(model, clip_model, train_loader, optimizer, criterion, device, epoch, start_batch=0):
    model.train()
    train_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, initial=start_batch, desc=f"Epoch {epoch+1} - Training")):
        meg_data, labels, subject_idxs, images = [b.to(device) for b in batch]

        #print(f"MEG data shape: {meg_data.shape}")
        #print(f"Images shape: {images.shape}")

        with torch.no_grad():
            true_clip_features = clip_model.encode_image(images).float()

        #print(f"True CLIP features shape: {true_clip_features.shape}")

        predicted_clip_features = model(meg_data)
        #print(f"Predicted CLIP features shape: {predicted_clip_features.shape}")

        predicted_clip_features = model(meg_data)
        loss = criterion(predicted_clip_features, true_clip_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 300 == 0:
            save_checkpoint(epoch, model, optimizer, train_loss / (batch_idx + 1), None, batch_idx + 1,
                    checkpoint_dir=args.checkpoint_dir",
                    filename=f"brain_to_clip_checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt")

    return train_loss / len(train_loader)

def validate(model, clip_model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            meg_data, labels, subject_idxs, images = [b.to(device) for b in batch]

            true_clip_features = clip_model.encode_image(images).float()
            predicted_clip_features = model(meg_data)
            loss = criterion(predicted_clip_features, true_clip_features)

            val_loss += loss.item()

    return val_loss / len(val_loader)



#
class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        clip_dim: int,
        num_subjects: int,
        hid_dim: int = 128,
        embedding_dim: int = 2
    ) -> None:
        super().__init__()

        self.subject_emb = nn.Embedding(num_subjects, embedding_dim)
        self.embedding_fc = nn.Linear(embedding_dim, in_channels)

        self.meg_blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim * 2),
        )

        self.clip_fc = nn.Linear(clip_dim, hid_dim * 2)

        self.combined_fc = nn.Linear(hid_dim * 4, hid_dim * 2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 2, num_classes),
        )

    def forward(self, meg_data: torch.Tensor, clip_features: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        # 被験者情報の処理
        subject_features = self.subject_emb(subject_idxs)
        subject_features = self.embedding_fc(subject_features)
        subject_features = subject_features.unsqueeze(2)
        subject_features = subject_features.repeat(1, 1, meg_data.size(2))

        # MEGデータの処理
        meg_features = self.meg_blocks(meg_data + subject_features)
        meg_features = meg_features.mean(dim=2)  # Global Average Pooling

        # CLIP特徴量の処理
        clip_features = self.clip_fc(clip_features)

        # 特徴量の結合
        combined_features = torch.cat([meg_features, clip_features], dim=1)
        combined_features = self.combined_fc(combined_features)

        # 分類
        return self.head(combined_features.unsqueeze(2))

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same", dilation=2)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same", dilation=3)

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = nn.functional.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = nn.functional.gelu(self.batchnorm1(X))

        X = self.conv2(X) + X
        X = nn.functional.gelu(self.batchnorm2(X))

        return self.dropout(X)


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, val_acc, batch_idx, checkpoint_dir, filename="checkpoint.pt"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_dir, filename="checkpoint.pt"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        val_acc = checkpoint['val_acc']
        print(f"Checkpoint loaded: {checkpoint_path}")
        return epoch, batch_idx, train_loss, val_loss, val_acc
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0, 0.0, float('inf'), 0.0
