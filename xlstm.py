import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from glob import glob
from math import floor
import matplotlib.pyplot as plt

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class XLSTM_Multi(nn.Module):
    def __init__(self, in_chan=1, hidden1=64, hidden2=96, future_steps=5):
        super().__init__()
        self.future_steps = future_steps

        self.enc1 = ConvLSTMCell(in_chan, hidden1)
        self.enc2 = ConvLSTMCell(hidden1, hidden2)

        self.dec1 = ConvLSTMCell(in_chan, hidden1)
        self.dec2 = ConvLSTMCell(hidden1, hidden2)

        self.conv_out = nn.Conv2d(hidden2, in_chan, kernel_size=1)

    def forward(self, inp_seq, future=None):
        if future is None:
            future = self.future_steps

        B, T, C, H, W = inp_seq.shape
        device = inp_seq.device

        h1 = torch.zeros(B, 64, H, W, device=device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, 96, H, W, device=device)
        c2 = torch.zeros_like(h2)

        
        for t in range(T):
            x = inp_seq[:, t]
            h1, c1 = self.enc1(x, h1, c1)
            h2, c2 = self.enc2(h1, h2, c2)

        
        dec_in = inp_seq[:, -1]

        
        dh1 = torch.zeros(B, 64, H, W, device=device)
        dc1 = torch.zeros_like(dh1)
        dh2 = torch.zeros(B, 96, H, W, device=device)
        dc2 = torch.zeros_like(dh2)

        preds = []

        
        for step in range(future):
            dh1, dc1 = self.dec1(dec_in, dh1, dc1)
            dh2, dc2 = self.dec2(dh1, dh2, dc2)

            out = torch.sigmoid(self.conv_out(dh2))
            preds.append(out)

            dec_in = out

        return torch.stack(preds, dim=1)

class SeqDataset(Dataset):
    def __init__(self, folder, past=10, future=5, size=128):
        self.past = past
        self.future = future
        self.size = size

        files = sorted(glob(os.path.join(folder, "*.png")))
        self.frames = [cv2.imread(f, 0) for f in files]
        self.frames = [cv2.resize(f, (size, size)) for f in self.frames]

        self.frames = np.array(self.frames)  # (N,H,W)
        self.frames = self.frames / 255.0

    def __len__(self):
        return len(self.frames) - (self.past + self.future)

    def __getitem__(self, idx):
        seq_in = self.frames[idx : idx + self.past]
        seq_out = self.frames[idx + self.past : idx + self.past + self.future]

        seq_in = torch.tensor(seq_in).unsqueeze(1).float()    # (past, 1, H, W)
        seq_out = torch.tensor(seq_out).unsqueeze(1).float()  # (future, 1, H, W)

        return seq_in, seq_out

def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    return (pred_dx - tgt_dx).abs().mean() + (pred_dy - tgt_dy).abs().mean()


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0

    for seq_in, seq_out in loader:
        seq_in = seq_in.to(device)
        seq_out = seq_out.to(device)

        opt.zero_grad()
        preds = model(seq_in, future=seq_out.shape[1])  # (B,future,1,H,W)

        mse = ((preds - seq_out)**2).mean()
        gdl = gradient_loss(preds, seq_out)
        loss = mse + 0.5 * gdl

        loss.backward()
        opt.step()

        total += loss.item()

    return total / len(loa
def main():
    DATA_FOLDER = "/home/kalyani/final/dataset"   # CHANGE THIS
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH = 4
    EPOCHS = 5
    PAST = 10
    FUTURE = 5
    SIZE = 128

    print("Device:", DEVICE)

    dataset = SeqDataset(DATA_FOLDER, past=PAST, future=FUTURE, size=SIZE)
    print("Dataset size:", len(dataset), "sequences")

    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    model = XLSTM_Multi(future_steps=FUTURE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------- TRAIN --------
    print("Training started...")
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, opt, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss = {loss:.6f}")

    torch.save(model.state_dict(), "xlstm_final.pth")
    print("Model saved as xlstm_final.pth")

    # -------- INFERENCE (predict last sequence) --------
    model.eval()
    seq_in, seq_out = dataset[-1]
    seq_in = seq_in.unsqueeze(0).to(DEVICE)
    preds = model(seq_in).detach().cpu()

    os.makedirs("pred_output", exist_ok=True)
    for i in range(FUTURE):
        img = preds[0, i, 0].numpy() * 255
        cv2.imwrite(f"pred_output/pred_{i}.png", img)

    print("Predictions saved in pred_output/")


if __name__ == "__main__":
    main()

