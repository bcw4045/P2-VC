import torch 
from torch.nn import functional as F
import torch.nn as nn

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)  
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses     


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class STFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss
    참조:
      - Parallel WaveGAN (ICASSP 2020): multi-resolution spectral loss
      - HiFi-GAN (NeurIPS 2020): spectral convergence + log magnitude loss
    """
    def __init__(self, 
                 fft_sizes=[400,800,1200], 
                 hop_sizes=[100,200,300], 
                 win_lengths=[400,800,1200]):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes   = fft_sizes
        self.hop_sizes   = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x, y):
        """
        Args:
          x: generated waveform, (B, 1, T)
          y: ground-truth waveform, (B, 1, T)
        Returns:
          sc_loss: spectral convergence loss
          mag_loss: log-magnitude loss
        """
        sc_loss   = 0.0
        mag_loss  = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # STFT
            X = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win,
                           window=torch.hann_window(win).to(x.device),
                           return_complex=True)  # (B, F, T')
            Y = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win,
                           window=torch.hann_window(win).to(y.device),
                           return_complex=True)

            # magnitude
            mag_x = torch.abs(X)
            mag_y = torch.abs(Y)

            # 1) Spectral Convergence Loss: || |Y| - |X| ||_F / || |Y| ||_F
            sc_loss += torch.norm(mag_y - mag_x, p='fro') / (torch.norm(mag_y, p='fro') + 1e-8)

            # 2) Log STFT Magnitude Loss: L1 on log mags
            mag_loss += F.l1_loss(torch.log(mag_y + 1e-7), torch.log(mag_x + 1e-7))

        # 평균내기
        sc_loss  = sc_loss  / len(self.fft_sizes)
        mag_loss = mag_loss / len(self.fft_sizes)
        return sc_loss, mag_loss