import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from text.symbols import symbols
from transformers import WavLMModel, AutoFeatureExtractor
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
import numpy as np
import random

from torch.autograd import Function

import editdistance


def calc_lev(seq1, seq2):
    return editdistance.eval(seq1, seq2)


class AAMSoftmax(nn.Module):
    def __init__(self, in_dim, n_classes, s=30.0, m=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m
    
    def logits(self, feats): 
        x = F.normalize(feats, dim=1)
        W = F.normalize(self.W, dim=1)
        return self.s * (x @ W.t())  # (B,S)

    def forward(self, feats, labels):
        x = F.normalize(feats, dim=1)      # (B, D)
        W = F.normalize(self.W, dim=1)     # (S, D)
        cos = F.linear(x, W)               # (B, S)
        sin = torch.sqrt(torch.clamp(1.0 - cos**2, 1e-7, 1.0))
        phi = cos * math.cos(self.m) - sin * math.sin(self.m)  # cos(theta+m)

        one_hot = F.one_hot(labels, num_classes=W.size(0)).float()
        logits = (one_hot * phi + (1.0 - one_hot) * cos) * self.s
        loss = F.cross_entropy(logits, labels)
        return loss, logits

def slerp(u: torch.Tensor, v: torch.Tensor, lam: torch.Tensor, eps: float = 1e-6):
    dot = (u * v).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(dot)
    sin_t = torch.sin(theta).clamp_min(eps)
    k0 = torch.sin((1 - lam) * theta) / sin_t
    k1 = torch.sin(lam * theta) / sin_t
    out = k0 * u + k1 * v
    return F.normalize(out, dim=1)



def _frame_layer_norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.transpose(1, 2), (x.size(1),)).transpose(1, 2)

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        num = (x * mask).sum()
        den = mask.sum().clamp_min(1.0)
        return num / den
    else:
        num = (x * mask.squeeze(1)).sum()
        den = mask.squeeze(1).sum().clamp_min(1.0)
        return num / den

@torch.no_grad()
def _expand_lambda(lam: torch.Tensor, T: int) -> torch.Tensor:
    return lam.expand(-1, 1, T)



class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, gin_channels_det=256, s_max=0.4):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None, d_g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()



class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        out, gate = out.chunk(2, dim=1)
        return out * torch.sigmoid(gate)  # Gated Linear Unit (GLU)


class ContentEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,       
                 content_channels: int,  
                 hidden_channels: int,   
                 kernel_size: int,       
                 dilation_rate: int,     
                 n_layers: int):         
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.resnet = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=0
        )
        self.proj_out = nn.Conv1d(hidden_channels, content_channels, 1)

    def forward(self, feats: torch.Tensor, lengths: torch.LongTensor):

        mask = torch.unsqueeze(commons.sequence_mask(lengths, feats.size(2)), 1).to(feats.dtype)

        h = self.proj_in(feats) * mask

        h = self.resnet(h, mask, g=None)

        content = self.proj_out(h) * mask
        return content, mask



class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, attn_heads=4, num_gated_layers=4, speaker_dim=256, num_speakers=1000):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, hidden_dim)  # Frame-wise FC
        
        self.gated_blocks = nn.ModuleList([
            GatedConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(num_gated_layers)
        ])
        
        self.attn_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global pooling
        
        self.projection = nn.Linear(hidden_dim, speaker_dim)  # Final speaker vector projection
        

    def forward(self, feat, lengths, return_embedding=False, train=True):
        # Input feats: (B, H(768), T)

        mask_t = commons.sequence_mask(lengths, feat.size(2)).unsqueeze(-1).to(feat.dtype)  # (B, T, 1)
        feat = feat.transpose(1, 2) # (B, T, H) for Linear

        x = self.fc(feat) * mask_t # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T) for Conv1d
        mask = mask_t.transpose(1, 2)

        # Gated Convs + residual
        for conv in self.gated_blocks:
            residual = x
            x = conv(x) * mask
            x = x + residual  # residual connection

        x = x.transpose(1, 2)  # (B, T, H) for attention
        key_padding = ~commons.sequence_mask(lengths, x.size(1)).to(torch.bool)
        # Multi-head attention
        x, _ = self.attn_layer(x, x, x, key_padding_mask=key_padding)  # self-attention

        # Global average pooling
        x = x.transpose(1, 2)  # (B, H, T)

        T = x.size(2)
        x = self.pooling(x * mask).squeeze(-1)  # (B, H)

        s = self.projection(x)  # (B, speaker_dim)

        emb = s.unsqueeze(-1).expand(-1, -1, T)

        if return_embedding:
            return emb 
        else:
            return s, emb





class P2VC(nn.Module):
    def __init__(self,
        segment_size,
        inter_channels,
        hidden_channels,
        n_heads,
        n_layers,
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes,
        kernel_size=3,
        n_speakers=107,
        gin_channels=107,
        **kwargs):
    
        super().__init__()
        self.inter_channels = inter_channels # 0
        self.hidden_channels = hidden_channels # 0
        self.n_heads = n_heads # 0
        self.n_layers = n_layers # 0
        self.kernel_size = kernel_size # 0 
        self.resblock = resblock # 0
        self.resblock_kernel_sizes = resblock_kernel_sizes # 0
        self.resblock_dilation_sizes = resblock_dilation_sizes # 0
        self.upsample_rates = upsample_rates # 0
        self.upsample_initial_channel = upsample_initial_channel # 0
        self.upsample_kernel_sizes = upsample_kernel_sizes # 0
        self.segment_size = segment_size
        self.n_speakers = n_speakers # 0 
        self.gin_channels = gin_channels


        # inter channel == 256
        # hidden channel == 256
        self.dec = Generator(inter_channels*2, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm=WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self.wavlm.eval()


        self.content_enc = ContentEncoder(
            self.wavlm.config.hidden_size,
            inter_channels, 
            hidden_channels,
            kernel_size,
            dilation_rate=2,
            n_layers=6
        )
        self.spk_enc = SpeakerEncoder(
            self.wavlm.config.hidden_size,
            hidden_dim=hidden_channels,
            attn_heads=n_heads,
            num_gated_layers=n_layers,
            speaker_dim=inter_channels,
            num_speakers=n_speakers
        )

        # CTC head
        self.ctc_fc = nn.Linear(inter_channels, len(symbols)+2)
        self.ctc_loss_fn = nn.CTCLoss(blank=len(symbols)+1, zero_infinity=True) 
        self.aam = AAMSoftmax(in_dim=inter_channels, n_classes=n_speakers, s=30.0, m=0.2)

    def forward(self, y, y_lengths, perturb_z, text_targets, target_lengths, sid=None, mix_idx=None): # x=text | y=wav
        # ------------- content disentanglement training ----------------- #
        with torch.no_grad():
            z = self.wavlm(y).last_hidden_state # z (B, T, H(768))
            perturb_z = self.wavlm(perturb_z).last_hidden_state
        feats=z.transpose(1,2)              # feats (B, H(768), T)
        feats_perturb = perturb_z.transpose(1,2)
        # ---------------------------------------------------------------- #

        feats_lengths = []
        for i in range(y.shape[0]):
            l = y_lengths[i].item()
            T_i = self.wavlm._get_feat_extract_output_lengths(l)  # 공식 제공
            feats_lengths.append(T_i)
        feats_lengths = torch.tensor(feats_lengths, device=y.device)

        # ---------------- content encoder -------------------------------#
        mixout = self.content_mixup_block(
            feats_clean=feats, feats_pert=feats_perturb, feats_lengths=feats_lengths,
            text_targets=text_targets, target_lengths=target_lengths
        )
        
        logp = mixout['logp_c']
        content_feature = mixout["c_for_dec"]
        
        # --------------------------------------------------------------- #

        # --------------------- disentanglement speaker training -------------------- #
        # --------------------- time perturbation feats -------------------- #
        feats_shuffled = self.random_shuffle_minlen_segments(feats, feats_lengths)
        spk = self.speaker_mixup_block(
            feats_clean=feats,
            feats_shuf=feats_shuffled,
            feats_lengths=feats_lengths,
            sid=sid,
            w_mix=0.2,        
            m=0.2, s=30.0,
            mix_mode="slerp"
        )
        # ------------------------------------------------------------------ #
        # --------------------------------------------------------------------------- #
   

        # CTC Loss for mixup content embedding
        phoneme_pred = logp.argmax(dim=-1) 

        phoneme_loss, phoneme_acc = None, None
        if text_targets is not None and target_lengths is not None:
            phoneme_loss = mixout["L_ctc_mix"]

            blank_id = len(symbols) + 1
            phoneme_acc, phoneme_cer = self.ctc_sequence_accuracy(
                phoneme_pred, text_targets, target_lengths, blank_id
            )

        # spk loss
        spk_loss = spk["L_spk"]

        with torch.no_grad():
            eval_logits = self.aam.logits(spk["s_clean"])  # (B,S)
            spk_pred = eval_logits.argmax(dim=-1)
            spk_acc  = (spk_pred == sid).float().mean().item()


        content_z, ids_slice = commons.rand_slice_segments(content_feature, feats_lengths, self.segment_size)
        spk_z = commons.slice_segments(spk["spk_emb"], ids_slice, self.segment_size)

        z = torch.cat([content_z, spk_z], dim=1)
        
        
        o = self.dec(z)
        # --------------------- MultiGPU -------------------- #
        phoneme_acc = torch.tensor(phoneme_acc, device=o.device)
        phoneme_cer = torch.tensor(phoneme_cer, device=o.device)
        spk_acc = torch.tensor(spk_acc, device=o.device)
        # --------------------------------------------- #

        return o, ids_slice, phoneme_loss, spk_loss, phoneme_acc, phoneme_cer, spk_acc

    @torch.no_grad()
    def infer(self, y, y_lengths,
            gt_phoneme=None, gt_phoneme_lengths=None,
            sid=None, save_path=None
        ): # x=text | y=wav


        wavlm_out = self.wavlm(y).last_hidden_state      # (B, T_feat, H)
        feats = wavlm_out.transpose(1, 2)                # (B, H, T_feat)
        feat_lengths = []
        for i in range(y.shape[0]):
            l = y_lengths[i].item()
            T_i = self.wavlm._get_feat_extract_output_lengths(l)  # 공식 제공
            feat_lengths.append(T_i)
        feat_lengths = torch.tensor(feat_lengths, device=y.device)

        content, _ = self.content_enc(feats, feat_lengths)
        
        logits_ctc = self.ctc_fc(content.transpose(1, 2))      # (B, T_feat, vocab_size)
        logp = logits_ctc.log_softmax(-1)
        phoneme_pred = logp.argmax(dim=-1)                     # (B, T_feat)
        
        phoneme_loss = None
        phoneme_acc = None
        phoneme_cer=None
        if gt_phoneme is not None and gt_phoneme_lengths is not None:
            ctc_logp = logp.transpose(0, 1)
            phoneme_loss = self.ctc_loss_fn(ctc_logp, gt_phoneme, feat_lengths, gt_phoneme_lengths)

            blank_id = len(symbols) + 1
            phoneme_acc, phoneme_cer = self.ctc_sequence_accuracy(
                phoneme_pred, gt_phoneme, gt_phoneme_lengths, blank_id, save_path
            )
            

        spk_vec, spk_emb = self.spk_enc(feats, feat_lengths, train=False)
        spk_logits = self.aam.logits(spk_vec)
        spk_pred = spk_logits.argmax(dim=-1)
        spk_loss = 0.0
        spk_acc = None
        if sid is not None:
            spk_acc = (spk_pred == sid).float().mean().item() 




        z = torch.cat([content, spk_emb], dim=1)
        recon_wav = self.dec(z)     


        if gt_phoneme is not None:
            # --------------------- MultiGPU -------------------- #
            phoneme_loss = torch.tensor(phoneme_loss, device=recon_wav.device)
            spk_loss = torch.tensor(spk_loss, device=recon_wav.device)
            phoneme_acc = torch.tensor(phoneme_acc, device=recon_wav.device)
            phoneme_cer = torch.tensor(phoneme_cer, device=recon_wav.device)
            spk_acc = torch.tensor(spk_acc, device=recon_wav.device)
            # --------------------------------------------- #

            return {
                'recon_wav': recon_wav,
                'phoneme_loss': phoneme_loss,
                'phoneme_acc': phoneme_acc,
                'phoneme_cer': phoneme_cer,
                'phoneme_pred': phoneme_pred,
                'spk_loss': spk_loss,
                'spk_acc': spk_acc,
                'spk_pred': spk_pred
            }
        else:
             return {
                'recon_wav': recon_wav,
            }
        
    
    @torch.no_grad()
    def vc(self, src, src_lengths,
            tgt, tgt_lengths,
        ): # x=text | y=wav

        src_wavlm_out = self.wavlm(src).last_hidden_state      # (B, T_feat, H)
        src_feats = src_wavlm_out.transpose(1, 2)                # (B, H, T_feat)
        src_feat_lengths = []
        for i in range(src.shape[0]):
            l = src_lengths[i].item()
            T_i = self.wavlm._get_feat_extract_output_lengths(l)  # 공식 제공
            src_feat_lengths.append(T_i)
        src_feat_lengths = torch.tensor(src_feat_lengths, device=src.device)


        tgt_wavlm_out = self.wavlm(tgt).last_hidden_state      # (B, T_feat, H)
        tgt_feats = tgt_wavlm_out.transpose(1, 2)                # (B, H, T_feat)
        tgt_feat_lengths = []
        for i in range(tgt.shape[0]):
            l = tgt_lengths[i].item()
            T_i = self.wavlm._get_feat_extract_output_lengths(l)  # 공식 제공
            tgt_feat_lengths.append(T_i)
        tgt_feat_lengths = torch.tensor(tgt_feat_lengths, device=tgt.device)

        tgt_feats, tgt_feat_lengths = self.match_length_to_src(src_feats, tgt_feats, tgt_feat_lengths)

        content, _ = self.content_enc(src_feats, src_feat_lengths)

        _, spk_emb = self.spk_enc(tgt_feats, tgt_feat_lengths, train=False)
        

        z = torch.cat([content, spk_emb], dim=1)
        recon_wav = self.dec(z)      # (B, 1, T_wav) 등


        return {
            'recon_wav': recon_wav
        }

    def match_length_to_src(self, src, tgt, tgt_feat_lengths):

        B, C, T_src = src.shape
        _, _, T_tgt = tgt.shape

        if isinstance(tgt_feat_lengths, torch.Tensor):
            tgt_feat_lengths = tgt_feat_lengths.clone()
        else:
            tgt_feat_lengths = torch.tensor(tgt_feat_lengths).clone()

        if T_tgt < T_src:
            repeat_times = T_src // T_tgt      
            remain = T_src % T_tgt             

            tgt_repeated = tgt.repeat(1, 1, repeat_times)  # (B, C, T_tgt * repeat_times)
            if remain > 0:
                tgt_repeated = torch.cat([tgt_repeated, tgt[:, :, :remain]], dim=2)
            tgt = tgt_repeated
            tgt_feat_lengths[:] = T_src
        elif T_tgt > T_src:
            tgt = tgt[:, :, :T_src]
            tgt_feat_lengths = torch.clamp(tgt_feat_lengths, max=T_src)


        return tgt, tgt_feat_lengths


    def ctc_decode(self, pred, blank_id):
        out = []
        prev = None
        for t in pred:
            t = t.item()
            if t != blank_id and (prev is None or t != prev):
                out.append(t)
            prev = t
        return out

    def indices_to_symbols(self, indices):
        return [symbols[idx] if idx < len(symbols) else '' for idx in indices]

    def ctc_sequence_accuracy(self, phoneme_pred, text_targets, target_lengths, blank_id,
                            save_path=None):

        batch_acc = []
        batch_cer = []
        pred_seqs = []
        target_seqs = []


        B = phoneme_pred.shape[0]
        for b in range(B):
            pred_seq = self.ctc_decode(phoneme_pred[b], blank_id)
            target_seq = text_targets[b, :target_lengths[b]].cpu().tolist()

            dist = calc_lev(pred_seq, target_seq)

            acc = max(0, (len(target_seq) - dist) / len(target_seq)) if len(target_seq) > 0 else 0
            cer = dist / len(target_seq) if len(target_seq) > 0 else 0.0
            
            batch_acc.append(acc)
            batch_cer.append(cer)
            pred_seqs.append(pred_seq)
            target_seqs.append(target_seq)


        ctc_acc = sum(batch_acc) / len(batch_acc)
        ctc_cer = sum(batch_cer) / len(batch_cer)

        if save_path is not None:
            with open(f"{save_path}/eval_asr_result.txt", 'w', encoding='utf-8') as f:
                for i, (pred, tgt, acc, cer) in enumerate(zip(pred_seqs, target_seqs, batch_acc, batch_cer)):
                    f.write(f"Sample {i}\tpred : {pred}\tlabel: {tgt}\tacc: {acc}\cer: {cer}\n")

        return ctc_acc, ctc_cer


    def random_shuffle_minlen_segments(self, feats, lengths):
        B, H, T = feats.shape
        min_len = lengths.min().item() 
        if min_len < 30:
            return feats
        seg = min_len // 3

        seg_idxs = [
            (0, seg),
            (seg, 2*seg),
            (2*seg, min_len)
        ]

        idx = torch.randperm(3)
        segments = [feats[:, :, start:end] for (start, end) in seg_idxs]


        feats_shuffled = feats.clone()

        out_start = 0
        for order in idx:
            seg = segments[order]
            l = seg.shape[-1]
            feats_shuffled[:, :, out_start:out_start+l] = seg
            out_start += l

        return feats_shuffled

    def speaker_mixup_block(
        self,
        feats_clean: torch.Tensor,     # (B, H, T)
        feats_shuf: torch.Tensor,      # (B, H, T)  (same speaker, time-shuffled content)
        feats_lengths: torch.Tensor,   # (B,)
        sid: torch.Tensor,             # (B,) speaker id
        w_mix: float = 0.2,            
        m: float = 0.2, s: float = 30.0,
        mix_mode: str = "slerp"        
    ):

        B = feats_clean.size(0)
        device = feats_clean.device


        s_clean, spk_emb_clean = self.spk_enc(feats_clean, feats_lengths, train=True)  # (B,D), (B,D,T)
        s_shuf,  _             = self.spk_enc(feats_shuf,  feats_lengths, train=True)  # (B,D)

        eps = 1e-3
        lam = torch.rand(B, 1, device=device).clamp_(eps, 1 - eps)  # (B,1)


        u, v = F.normalize(s_clean, dim=1), F.normalize(s_shuf, dim=1)
        if mix_mode == "slerp":
            s_mix = slerp(u, v, lam)                       # (B, D), unit-norm
        else:
            s_mix = F.normalize(lam * u + (1 - lam) * v, dim=1)

        if not hasattr(self, "aam"):
            self.aam = AAMSoftmax(in_dim=s_clean.size(1), n_classes=self.n_speakers, s=s, m=m).to(device)

        L_aam_clean, spk_logits = self.aam(s_clean, sid)
        L_aam_shuf,  _ = self.aam(s_shuf,  sid)
        L_aam_mix,   _ = self.aam(s_mix,   sid)           # intra-speaker mix (same label)

        L_spk_main = L_aam_clean + L_aam_shuf + w_mix * L_aam_mix

        L_spk = L_spk_main


        return {
            "spk_emb": spk_emb_clean,
            "spk_logits" : spk_logits,
            "s_clean" : s_clean,
            "L_spk": L_spk,
            "L_spk_main": L_spk_main,
            "L_aam_clean": L_aam_clean,
            "L_aam_shuf": L_aam_shuf,
            "L_aam_mix": L_aam_mix,
        }

    def _pack_targets(self, text_targets, target_lengths, repeat_k=1):
        if text_targets.dim() == 2:
            B, Lmax = text_targets.size()

            flat = []
            for b in range(B):
                l = int(target_lengths[b].item())
                flat.append(text_targets[b, :l])
            targets_1d = torch.cat(flat, dim=0)
        else:
            targets_1d = text_targets

        target_lengths_rep = target_lengths.repeat(repeat_k)
        targets_1d_rep = targets_1d.repeat(repeat_k)  
        return targets_1d_rep, target_lengths_rep


    def content_mixup_block(
        self,
        feats_clean, feats_pert, feats_lengths,
        text_targets, target_lengths
    ):

        B, H, T = feats_clean.shape
        device  = feats_clean.device

        c, _  = self.content_enc(feats_clean,  feats_lengths)   # (B,C,T)
        cp,_  = self.content_enc(feats_pert,   feats_lengths)   # (B,C,T)

        lam = torch.rand_like(c[:, :1, :1]).clamp_(1e-3, 1-1e-3).to(device)

        c_mix = lam * c + (1 - lam) * cp

        c_all = torch.cat([c, cp, c_mix], dim=0)               # (3B,C,T)
        logits_all = self.ctc_fc(c_all.transpose(1, 2))        # (3B,T,V)
        logp_all   = logits_all.log_softmax(dim=-1)            # (3B,T,V)

        logp_c, logp_p, logp_m = torch.chunk(logp_all, 3, dim=0)  # (B,T,V)

        feats_lengths_3x = feats_lengths.repeat(3)
        targets_1d_3x, target_lengths_3x = self._pack_targets(
            text_targets, target_lengths, repeat_k=3
        )

        logp_all_TBV = logp_all.transpose(0, 1)               # (T,3B,V)

        L_ctc_all = self.ctc_loss_fn(logp_all_TBV, targets_1d_3x, feats_lengths_3x, target_lengths_3x)

        c_for_dec = c     # or just: c

        return {
            "c_clean": c,
            "c_pert":  cp,
            "c_mix":   c_mix,
            "logp_c":   logp_c,
            "c_for_dec": c_for_dec,
            "L_ctc_mix": L_ctc_all,
        }
