import warnings

warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import numpy as np
from tqdm import tqdm
import copy
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import params
from models import (
    P2VC,
    MultiPeriodDiscriminator
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    STFTLoss
)

import matplotlib.pyplot as plt
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
import time

import commons
from meldataset import mel_spectrogram

import torch.nn as nn
import os
import torch.nn.functional as F
import json
import warnings

warnings.filterwarnings("ignore")

def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.test_filelist_path
add_blank = params.add_blank
cmu_dict = params.cmu_dict

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
random_seed = params.seed

inter_channels = params.inter_channels
hidden_channels = params.hidden_channels
n_layers = params.n_layers
n_heads = params.n_heads
resblock = params.resblock
resblock_kernel_sizes = params.resblock_kernel_sizes
resblock_dilation_sizes = params.resblock_dilation_sizes
upsample_rates = params.upsample_rates
upsample_initial_channel = params.upsample_initial_channel
upsample_kernel_sizes = params.upsample_kernel_sizes


sample_rate = params.sample_rate
segment_size = params.segment_size
note = params.note

wandb_project = params.wandb_project
wandb_name = params.wandb_name
wandb_resume = params.wandb_resume

resume = params.resume
ckpt = params.ckpt
note = params.note

params_dict = {
    "train_filelist_path": params.train_filelist_path,
    "valid_filelist_path": params.valid_filelist_path,
    "add_blank": params.add_blank,
    "log_dir": params.log_dir,
    "n_epochs": params.n_epochs,
    "batch_size": params.batch_size,
    "learning_rate": params.learning_rate,
    "random_seed": params.seed,
    "inter_channels": params.inter_channels,
    "hidden_channels": params.hidden_channels,
    "n_layers": params.n_layers,
    "n_heads": params.n_heads,
    "sample_rate": params.sample_rate,
    "segment_size": segment_size,
    "resblock": params.resblock,
    "resblock_kernel_sizes": params.resblock_kernel_sizes,
    "resblock_dilation_sizes": params.resblock_dilation_sizes,
    "upsample_rates": params.upsample_rates,
    "upsample_initial_channel": params.upsample_initial_channel,
    "upsample_kernel_sizes": params.upsample_kernel_sizes,
    "cmu_dict" : cmu_dict,
    "wandb_project" : wandb_project,
    "wandb_name" : wandb_name,
    "wandb_resume" : wandb_resume,
    "resume" : resume,
    "ckpt" : ckpt,
    "note" : note
}

check_path = log_dir

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Wandb Starting....///')
    if resume:
        check_path = f"{check_path.split('/')[0]}/{wandb_name}" # 내가 직접 수정
        wandb_path = check_path.split('/')[-1]
        wandb.init(
            name=f"{wandb_path}", project=wandb_project,  resume=wandb_resume,
            id=''
        )
        log_dir = check_path
    else:
        wandb_path = check_path.split('/')[-1]
        wandb.init(
            name=f"{wandb_path}", project=wandb_project,  resume=wandb_resume
        )

        print('Initializing logger...')
        os.makedirs(log_dir, exist_ok=True)

    print('Logging parameter config...')
    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(params_dict, f, indent=4)
        

    print('Initializing data loaders...')
    ##################### train dataset #######################################################
    train_dataset = TextMelSpeakerDataset(train_filelist_path, cmu_dict, add_blank, sample_rate)

    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True, pin_memory=True,prefetch_factor=8,
                        num_workers=1, shuffle=True,persistent_workers=True)
    ############################################################################################

    test_dataset = TextMelSpeakerDataset(valid_filelist_path, cmu_dict, add_blank, sample_rate)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True, pin_memory=True,prefetch_factor=8,
                        num_workers=1, shuffle=False,persistent_workers=True)

    ############################################################################################
    print('Initializing model...')

    model = P2VC(segment_size, inter_channels, hidden_channels, n_heads, n_layers, 
                    resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                    upsample_initial_channel, upsample_kernel_sizes).cuda()


    net_d = MultiPeriodDiscriminator().cuda()
    
    optimizer_g = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.AdamW(net_d.parameters(), 2e-4, betas=[0.8, 0.99], eps=1e-9)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.9)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999875, last_epoch=-1)

    stft_loss = STFTLoss().cuda()

    wandb.watch(model)
    wandb.watch(net_d)

    if resume:
        checkpoint = torch.load(f"{check_path}/{ckpt}", map_location=lambda loc, storage: loc)
        model.load_state_dict(checkpoint['model_state_dict'])
        net_d.load_state_dict(checkpoint['netd_state_dict'])
        print('Initializing optimizer...')

        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict']) 

        scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        iteration = checkpoint['iter']
        current = checkpoint['epoch']

    else:
        current = 0
        iteration = 0

    model = nn.DataParallel(model)
    net_d = nn.DataParallel(net_d)

    print('Logging test batch...')
    ########## train test batch ##########################################
    train_test_batch = train_dataset.sample_test_batch(size=params.test_size, idx_set=[0, 1])
    for i, item in enumerate(train_test_batch):
        audio, spk = item['y'], item['spk_tgt']
        mel = mel_spectrogram(audio, 1024, 80, sample_rate, 256,
                              1024, 0, 8000, center=False)
        i = int(spk)
        save_plot(mel.squeeze(), f'{log_dir}/train_original_{i}.png')
        wandb.log({
                    "train/origin": wandb.Image(f'{log_dir}/train_original_{i}.png')
                })
    ##########################################################################

    ########### eval test batch #############################################
    eval_test_batch = test_dataset.sample_test_batch(size=params.test_size, idx_set=[0, 1])
    for i, item in enumerate(eval_test_batch):
        audio, spk = item['y'], item['spk_tgt']
        mel = mel_spectrogram(audio, 1024, 80, sample_rate, 256,
                              1024, 0, 8000, center=False)
        i = int(spk)
        # logger.add_image(f'image_{spk_idx}/ground_truth', plot_tensor(mel.squeeze()),
        #                  global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/test_original_{i}.png')
        wandb.log({
                    "val/origin": wandb.Image(f'{log_dir}/test_original_{i}.png')
                })

    ##########################################################################

    print('Start training...')
    model.train()

    
    for epoch in range(current+1, n_epochs + 1):
        
        spk_accs=[]
        ctc_accs=[]
        ctc_cers=[]

        spk_losses=[]
        ctc_losses=[]
        recon_losses=[]
        stft_losses=[]
        fm_losses=[]
        disc_losses=[]
        gen_losses=[]


        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                perturb_y = batch['perturb_y'].cuda()
                tgt_pho, pho_lengths = batch['tgt_pho'].cuda(), batch['pho_lengths'].cuda()
                spk_tgt = batch['spk_tgt'].cuda()

                pred_wav, ids_slice, loss_ctc, loss_spk, pho_acc, pho_cer, spk_acc = model(y, y_lengths, perturb_y, tgt_pho, pho_lengths, spk_tgt)
                

                # --------------------- multi gpu config -------------------- #
                # mi_loss = mi_loss.mean()
                loss_ctc = loss_ctc.mean()
                loss_spk = loss_spk.mean()
                pho_acc = pho_acc.mean()
                pho_cer = pho_cer.mean()
                spk_acc = spk_acc.mean()
                # --------------------------------------------- #


                y = commons.slice_segments(y.unsqueeze(1), ids_slice * 320, segment_size*320)
                ############# wav to mel ################################
                y_mel = mel_spectrogram(y.squeeze(1), 1024, 80, sample_rate, 256,
                            1024, 0, 8000, center=False).squeeze()

                y_hat_mel = mel_spectrogram(pred_wav.squeeze(1), 1024, 80, sample_rate, 256,
                            1024, 0, 8000, center=False).squeeze()
                #########################################################
                
                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, pred_wav.detach())
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

                optimizer_d.zero_grad()
                loss_disc_all.backward()
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                optimizer_d.step()
                ##############################################################################

                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, pred_wav)


                loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45 # 45는 weights

                stft_sc, stft_mag = stft_loss(pred_wav, y)   # (gen, GT)
                loss_stft = stft_sc + stft_mag

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_ctc + loss_spk + loss_stft


                optimizer_g.zero_grad()
                loss_gen_all.backward()
                grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
                optimizer_g.step()

                
                msg = f'Epoch: {epoch}, iteration: {iteration} | disc_loss: {loss_disc_all.item():.4f}, gen_loss: {loss_gen.item():.4f}, stft_loss: {loss_stft.item():.4f}, recon_loss: {loss_mel.item():.4f}, ctc_loss: {loss_ctc.item():.4f}, ctc_acc: {pho_acc:.4f}, ctc_cer: {pho_cer:.4f}, spk_loss: {loss_spk.item():.4f}, spk_acc: {spk_acc:.4f}, loss_fm: {loss_fm.item():.4f}, learning_rate: {optimizer_g.param_groups[0]["lr"]}'
                progress_bar.set_description(msg)
                spk_accs.append(spk_acc.item())

                ctc_accs.append(pho_acc.item())
                ctc_cers.append(pho_cer.item())

                ctc_losses.append(loss_ctc.item())
                spk_losses.append(loss_spk.item())
                recon_losses.append(loss_mel.item())

                stft_losses.append(loss_stft.item())
                fm_losses.append(loss_fm.item())
                disc_losses.append(loss_disc_all.item())
                gen_losses.append(loss_gen.item())
                

                wandb.log(
                    {
                        "train/lr": optimizer_g.param_groups[0]["lr"],
                        "train/loss_disc": loss_disc_all.item(),
                        "train/loss_gen_all" : loss_gen_all.item(),
                        "train/loss_gen": loss_gen.item(),
                        "train/loss_fm": loss_fm.item(),
                        "train/loss_mel": loss_mel.item(),
                        "train/loss_stft": loss_stft.item(),
                        "train/spk_loss": loss_spk.item(),
                        "train/ctc_loss": loss_ctc.item(),
                        "train/spk_acc": spk_acc,
                        "train/ctc_acc": pho_acc,
                        "train/ctc_cer": pho_cer,
                    }
                )
                iteration += 1

            record_lr = optimizer_g.param_groups[0]["lr"]
            
            scheduler_d.step()
            scheduler_g.step()


        msg = 'Epoch %d: disc loss = %.3f ' % (epoch, np.mean(disc_losses))
        msg += '| gen loss = %.3f ' % np.mean(gen_losses)
        msg += '| fm loss = %.3f ' % np.mean(fm_losses)
        msg += '| spk loss = %.3f ' % np.mean(spk_losses)
        msg += '| spk acc = %.3f ' % np.mean(spk_accs)
        msg += '| ctc loss = %.3f ' % np.mean(ctc_losses)
        msg += '| ctc acc = %.3f ' % np.mean(ctc_accs)
        msg += '| ctc cer = %.3f ' % np.mean(ctc_cers)
        msg += '| stft loss = %.3f ' % np.mean(stft_losses)
        msg += '| recon loss = %.3f ' % np.mean(recon_losses)
        msg += '| learning rate = %.6f\n' % record_lr


        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg)

        if epoch % params.save_every > 0:
            continue
        
        model.eval()
        print('Sample Synthesis...')
        with torch.no_grad():
            ################ train sample synthesis #######################
            for item in train_test_batch:
                y = item['y'].cuda()
                tgt_pho = item['tgt_pho'].unsqueeze(0).cuda()
                spk_tgt = item['spk_tgt'].unsqueeze(0).cuda()

                y_lengths = torch.LongTensor([y.shape[-1]]).cuda()
                pho_lengths = torch.LongTensor([tgt_pho.shape[-1]]).cuda()

                i = int(item['spk_tgt'])


                result = model.module.infer(y, y_lengths, tgt_pho, pho_lengths, spk_tgt)

                pred_mel = mel_spectrogram(result['recon_wav'].squeeze(1).cpu(), 1024, 80, sample_rate, 256,
                              1024, 0, 8000, center=False)

                

                save_plot(pred_mel.squeeze(), 
                          f'{log_dir}/train_generated_wav_{i}.png')
                
                wandb.log(
                    {
                        "train/output_mel": wandb.Image(f'{log_dir}/train_generated_wav_{i}.png'),
                    }
                )
            #################################################################

            ################ eval sample synthesis ##########################
            for item in eval_test_batch:
                y = item['y'].cuda()
                tgt_pho = item['tgt_pho'].unsqueeze(0).cuda()
                spk_tgt = item['spk_tgt'].unsqueeze(0).cuda()

                y_lengths = torch.LongTensor([y.shape[-1]]).cuda()
                pho_lengths = torch.LongTensor([tgt_pho.shape[-1]]).cuda()

                i = int(item['spk_tgt'])
                result = model.module.infer(y, y_lengths, tgt_pho, pho_lengths, spk_tgt)

                pred_mel = mel_spectrogram(result['recon_wav'].squeeze(1).cpu(), 1024, 80, sample_rate, 256,
                              1024, 0, 8000, center=False)


                save_plot(pred_mel.squeeze(), 
                          f'{log_dir}/eval_generated_wav_{i}.png')
                
                
                wandb.log(
                    {
                        "val/output_mel": wandb.Image(f'{log_dir}/eval_generated_wav_{i}.png')
                    }
                )
            ########################################################################

            print(f"Evaluation...")

            val_spk_loss_list = []
            val_spk_acc_list = []
            val_ctc_loss_list = []
            val_ctc_acc_list = []
            val_ctc_cer_list = []

            with tqdm(test_loader, total=len(test_dataset)//batch_size, colour='green') as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                    tgt_pho, pho_lengths = batch['tgt_pho'].cuda(), batch['pho_lengths'].cuda()
                    spk_tgt = batch['spk_tgt'].cuda()


                    # pred_wav => (B, 1, T)
                    result = model.module.infer(y, y_lengths, tgt_pho, pho_lengths, spk_tgt)

                    if result['spk_acc'] is not None:
                        val_spk_acc_list.append(result['spk_acc'])
                    if result['phoneme_loss'] is not None:
                        val_ctc_loss_list.append(result['phoneme_loss'].item())
                    if result['phoneme_acc'] is not None:
                        val_ctc_acc_list.append(result['phoneme_acc'])
                    if result['phoneme_cer'] is not None:
                        val_ctc_cer_list.append(result['phoneme_cer'])

            # mean_spk_loss = sum(val_spk_loss_list) / len(val_spk_loss_list) if val_spk_loss_list else 0.0
            mean_spk_acc  = sum(val_spk_acc_list)  / len(val_spk_acc_list)  if val_spk_acc_list else 0.0
            mean_ctc_loss = sum(val_ctc_loss_list) / len(val_ctc_loss_list) if val_ctc_loss_list else 0.0
            mean_ctc_acc  = sum(val_ctc_acc_list)  / len(val_ctc_acc_list)  if val_ctc_acc_list else 0.0
            mean_ctc_cer  = sum(val_ctc_cer_list)  / len(val_ctc_cer_list)  if val_ctc_cer_list else 0.0

            wandb.log({
                "val/spk_acc": mean_spk_acc,
                "val/ctc_loss": mean_ctc_loss,
                "val/ctc_acc": mean_ctc_acc,
                "val/ctc_cer": mean_ctc_cer,
            }) 

        torch.save({
            'model_state_dict': model.module.state_dict(),
            'netd_state_dict': net_d.module.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict(),
            'epoch': epoch,
            'iter' : iteration
        }, f=f"{log_dir}/P2VC_{epoch}.pt")
        model.train()