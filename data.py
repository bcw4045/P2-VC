import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from params import seed as random_seed
import torchaudio.transforms as T
import function_f as f
from transformers import WavLMModel, AutoFeatureExtractor

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text



class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path,
                add_blank=True, sample_rate=22050,
                 ):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')

        self.sample_rate = sample_rate
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text, speaker = line[0], line[1], line[2]
        filepath = filepath

        #---------------------origin wav--------------------------#
        audio, perturb_wav, sr = self.get_audio(filepath)
        target_pho = self.get_text(text)
        spk_target = torch.tensor(int(speaker.strip()), dtype=torch.long)
        #---------------------------------------------------------#

        return (audio, target_pho, spk_target, perturb_wav)
    
    def get_audio(self, filepath):
        wav, sr = ta.load(filepath)
        if sr != self.sample_rate:
            wav, sr = self.convert_sr(wav, sr, self.sample_rate)
            max_value = torch.max(torch.abs(wav))
            wav = wav / max_value
        assert sr == self.sample_rate
        sr = self.sample_rate


        frame_size = 320
        min_num_frames = 26 
        min_length = frame_size * min_num_frames  # 8320

        cur_length = wav.size(1)  # wav shape: (1, N)
        if cur_length < min_length:
            pad_size = min_length - cur_length
            padding = torch.zeros(1, pad_size, dtype=wav.dtype, device=wav.device)
            wav = torch.cat([wav, padding], dim=1)

        
        perturb_wav = f.g(wav, sr).unsqueeze(0)
        
        min_val = torch.min(perturb_wav)
        max_val = torch.max(perturb_wav)
        perturb_wav = 2 * (perturb_wav - min_val) / (max_val - min_val) - 1

       

        return wav, perturb_wav, sr

    def convert_sr(self, wav, origin_sr, target_sr):
        resampler = T.Resample(orig_freq=origin_sr, new_freq=target_sr)
        return resampler(wav), target_sr

    def get_text(self, text):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    

    def __getitem__(self, index):
        audio, target_pho, spk_id, perturb_audio = self.get_triplet(self.filelist[index])
        item = {'y': audio, 'tgt_pho':target_pho, 'spk_tgt': spk_id, 'perturb_audio':perturb_audio}
        return item

    def __len__(self):
        return len(self.filelist)


    def sample_test_batch(self, size, idx_set=None):
        if idx_set is None:
            idx = np.random.choice(range(len(self)), size=size, replace=False)
        else:
            idx = idx_set#[14, 69]
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        # ---------------------- adjust audio lengths -------------------- #
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y = torch.zeros((B, y_max_length), dtype=torch.float32)
        # ---------------------------------------------------------------- #

        # ---------------------- adjust perturb audio lengths ------------ #
        perturb_y = torch.zeros((B, y_max_length), dtype=torch.float32)
        # ---------------------------------------------------------------- #

        pho_max_length = max([item['tgt_pho'].shape[-1] for item in batch])
        pho = torch.zeros((B, pho_max_length), dtype=torch.long)
        
        y_lengths, pho_lengths = [], []
        spk_tgt = []

        segment_idx = []

        for i, item in enumerate(batch):
            y_, spk_, pho_, perturb_y_ = item['y'], item['spk_tgt'], item["tgt_pho"], item["perturb_audio"]
            
            y_lengths.append(y_.shape[-1])
            y[i, :y_.shape[-1]] = y_
            perturb_y[i, :perturb_y_.shape[-1]] = perturb_y_


            pho_lengths.append(pho_.shape[-1])
            pho[i, :pho_.shape[-1]] = pho_

            if y_.shape[-1]//256 - 192 < 0:
                seg_idx = 0
            else:
                seg_idx = random.randint(0, y_.shape[-1]//256 - 192)
            
            segment_idx.append(seg_idx)
            spk_tgt.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        pho_lengths = torch.LongTensor(pho_lengths)
        spk = torch.stack(spk_tgt, dim=0).to(torch.long)
        segment_idx =torch.LongTensor(segment_idx)


        return {'y': y, 'y_lengths': y_lengths, 'tgt_pho':pho, 'pho_lengths': pho_lengths, 'spk_tgt': spk, 'perturb_y':perturb_y, 'seg_idx':segment_idx}


        