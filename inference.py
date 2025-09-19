import argparse, torch, torchaudio
from torch.cuda.amp import autocast
from models import P2VC

SR = 16000

# 학습 설정과 다르면 숫자만 맞춰주세요
CFG = dict(
    segment_size=25,
    inter_channels=256,
    hidden_channels=256,
    n_heads=8,
    n_layers=4,
    resblock=1,
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
    upsample_rates=[4,4,5,4],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[8, 8, 11, 8],
)

def load_wav_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)             # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)     # mono
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.to(torch.float32).clamp_(-1, 1)   # (1, T)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--tgt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')  # GPU 강제
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다. GPU 환경에서 실행해주세요.")

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True  # 성능 최적화

    # 1) 모델 로드
    model = P2VC(**CFG).to(device).eval()
    state = torch.load(args.ckpt, map_location='cpu')
    sd = state.get('state_dict', state.get('model', state))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'[warn] missing={len(missing)}, unexpected={len(unexpected)}')

    # 2) 오디오 로드 -> GPU 이동
    src = load_wav_mono_16k(args.src).pin_memory().to(device, non_blocking=True)   # (1, T)
    tgt = load_wav_mono_16k(args.tgt).pin_memory().to(device, non_blocking=True)   # (1, T)
    src_len = torch.tensor([src.shape[-1]], device=device, dtype=torch.long)
    tgt_len = torch.tensor([tgt.shape[-1]], device=device, dtype=torch.long)

    # 3) VC 실행 (AMP로 빠르게)
    with autocast(device_type='cuda', dtype=torch.float16):
        out = model.vc(src=src, src_lengths=src_len, tgt=tgt, tgt_lengths=tgt_len)
        wav = out['recon_wav']  # (1, 1, T)

    wav = wav.squeeze(1).detach().cpu()  # 저장을 위해 CPU로 이동

    # 4) 저장
    torchaudio.save(args.out, wav, sample_rate=SR, encoding='PCM_16')
    print(f'[done] saved -> {args.out}')

if __name__ == '__main__':
    main()
