"""
RIR 데이터셋 사전 생성 스크립트
=====================================
gpuRIR로 혼합 신호와 타겟 직접음을 미리 시뮬레이션해 wav 파일로 저장.

저장 구조:
    {out_dir}/
        train/
            metadata_{shard}.csv   (idx, mixture, enrollment, target, n_src, rt60, snr)
            000000/
                mixture.wav        (n_mics 채널, 잔향+노이즈 혼합)
                enrollment.wav     (1채널, 동일 화자의 다른 발화 clean)
                target.wav         (1채널, ref mic 직접음)
            000001/ ...
        val/
            metadata_{shard}.csv
            000000/ ...

사용법 (단일 GPU):
    python generate_rir_dataset.py \\
        --config ../../DeFTMamba/config.yaml \\
        --audio_dir ../../dataset/LibriSpeech \\
        --noise_dir ../../dataset/wham_noise \\
        --out_dir   ../../dataset/rir_dataset \\
        --gpu 0

사용법 (멀티 GPU 병렬):
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$((i+4)) python generate_rir_dataset.py \\
            --config ../../DeFTMamba/config.yaml \\
            --out_dir ../../dataset/rir_dataset \\
            --shard $i --n_shards 4 --gpu $((i+4)) &
    done
    wait
"""

import os
import math
import random
import argparse
import csv
from collections import defaultdict

import numpy as np
import scipy.signal
import soundfile as sf
import yaml
from tqdm import tqdm

import gpuRIR


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def acoustic_power(s):
    w, o = 640, 320
    s = np.ascontiguousarray(s)
    sh = (s.size - w + 1, w)
    S = np.lib.stride_tricks.as_strided(s, strides=s.strides * 2, shape=sh)[::o]
    wp = np.mean(S ** 2, axis=-1)
    th = 0.01 * wp.max()
    return np.mean(wp[wp > th])


def sph2cart(az_deg, el_deg, r):
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    return np.array([
        r * np.cos(el) * np.cos(az),
        r * np.cos(el) * np.sin(az),
        r * np.sin(el),
    ])


def default_mic_array(center, n_mics=4, r=0.042):
    offsets = np.array([
        sph2cart( 45,  35, r),
        sph2cart(-45, -35, r),
        sph2cart(135, -35, r),
        sph2cart(-135, 35, r),
    ])
    return center + offsets[:n_mics]


def simulate_rir(source_wav, traj_pts, mic_pos, room_sz, beta, fs,
                 nb_img, Tmax, Tdiff, timestamps):
    n_samples = len(source_wav)

    RIRs = gpuRIR.simulateRIR(
        room_sz, beta, traj_pts, mic_pos,
        nb_img, Tmax, fs, Tdiff=Tdiff,
        orV_rcv=None, mic_pattern='omni')
    reverb = gpuRIR.simulateTrajectory(
        source_wav, RIRs, timestamps=timestamps, fs=fs)[:n_samples]

    dp_RIRs = gpuRIR.simulateRIR(
        room_sz, beta, traj_pts, mic_pos,
        [1, 1, 1], 0.1, fs,
        orV_rcv=None, mic_pattern='omni')
    direct = gpuRIR.simulateTrajectory(
        source_wav, dp_RIRs, timestamps=timestamps, fs=fs)[:n_samples]

    return reverb, direct


# ─────────────────────────────────────────────
#  파일 수집
# ─────────────────────────────────────────────

def collect_files_by_speaker(audio_dir):
    """LibriSpeech 구조에서 speaker_id별 파일 목록 반환.
    speaker_id = audio_dir 기준 두 번째 서브디렉토리 이름.
    예: train-clean-100/289/... → speaker_id = '289'

    반환: (by_speaker, target_speakers)
        by_speaker     : {speaker_id: [모든 파일]}  — 간섭 화자 포함
        target_speakers: 파일 >= 2개인 화자 목록    — enrollment용 다른 발화 보장
    """
    by_speaker = defaultdict(list)
    for root, _, fnames in os.walk(audio_dir):
        for f in fnames:
            if not (f.endswith('.wav') or f.endswith('.flac')):
                continue
            rel = os.path.relpath(root, audio_dir)
            parts = rel.split(os.sep)
            speaker_id = parts[1] if len(parts) >= 2 else parts[0]
            by_speaker[speaker_id].append(os.path.join(root, f))

    if not by_speaker:
        raise FileNotFoundError(f"wav/flac 파일 없음: {audio_dir}")

    for k in by_speaker:
        by_speaker[k].sort()

    target_speakers = [spk for spk, files in by_speaker.items() if len(files) >= 2]
    if not target_speakers:
        raise RuntimeError("파일이 2개 이상인 화자가 없습니다. enrollment용 다른 발화를 확보할 수 없습니다.")

    print(f"  전체 화자: {len(by_speaker)}, 타겟 가능(파일≥2): {len(target_speakers)}")
    return dict(by_speaker), target_speakers


def collect_noise_files(noise_dir):
    files = [
        os.path.join(root, f)
        for root, _, fnames in os.walk(noise_dir)
        for f in fnames if f.endswith('.wav')
    ]
    if not files:
        raise FileNotFoundError(f"wav 파일 없음: {noise_dir}")
    return files


def load_wav(path, fs, n_samples):
    """wav 로드 후 n_samples로 crop/pad. (crop_start, wav) 반환."""
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != fs:
        wav = scipy.signal.resample_poly(wav, fs, sr)
    n = int(n_samples)
    if len(wav) >= n:
        st = random.randint(0, len(wav) - n)
        return wav[st:st + n].astype(np.float32), st
    else:
        return np.pad(wav, (0, n - len(wav))).astype(np.float32), 0


def load_noise(path, fs, n_samples, n_mics):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != fs:
        wav = scipy.signal.resample_poly(wav, fs, sr)
    total = int(n_samples) * n_mics
    wav_rep = np.tile(wav, math.ceil(total / len(wav)))[:total]
    noise_mc = np.stack(
        [wav_rep[i * int(n_samples):(i + 1) * int(n_samples)] for i in range(n_mics)],
        axis=1)
    return noise_mc.astype(np.float32)


# ─────────────────────────────────────────────
#  샘플 1개 생성
# ─────────────────────────────────────────────

def pick_enrollment(speaker_files, tgt_path, fs, n_samp):
    """동일 화자의 다른 발화(파일)에서 enrollment 로드.
    speaker_files는 반드시 tgt_path 외의 파일을 포함해야 함 (호출 전 보장).
    """
    candidates = [f for f in speaker_files if f != tgt_path]
    wav, sr = sf.read(random.choice(candidates))
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != fs:
        wav = scipy.signal.resample_poly(wav, fs, sr)
    n = int(n_samp)
    if len(wav) >= n:
        st = random.randint(0, len(wav) - n)
        return wav[st:st + n].astype(np.float32)
    return np.pad(wav, (0, n - len(wav))).astype(np.float32)


def generate_sample(by_speaker, target_speakers, noise_files, p):
    fs     = p['fs']
    n_samp = int(p['sec'] * fs)
    n_mics = p['n_mics']
    nb_pts = int(p['sec'] * 10)

    # 방 파라미터
    w = p['room_w_min'] + random.random() * (p['room_w_max'] - p['room_w_min'])
    l = p['room_l_min'] + random.random() * (p['room_l_max'] - p['room_l_min'])
    h = p['room_h_min'] + random.random() * (p['room_h_max'] - p['room_h_min'])
    room_sz = np.array([w, l, h])

    rt60 = p['rt60_min'] + random.random() * (p['rt60_max'] - p['rt60_min'])
    snr  = p['snr_min']  + random.random() * (p['snr_max']  - p['snr_min'])

    center  = room_sz / 2
    mic_pos = default_mic_array(center, n_mics)

    abs_w  = [0.8] * 5 + [0.5]
    beta   = gpuRIR.beta_SabineEstimation(room_sz, rt60, abs_weights=abs_w)
    Tdiff  = gpuRIR.att2t_SabineEstimator(12, rt60)
    Tmax   = gpuRIR.att2t_SabineEstimator(40, rt60)
    nb_img = gpuRIR.t2n(Tdiff, room_sz)
    timestamps = np.arange(nb_pts) * p['sec'] / nb_pts

    margin   = p['room_margin']
    all_speakers = list(by_speaker.keys())
    n_src    = random.randint(p['n_src_min'], p['n_src_max'])

    # 타겟은 반드시 target_speakers(파일≥2)에서 먼저 선택
    tgt_spk = random.choice(target_speakers)
    tgt_file_for_mix = random.choice(by_speaker[tgt_spk])

    reverb_list, direct_list, src_files, src_speakers = [], [], [], []
    used_speakers = {tgt_spk}

    # 첫 번째 소스 = 타겟 화자
    wav, _ = load_wav(tgt_file_for_mix, fs, n_samp)
    lo  = np.full(3, margin); hi = room_sz - lo
    pos = lo + np.random.random(3) * (hi - lo)
    reverb, direct = simulate_rir(wav, np.tile(pos, (nb_pts, 1)), mic_pos,
                                  room_sz, beta, fs, nb_img, Tmax, Tdiff, timestamps)
    reverb_list.append(reverb); direct_list.append(direct)
    src_files.append(tgt_file_for_mix); src_speakers.append(tgt_spk)

    for _ in range(n_src - 1):
        # 간섭 화자: 전체 화자 풀에서 중복 없이
        candidates = [s for s in all_speakers if s not in used_speakers]
        if not candidates:
            candidates = all_speakers
        spk = random.choice(candidates)
        used_speakers.add(spk)

        path = random.choice(by_speaker[spk])
        wav, _ = load_wav(path, fs, n_samp)

        lo   = np.full(3, margin)
        hi   = room_sz - lo
        pos  = lo + np.random.random(3) * (hi - lo)
        traj = np.tile(pos, (nb_pts, 1))

        reverb, direct = simulate_rir(
            wav, traj, mic_pos, room_sz, beta, fs,
            nb_img, Tmax, Tdiff, timestamps)
        reverb_list.append(reverb)
        direct_list.append(direct)
        src_files.append(path)
        src_speakers.append(spk)

    # 혼합 + 노이즈
    mix_wav = np.sum(reverb_list, axis=0)
    noise   = load_noise(random.choice(noise_files), fs, n_samp, n_mics)
    ref_pow = acoustic_power(mix_wav[:, 0])
    noi_pow = acoustic_power(noise[:, 0]) + 1e-12
    noise  *= np.sqrt(ref_pow / (10 ** (snr / 10)) / noi_pow)
    mix_wav += noise

    # 타겟은 항상 인덱스 0 (위에서 먼저 추가한 타겟 화자)
    tgt_direct = direct_list[0]

    # 동일 화자의 다른 발화 — 파일 ≥ 2 보장됨
    enroll_wav = pick_enrollment(by_speaker[tgt_spk], tgt_file_for_mix, fs, n_samp)

    # peak 정규화 (enrollment는 독립 정규화)
    scale      = np.max(np.abs(mix_wav)) + 1e-12
    mix_wav    = (mix_wav    / scale).astype(np.float32)
    tgt_direct = (tgt_direct / scale).astype(np.float32)
    enroll_wav = (enroll_wav / (np.max(np.abs(enroll_wav)) + 1e-12)).astype(np.float32)

    return mix_wav, enroll_wav, tgt_direct[:, 0], {
        'n_src': n_src,
        'rt60' : round(rt60, 4),
        'snr'  : round(snr, 2),
    }


# ─────────────────────────────────────────────
#  split 단위 생성
# ─────────────────────────────────────────────

def generate_split(split, total, start, end, by_speaker, target_speakers, noise_files, p, out_dir, fs, shard):
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    meta_path = os.path.join(split_dir, f'metadata_{shard}.csv')

    with open(meta_path, 'w', newline='') as meta_f:
        writer = csv.writer(meta_f)

        desc = f"{split} shard{shard} [{start}–{end})"
        for idx in tqdm(range(start, end), desc=desc, unit='sample'):
            sample_dir = os.path.join(split_dir, f'{idx:06d}')
            mix_path   = os.path.join(sample_dir, 'mixture.wav')
            enr_path   = os.path.join(sample_dir, 'enrollment.wav')
            tgt_path   = os.path.join(sample_dir, 'target.wav')

            if os.path.exists(mix_path) and os.path.exists(enr_path) and os.path.exists(tgt_path):
                writer.writerow([
                    idx,
                    os.path.relpath(mix_path, out_dir),
                    os.path.relpath(enr_path, out_dir),
                    os.path.relpath(tgt_path, out_dir),
                    '', '', '',
                ])
                continue

            os.makedirs(sample_dir, exist_ok=True)
            mix_wav, enroll_wav, tgt_wav, meta = generate_sample(by_speaker, target_speakers, noise_files, p)

            sf.write(mix_path, mix_wav,    fs, subtype='PCM_16')
            sf.write(enr_path, enroll_wav, fs, subtype='PCM_16')
            sf.write(tgt_path, tgt_wav,    fs, subtype='PCM_16')

            writer.writerow([
                idx,
                os.path.relpath(mix_path, out_dir),
                os.path.relpath(enr_path, out_dir),
                os.path.relpath(tgt_path, out_dir),
                meta['n_src'], meta['rt60'], meta['snr'],
            ])
            meta_f.flush()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RIR 데이터셋 사전 생성')
    parser.add_argument('--config',     required=True,  help='config.yaml 경로')
    parser.add_argument('--out_dir',    required=True,  help='저장 루트 디렉토리')
    parser.add_argument('--audio_dir',  default=None,   help='음원 루트 (config의 audio_dir 사용)')
    parser.add_argument('--noise_dir',  default=None,   help='노이즈 루트 (config의 noise_dir 사용)')
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--shard',      type=int, default=0)
    parser.add_argument('--n_shards',   type=int, default=1)
    parser.add_argument('--splits',     nargs='+', default=['train', 'val'])
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    random.seed(args.seed + args.shard)
    np.random.seed(args.seed + args.shard)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    p = cfg['data']
    fs = p['fs']

    # config에서 audio_dir / noise_dir 가져오기 (생성용 config 키)
    audio_dir = args.audio_dir or p.get('audio_dir', 'dataset/LibriSpeech')
    noise_dir = args.noise_dir or p.get('noise_dir', 'dataset/wham_noise')

    by_speaker, target_speakers = collect_files_by_speaker(audio_dir)
    noise_files = collect_noise_files(noise_dir)
    all_files   = sum(len(v) for v in by_speaker.values())
    print(f"Speakers: {len(by_speaker)},  Audio files: {all_files},  Noise files: {len(noise_files)}")

    split_sizes = {
        'train': p.get('train_size', 20000),
        'val':   p.get('val_size',   2000),
    }

    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits:
        total     = split_sizes[split]
        per_shard = math.ceil(total / args.n_shards)
        start     = args.shard * per_shard
        end       = min(start + per_shard, total)
        if start >= total:
            print(f"[{split}] shard {args.shard}: 담당 범위 없음, 건너뜀")
            continue

        print(f"[{split}] {start} ~ {end-1} ({end-start} samples, GPU {args.gpu})")
        generate_split(split, total, start, end, by_speaker, target_speakers, noise_files, p, args.out_dir, fs, args.shard)

    print("완료.")


if __name__ == '__main__':
    main()
