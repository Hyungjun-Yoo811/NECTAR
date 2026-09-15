"""
Step 3a of 4: builds the train enrollment CSV. For each target utterance,
every other utterance from the same speaker is listed as an enrollment
candidate (the mixture utterance itself is excluded to prevent leakage).
Adapted from speakerbeam's create_enrollment_csv_all.py (Copyright 2021
Brno Univ. of Technology / NTT, Katerina Zmolikova).
"""

import sys
from collections import defaultdict

mix_csv    = sys.argv[1]  # Input: mixture CSV from step2
out_enr_csv = sys.argv[2]  # Output: enrollment CSV

# speaker ID -> {utterance IDs}, utterance ID -> (file path, length)
spk2utts   = defaultdict(set)
utt2pathlen = {}
mix_ids = []

with open(mix_csv) as f:
    f.readline()  # skip header
    for line in f:
        mix_id, _, s1_path, s2_path, _, length = line.strip().split(',')
        mix_ids.append(mix_id)
        # mixture_id format: "<spk1>-<utt1>_<spk2>-<utt2>",
        # e.g. "1578-6379-0038_6415-111615-0009"
        utt1id, utt2id = mix_id.split('_')
        spk1 = utt1id.split('-')[0]
        spk2 = utt2id.split('-')[0]
        spk2utts[spk1].add(utt1id)
        spk2utts[spk2].add(utt2id)
        utt2pathlen[utt1id] = (s1_path, length)
        utt2pathlen[utt2id] = (s2_path, length)

with open(out_enr_csv, 'w') as f:
    f.write('mixture_id,utterance_id,enr_path1,length1,enr_path2,length2,...\n')
    for mix_id in mix_ids:
        utt1, utt2 = mix_id.split('_')

        # Speaker 1: every other utterance except utt1 itself
        f.write(f'{mix_id},{utt1},')
        enr_all = []
        for utt_id in spk2utts[utt1.split('-')[0]]:
            if utt_id == utt1:
                continue  # exclude the target utterance itself to avoid leakage
            enr_utt, length = utt2pathlen[utt_id]
            enr_all += [enr_utt, length]
        f.write(','.join(enr_all) + '\n')

        # Speaker 2: every other utterance except utt2 itself
        f.write(f'{mix_id},{utt2},')
        enr_all = []
        for utt_id in spk2utts[utt2.split('-')[0]]:
            if utt_id == utt2:
                continue
            enr_utt, length = utt2pathlen[utt_id]
            enr_all += [enr_utt, length]
        f.write(','.join(enr_all) + '\n')
