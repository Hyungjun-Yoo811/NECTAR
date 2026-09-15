"""
Step 3b of 4: builds the dev/test enrollment CSV using a fixed mixture ->
enrollment mapping (map_mixture2enrollment), so evaluation always uses the
same enrollment utterance for fair comparison across experiments.
Adapted from speakerbeam's create_enrollment_csv_fixed.py (Copyright 2021
Brno Univ. of Technology / NTT, Katerina Zmolikova).
"""

import sys

mix_csv         = sys.argv[1]  # Input: mixture CSV
map_mix2enroll  = sys.argv[2]  # Input: predefined mixture-to-enrollment mapping
out_enr_csv     = sys.argv[3]  # Output: enrollment CSV

# "s1/<mix_id>" or "s2/<mix_id>" -> (file path, length)
utt2pathlen = {}
mix_ids = []

with open(mix_csv) as f:
    f.readline()  # skip header
    for line in f:
        mix_id, _, s1_path, s2_path, _, length = line.strip().split(',')
        mix_ids.append(mix_id)
        utt2pathlen[f's1/{mix_id}'] = (s1_path, length)
        utt2pathlen[f's2/{mix_id}'] = (s2_path, length)

# Parse the map file: mixture_ID  target_utterance_ID  enrollment_ID
mix2enroll = {}
with open(map_mix2enroll) as f:
    for line in f:
        mix_id, utt_id, enroll_id = line.strip().split()
        mix2enroll[(mix_id, utt_id)] = enroll_id

with open(out_enr_csv, 'w') as f:
    f.write('mixture_id,utterance_id,enr_path1,length1\n')
    for mix_id in mix_ids:
        utt1, utt2 = mix_id.split('_')

        # Speaker 1
        enr_id = mix2enroll[(mix_id, utt1)]
        enr_utt, length = utt2pathlen[enr_id]
        f.write(f'{mix_id},{utt1},{enr_utt},{length}\n')

        # Speaker 2
        enr_id = mix2enroll[(mix_id, utt2)]
        enr_utt, length = utt2pathlen[enr_id]
        f.write(f'{mix_id},{utt2},{enr_utt},{length}\n')
