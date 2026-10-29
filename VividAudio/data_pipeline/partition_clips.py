# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process audio clips.')
parser.add_argument('--data_dir',
                    type=Path,
                    help='Path to the directory containing audio files',
                    required=True)
parser.add_argument('--output_dir',
                    type=Path,
                    help='Path to the output tsv file',
                    required=True)
parser.add_argument('--start', type=int, help='Start index for processing files', default=0)
parser.add_argument('--end', type=int, help='End index for processing files', default=-1)
parser.add_argument('--clip-seconds', type=float, default=8.1)
parser.add_argument('--max-segments-per-file', type=int, default=5)
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
start = args.start
end = args.end

output_data = []

blacklisted = 0
if end == -1:
    end = len(os.listdir(data_dir))
audio_files = sorted(os.listdir(data_dir))[start:end]
print(f'Processing {len(audio_files)} files from {start} to {end}')

for audio_file in tqdm(audio_files):
    audio_file_path = data_dir / audio_file
    audio_name = audio_file_path.stem

    waveform, sample_rate = torchaudio.load(audio_file_path)

    # waveform: (1/2) * length
    if waveform.shape[1] < sample_rate * args.clip_seconds:
        continue

    segment_length = int(sample_rate * args.clip_seconds)
    total_length = waveform.shape[1]
    num_segments = min(args.max_segments_per_file, total_length // segment_length)
    if num_segments > 1:
        segment_interval = (total_length - segment_length) // (num_segments - 1)
    else:
        segment_interval = 0

    for i in range(num_segments):
        start_sample = i * segment_interval
        end_sample = start_sample + segment_length
        audio_id = f'{audio_name}_{i}'
        output_data.append((audio_id, audio_name, start_sample, end_sample))

output_dir.parent.mkdir(parents=True, exist_ok=True)
output_df = pd.DataFrame(output_data, columns=['id', 'name', 'start_sample', 'end_sample'])
output_df.to_csv(output_dir, index=False, sep='\t')
