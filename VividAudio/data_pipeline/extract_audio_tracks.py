# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract mono 16 kHz WAV tracks from videos")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.manifest, sep="\t", dtype={"id": str}).to_dict("records")
    for row in tqdm(rows):
        sample_id = row["id"]
        source = args.video_dir / f"{sample_id}.mp4"
        output = args.output_dir / f"{sample_id}.wav"
        if output.exists():
            continue
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                "-t", str(args.duration), str(output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
