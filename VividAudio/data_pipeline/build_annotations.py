# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Normalize event JSONL into VividAudio RAG annotations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("id") or record.get("audio_name", "")).rsplit(".", 1)[0]
            if not sample_id:
                raise ValueError(f"Line {line_number}: missing id")
            events = []
            for event_index, event in enumerate(record.get("events", [])):
                start = float(event["start"])
                end = float(event["end"])
                if not 0 <= start < end <= args.duration:
                    raise ValueError(
                        f"Line {line_number}: invalid interval [{start}, {end}]")
                events.append({
                    "start": start,
                    "end": end,
                    "energy": float(event.get("energy", 1.0)),
                    "label": str(event.get("label", event.get("event", ""))),
                    "desc": str(event.get("desc", event.get("description", ""))),
                    "confidence": float(event.get("confidence", 1.0)),
                    "group": int(event.get("group", event_index)),
                    "neighbors": list(event.get("neighbors", [])),
                })
            output_path = args.output_dir / f"{sample_id}_events.json"
            output_path.write_text(
                json.dumps({"events": events}, ensure_ascii=False, indent=2) + "\n")
            count += 1
    print(f"Wrote {count} annotations to {args.output_dir}")


if __name__ == "__main__":
    main()
