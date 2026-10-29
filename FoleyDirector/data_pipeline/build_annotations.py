# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Normalize event JSONL into FoleyDirector eight-slot annotations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open() as source, args.output.open("w") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("id") or record.get("audio_name", "")).rsplit(".", 1)[0]
            if not sample_id:
                raise ValueError(f"Line {line_number}: missing id")
            slots = [{"event": "", "loudness": "", "timbre": ""} for _ in range(8)]
            normalized_events = []
            for event in record.get("events", []):
                start = float(event["start"])
                end = float(event["end"])
                if not 0 <= start < end <= args.duration:
                    raise ValueError(
                        f"Line {line_number}: invalid interval [{start}, {end}]")
                label = str(event.get("label", event.get("event", ""))).strip()
                midpoint = (start + end) / 2
                slot_index = min(7, int(midpoint / args.duration * 8))
                if label and not slots[slot_index]["event"]:
                    slots[slot_index] = {
                        "event": label,
                        "loudness": str(event.get("loudness", "")).strip(),
                        "timbre": str(event.get("timbre", "")).strip(),
                    }
                normalized_events.append({
                    "label": label,
                    "segments": [[start, end]],
                })
            target.write(json.dumps({
                "id": sample_id,
                "slots": slots,
                "events": normalized_events,
            }, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} annotations to {args.output}")


if __name__ == "__main__":
    main()
