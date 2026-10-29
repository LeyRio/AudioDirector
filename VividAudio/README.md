# VividAudio Executable and Trainable Source

This directory is the minimal source delivery for VividAudio. It includes
standard/Router inference, distributed training, and the data construction
pipeline used by training.

## Install

Python 3.10 or newer and a CUDA-enabled PyTorch installation are required.

```bash
pip install torch torchvision torchaudio
pip install -e .
chmod +x infer.sh
chmod +x train.sh prepare_data.sh
```

Install optional dependencies when using the Muon optimizer or the MMAudio
evaluation pipeline:

```bash
pip install -e ".[muon]"
pip install -e ".[evaluation]"
```

The `evaluation` extra installs the public
[av-benchmark](https://github.com/hkchengrex/av-benchmark) package. To install
all optional components, use
`pip install -e ".[encoders,evaluation,muon]"`.

`DATA_PREFIX` must point to the auxiliary model root:

```text
$DATA_PREFIX/
├── checkpoints/
│   ├── v1-44.pth
│   ├── synchformer_state_dict.pth
│   ├── empty_string.pth
│   └── BEATs_iter3_plus_AS2M.pt
└── ckpt/
    ├── DFN5B-CLIP-ViT-H-14-384/
    ├── ViT-H-14-378-quickgelu__dfn5b/
    └── bigvgan_v2_44khz_128band_512x/
```

## Released Checkpoint

The main RAGAudioMM/VividAudio SFT checkpoint is included at:

```text
checkpoints/vividaudio.pth
```

This is the 400,000-step SFT checkpoint used by the original benchmark and
long-video inference workflows. Its architecture requires global RAG
conditioning and decoupled global RAG with a Q-Former extractor configured
with 32 queries and 8 layers. The bundled `infer.sh` sets these values
explicitly.

Verify the copied file before use:

```bash
sha256sum checkpoints/vividaudio.pth
```

Expected SHA-256:
`dd00836a01d13e64d9b86b0885b4c74b05bc618fa91ecac90c7ccbe4de183bf3`.

## Request Format

Inference accepts JSON Lines. Each line is one request:

```json
{
  "name": "case_001",
  "video_path": "/path/to/video.mp4",
  "prompt": "footsteps followed by a door closing",
  "negative_prompt": "",
  "rag_audio_list": ["/path/to/footsteps.wav", "/path/to/door.wav"],
  "rag_seg_in_ref": [[0.0, 2.0], [1.0, 2.0]],
  "rag_seg_in_gen": [[0.0, 2.0], [4.0, 5.0]]
}
```

The three `rag_*` arrays are parallel. Each reference interval is placed at
its corresponding generation interval. See `example_requests.jsonl`.

## Run

Standard local/global dual-path inference:

```bash
./infer.sh \
  checkpoints/vividaudio.pth \
  /path/to/requests.jsonl \
  ./output \
  normal \
  /path/to/data_root
```

Event-local Router inference with the same checkpoint:

```bash
./infer.sh \
  checkpoints/vividaudio.pth \
  /path/to/requests.jsonl \
  ./output_router \
  router \
  /path/to/data_root
```

The checkpoint may be either a raw state dict or a training checkpoint with a
top-level `weights` field. Router mode changes inference routing only and does
not change the checkpoint format.
The default architecture is `medium_44k`; set `MODEL_VARIANT` before invoking
the script when using another released variant.

## Build Training Data

The input manifest is TSV with `id` and `label` columns. Each ID maps to
`<VIDEO_DIR>/<id>.mp4`. Event annotations are JSON Lines:

```json
{"id":"case_001","events":[{"start":0.2,"end":1.1,"label":"footstep","desc":"wooden footsteps","energy":0.8,"group":0}]}
```

Build the aligned TensorDict memmap, 16 kHz reference waveforms, and per-sample
RAG annotation files:

```bash
./prepare_data.sh \
  /path/to/assets \
  /path/to/videos \
  /path/to/manifest.tsv \
  /path/to/events.jsonl \
  /path/to/training_data \
  8
```

The working shard directory must be empty. `ffmpeg` is required to extract
the reference waveforms.

## Train

```bash
./train.sh /path/to/assets /path/to/training_data /path/to/initial.pth 8
```

The optional initial file may be raw weights or a training checkpoint.
To resume optimizer state, pass a Hydra override:

```bash
./train.sh /path/to/assets /path/to/training_data "" 8 \
  checkpoint=/path/to/vividaudio_ckpt_last.pth
```

The portable pipeline defaults to waveform-based RAG construction
(`w_prefetch=false`). Training outputs default to
`./train_output/vividaudio`; set `OUTPUT_ROOT` to relocate them.

## Source Layout

```text
infer.sh
inference.py
train.sh
train.py
prepare_data.sh
data_pipeline/
example_requests.jsonl
vividaudio/
beats/
configs/train.yaml
pyproject.toml
LICENSE
NOTICE
THIRD_PARTY_LICENSES.md
```

## License

VividAudio is distributed under the MIT License and includes third-party
components under their respective terms. See [`LICENSE`](./LICENSE),
[`NOTICE`](./NOTICE), and
[`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md).
