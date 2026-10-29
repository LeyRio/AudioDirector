# FoleyDirector Executable and Trainable Source

This directory is the minimal source delivery for FoleyDirector. It contains
checkpoint-driven inference, distributed training, and the data construction
pipeline required by training.

The model renders the visual/global track and event-local text tracks
independently. At every transformer stage, the rendered tracks are fused over
eight temporal slots using the weights in the request. This is the executable
form of the dual-track rendering and fusion path used by FreeAudio.

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
all optional components, use `pip install -e ".[evaluation,muon]"`.

`DATA_PREFIX` must point to the auxiliary model root:

```text
$DATA_PREFIX/
├── checkpoints/
│   ├── v1-44.pth
│   ├── empty_string.pth
│   └── synchformer_state_dict.pth
└── ckpt/
    ├── DFN5B-CLIP-ViT-H-14-384/
    ├── ViT-H-14-378-quickgelu__dfn5b/
    └── bigvgan_v2_44khz_128band_512x/
```

For a 16 kHz variant, also place `best_netG.pt` under `checkpoints/`.

## Released Checkpoint

The main FreeAudio/FoleyDirector checkpoint is included at:

```text
checkpoints/foleydirector.pth
```

This is the 1,200,000-step full-data checkpoint selected by the original
FreeAudio inference workflow. It was trained with delayed fusion,
interleaved RoPE, an eight-slot FGC prompt layout, and FGC pooling. Keep these
architecture settings unchanged when integrating the checkpoint into another
inference entry point.

Verify the copied file before use:

```bash
sha256sum checkpoints/foleydirector.pth
```

Expected SHA-256:
`56d186e667ab17ee04ce6898fc6235e930ef57f08d69cb856c450e58402d7464`.

## Run

```bash
./infer.sh \
  checkpoints/foleydirector.pth \
  /path/to/video.mp4 \
  example_tracks.json \
  ./output \
  /path/to/data_root
```

The checkpoint may be either a raw state dict or a training checkpoint with a
top-level `weights` field.
The default architecture is `medium_44k`; set `MODEL_VARIANT` before invoking
the script when using another released variant.

Each track has one global prompt, eight event labels, and eight non-negative
mix weights. Weights are normalized across tracks independently for every
slot. See `example_tracks.json`.

Outputs:

```text
output/<video-name>.flac
output/<video-name>.mp4
```

Use `python infer.py --help` for sampling and precision options.

## Build Training Data

The source manifest is a tab-separated file with `id` and `label` columns.
Each ID maps to `<VIDEO_DIR>/<id>.mp4`. The event input is JSON Lines:

```json
{"id":"case_001","events":[{"start":0.2,"end":1.1,"label":"footstep","loudness":"soft","timbre":"wooden"}]}
```

Build the aligned TensorDict memmap and eight-slot FGC annotations:

```bash
./prepare_data.sh \
  /path/to/assets \
  /path/to/videos \
  /path/to/manifest.tsv \
  /path/to/events.jsonl \
  /path/to/training_data \
  8
```

The working shard directory must be empty. This prevents stale shards from a
previous extraction from entering the final memmap.

## Train

Launch one process per GPU. The optional third argument initializes from raw
weights or from the `weights` field of a training checkpoint.

```bash
./train.sh /path/to/assets /path/to/training_data /path/to/initial.pth 8
```

Resume an optimizer checkpoint without initial weights:

```bash
./train.sh /path/to/assets /path/to/training_data "" 8 \
  checkpoint=/path/to/foleydirector_ckpt_last.pth
```

Hydra overrides may be appended to either command. Training outputs default to
`./train_output/foleydirector`; set `OUTPUT_ROOT` to relocate them.

## Source Layout

```text
infer.sh
infer.py
train.sh
train.py
prepare_data.sh
data_pipeline/
example_tracks.json
foleydirector/
configs/train.yaml
pyproject.toml
requirements.txt
LICENSE
NOTICE
THIRD_PARTY_LICENSES.md
```

The internal `MMAudio` class name is retained for checkpoint compatibility.

## License

FoleyDirector is distributed under the Apache License 2.0. It includes
software originally released by Sony Research Inc. under the MIT License and
other third-party components under their respective terms. See
[`LICENSE`](./LICENSE), [`NOTICE`](./NOTICE), and
[`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md).
