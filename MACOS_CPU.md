# SAM 3 on macOS CPU

This fork removes CUDA-only assumptions from the SAM 3 image inference path and
provides a small text-prompt segmentation CLI for images and videos. Video files
are processed frame by frame with the image model, so no CUDA tracker is needed.
The `mps` branch also provides experimental Apple GPU acceleration; see
`MPS_PORTING.md`.

## Requirements

- Apple Silicon Mac
- Python 3.12
- Access to the gated `facebook/sam3` checkpoint on Hugging Face
- `uv` and the Hugging Face CLI

The checkpoint is downloaded at runtime and is intentionally excluded from Git.

## Install

```bash
hf auth login
uv sync
```

## Segment an image

```bash
uv run python scripts/macos_cpu_text_prompt.py \
  input.jpg \
  --prompt blade \
  --output output.jpg \
  --device cpu
```

## Segment a video

```bash
uv run python scripts/macos_cpu_text_prompt.py \
  input.mov \
  --prompt blade \
  --output output.mp4 \
  --threshold 0.5 \
  --device cpu
```

The first run downloads the checkpoint. Set `SAM3_HF_CACHE` to select a cache
directory. CPU inference is substantially slower than CUDA inference, especially
for high-resolution video.

## Notes

- `--device auto` selects MPS when available; use `--device cpu` to force CPU.
- Video audio is not copied to the generated preview.
- Raw checkpoint files, media, caches, and generated outputs are ignored by Git.
- The upstream CUDA installation and examples remain documented in `README.md`.
