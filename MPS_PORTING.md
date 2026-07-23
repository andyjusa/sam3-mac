# MPS porting status

SAM 3 image inference is viable on Apple Silicon with PyTorch MPS. The image
encoder and text-prompt grounding path complete on MPS without enabling global
CPU fallback on the tested environment.

## Tested environment

- Apple Silicon macOS 26.5.2
- Python 3.12
- PyTorch 2.12.1
- Input: `assets/images/truck.jpg` at 1800 x 1200
- Prompt: `truck`

Observed warm-run stages:

- Model construction and checkpoint loading: 4.8 seconds
- Image encoding: 2.0 seconds
- Text prompt and mask generation: 2.5 seconds
- Result: one mask with score 0.865

These numbers are a feasibility measurement, not a benchmark. Unified memory
pressure and thermal state can materially affect performance.

## Compatibility change

PyTorch MPS does not implement `aten::_assert_async`. The MPS path performs the
same invariant checks synchronously with `Tensor.item()`. CUDA and CPU retain
the original asynchronous assertion behavior.

Run the CLI with explicit MPS selection:

```bash
uv run python scripts/macos_cpu_text_prompt.py \
  assets/images/truck.jpg \
  --prompt truck \
  --output outputs/truck-mps.jpg \
  --device mps
```

`--device auto` selects MPS when available and otherwise uses CPU.

## Current scope and risks

- Image text-prompt segmentation is verified.
- Frame-by-frame video through the image model uses the same verified path.
- The stateful video tracker and SAM 3.1 multiplex tracker are not yet verified
  on MPS and still contain CUDA-specific loading and memory-management paths.
- MPS does not provide CUDA mixed precision or Triton kernels. The port relies
  on float32 and existing generic PyTorch fallbacks.
- Large images and long videos can exceed unified memory; process videos one
  frame at a time and close other memory-heavy applications.
