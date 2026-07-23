import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4"}
MASK_ALPHA = 0.42
COLORS = np.array(
    [
        (32, 180, 255),
        (0, 220, 120),
        (255, 90, 80),
        (210, 80, 220),
        (60, 240, 240),
        (230, 170, 50),
        (120, 120, 255),
        (80, 255, 180),
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM 3 text prompts on macOS CPU.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def segment_frame(frame_bgr, processor, prompt):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    state = processor.set_image(Image.fromarray(image_rgb))
    result = processor.set_text_prompt(prompt, state)
    scores = as_numpy(result["scores"]).astype(float)
    boxes = as_numpy(result["boxes"]).astype(float)
    masks = as_numpy(result["masks"]).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    order = np.argsort(-scores)
    overlay = frame_bgr.copy()
    labels = frame_bgr.copy()
    for index, (score, box, mask) in enumerate(
        zip(scores[order], boxes[order], masks[order])
    ):
        color = tuple(int(value) for value in COLORS[index % len(COLORS)])
        overlay[mask] = color
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(labels, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            labels,
            f"{prompt} {score:.2f}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, MASK_ALPHA, labels, 1.0 - MASK_ALPHA, 0)


def process_image(path, output, processor, prompt):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), segment_frame(frame, processor, prompt)):
        raise RuntimeError(f"Cannot write image: {output}")


def process_video(path, output, processor, prompt):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video: {output}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(segment_frame(frame, processor, prompt))
    finally:
        capture.release()
        writer.release()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    model = build_sam3_image_model(device="cpu").eval()
    processor = Sam3Processor(
        model, confidence_threshold=args.threshold, device="cpu"
    )
    if args.input.suffix.lower() in VIDEO_SUFFIXES:
        process_video(args.input, args.output, processor, args.prompt)
    else:
        process_image(args.input, args.output, processor, args.prompt)


if __name__ == "__main__":
    main()
