import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageOps


def load_mask(mask_file: Path, label_values: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(mask_file) as img:
        mask = np.asarray(img)

    if mask.ndim == 3:
        foreground = np.any(mask > 0, axis=-1)
        flat_values = mask.reshape(-1, mask.shape[-1])
        unique_values = np.unique(flat_values, axis=0)
    else:
        foreground = np.isin(mask, label_values) if label_values else mask > 0
        unique_values = np.unique(mask)

    return foreground, unique_values


def infer_nnunet_image_path(mask_file: Path) -> Path | None:
    label_dir = mask_file.parent
    if label_dir.name not in ("labelsTr", "labelsTs"):
        return None

    dataset_dir = label_dir.parent
    images_dir = dataset_dir / ("imagesTr" if label_dir.name == "labelsTr" else "imagesTs")

    endings = [".png", ".bmp", ".tif", ".jpg", ".jpeg"]
    dataset_json = dataset_dir / "dataset.json"
    if dataset_json.is_file():
        with dataset_json.open() as f:
            file_ending = json.load(f).get("file_ending")
        if file_ending:
            endings.insert(0, file_ending)

    for ending in dict.fromkeys(endings):
        candidate = images_dir / f"{mask_file.stem}_0000{ending}"
        if candidate.is_file():
            return candidate
    return None


def save_mask_preview(foreground: np.ndarray, output_file: Path) -> None:
    preview = Image.fromarray(foreground.astype(np.uint8) * 255, mode="L")
    preview.save(output_file)


def save_overlay(
    image_file: Path,
    foreground: np.ndarray,
    output_file: Path,
    color: str,
    alpha: float,
) -> None:
    with Image.open(image_file) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")

    if image.size != (foreground.shape[1], foreground.shape[0]):
        raise RuntimeError(
            f"Image/mask size mismatch: image {image.size}, mask {(foreground.shape[1], foreground.shape[0])}"
        )

    rgb = np.asarray(image).astype(np.float32)
    overlay_color = np.asarray(ImageColor.getrgb(color), dtype=np.float32)
    rgb[foreground] = rgb[foreground] * (1 - alpha) + overlay_color * alpha
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(output_file)


def default_output(mask_file: Path, mode: str) -> Path:
    suffix = "overlay" if mode == "overlay" else "mask_preview"
    return Path.cwd() / f"{mask_file.stem}_{suffix}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visible preview or overlay for nnU-Net label masks.")
    parser.add_argument("mask", type=Path, help="Path to a label mask, for example labelsTr/case.png")
    parser.add_argument("--image", type=Path, help="Optional image to overlay the mask on.")
    parser.add_argument("--output", type=Path, help="Output PNG path. Default: current directory.")
    parser.add_argument(
        "--mode",
        choices=("auto", "mask", "overlay"),
        default="auto",
        help="auto overlays when an image is available, otherwise writes a black/white mask preview.",
    )
    parser.add_argument(
        "--label-values",
        type=int,
        nargs="+",
        help="Specific label values to visualize. Default: all nonzero labels.",
    )
    parser.add_argument("--color", default="#ff2d55", help="Overlay color. Default: #ff2d55")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity from 0 to 1. Default: 0.45")
    parser.add_argument("--no-infer-image", action="store_true", help="Do not infer imagesTr/imagesTs from labelsTr/labelsTs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_file = args.mask
    if not mask_file.is_file():
        raise FileNotFoundError(mask_file)
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    foreground, unique_values = load_mask(mask_file, args.label_values)
    mode = args.mode
    image_file = args.image
    if mode != "mask" and image_file is None and not args.no_infer_image:
        image_file = infer_nnunet_image_path(mask_file)

    if mode == "auto":
        mode = "overlay" if image_file is not None else "mask"

    if mode == "overlay" and image_file is None:
        raise RuntimeError("Overlay mode needs --image, or a mask inside labelsTr/labelsTs so the image can be inferred.")
    if image_file is not None and not image_file.is_file():
        raise FileNotFoundError(image_file)

    output_file = args.output or default_output(mask_file, mode)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if mode == "overlay":
        save_overlay(image_file, foreground, output_file, args.color, args.alpha)
    else:
        save_mask_preview(foreground, output_file)

    print(f"Mask: {mask_file}")
    if mode == "overlay" and image_file is not None:
        print(f"Image: {image_file}")
    print(f"Unique mask values: {unique_values.tolist()}")
    print(f"Foreground pixels: {int(foreground.sum())}")
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()
