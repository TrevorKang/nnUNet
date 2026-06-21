import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "TVUS"
DEFAULT_NNUNET_RAW = Path(os.environ.get("nnUNet_raw", REPO_ROOT / "nnUNet_data" / "nnUNet_raw"))


@dataclass(frozen=True)
class TVUSCase:
    split: str
    case_id: str
    image_file: Path
    mask_file: Path


def _case_id(split: str, image_file: Path) -> str:
    stem = "".join(c if c.isalnum() else "_" for c in image_file.stem)
    return f"TVUS_{split}_{stem}"


def _find_mask(split_dir: Path, image_file: Path) -> Path:
    masks_dir = split_dir / "masks"
    expected_names = (
        f"{image_file.stem}.PNG",
        f"{image_file.stem}.png",
        f"{image_file.stem}.JPG",
        f"{image_file.stem}.jpg",
    )
    for name in expected_names:
        candidate = masks_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask found for {image_file} in {masks_dir}")


def _images_from_coco(split_dir: Path) -> list[Path]:
    annotation_file = split_dir / "_annotations.coco.json"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_file}")

    with annotation_file.open() as f:
        annotation = json.load(f)

    images = annotation.get("images", [])
    if not images:
        raise RuntimeError(f"No images listed in {annotation_file}")

    result = []
    for item in images:
        file_name = item.get("file_name")
        if not file_name:
            raise RuntimeError(f"Image entry without file_name in {annotation_file}: {item}")
        image_file = split_dir / file_name
        if not image_file.is_file():
            raise FileNotFoundError(f"Image listed in {annotation_file} does not exist: {image_file}")
        result.append(image_file)
    return result


def collect_cases(source: Path, splits: Iterable[str]) -> list[TVUSCase]:
    cases = []
    seen_case_ids = set()
    for split in splits:
        split_dir = source / split
        for image_file in _images_from_coco(split_dir):
            case_id = _case_id(split, image_file)
            if case_id in seen_case_ids:
                raise RuntimeError(f"Duplicate case id generated: {case_id}")
            seen_case_ids.add(case_id)
            cases.append(
                TVUSCase(
                    split=split,
                    case_id=case_id,
                    image_file=image_file,
                    mask_file=_find_mask(split_dir, image_file),
                )
            )
    return cases


def convert_image_to_rgb_png(input_file: Path, output_file: Path) -> tuple[int, int]:
    with Image.open(input_file) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        rgb.save(output_file, format="PNG")
        return rgb.size


def convert_mask_to_label_png(input_file: Path, output_file: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(input_file) as mask:
        label = mask.convert("L")
        if label.size != expected_size:
            raise RuntimeError(
                f"Image/mask size mismatch for {input_file}: expected {expected_size}, got {label.size}"
            )
        label = label.point(lambda p: 1 if p > 0 else 0)
        label.save(output_file, format="PNG")


def convert_cases(cases: Iterable[TVUSCase], images_dir: Path, labels_dir: Path) -> int:
    count = 0
    for case in cases:
        image_out = images_dir / f"{case.case_id}_0000.png"
        label_out = labels_dir / f"{case.case_id}.png"
        image_size = convert_image_to_rgb_png(case.image_file, image_out)
        convert_mask_to_label_png(case.mask_file, label_out, image_size)
        count += 1
    return count


def make_dataset(
    source: Path,
    nnunet_raw: Path,
    dataset_id: int,
    dataset_name: str,
    overwrite: bool,
) -> Path:
    target = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already exists. Use --overwrite to replace it.")
        shutil.rmtree(target)

    images_tr = target / "imagesTr"
    labels_tr = target / "labelsTr"
    images_ts = target / "imagesTs"
    labels_ts = target / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)

    training_cases = collect_cases(source, ("train", "valid"))
    test_cases = collect_cases(source, ("test",))

    num_training = convert_cases(training_cases, images_tr, labels_tr)
    convert_cases(test_cases, images_ts, labels_ts)

    generate_dataset_json(
        str(target),
        channel_names={0: "R", 1: "G", 2: "B"},
        labels={"background": 0, "ovary": 1},
        num_training_cases=num_training,
        file_ending=".png",
        dataset_name=dataset_name,
        description="TVUS ovary segmentation converted from JPG images and binary PNG masks.",
        overwrite_image_reader_writer="NaturalImage2DIO",
        converted_by="Codex",
    )

    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert data/TVUS to nnU-Net v2 format. Train and valid are merged into imagesTr/labelsTr; "
            "test is kept separate in imagesTs/labelsTs."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"TVUS source folder. Default: {DEFAULT_SOURCE}")
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=DEFAULT_NNUNET_RAW,
        help=f"nnUNet_raw output root. Default: {DEFAULT_NNUNET_RAW}",
    )
    parser.add_argument("--dataset-id", type=int, default=230, help="nnU-Net dataset id. Default: 230")
    parser.add_argument("--dataset-name", default="TVUS", help="nnU-Net dataset name. Default: TVUS")
    parser.add_argument("--overwrite", action="store_true", help="Replace the target dataset folder if it exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = make_dataset(
        source=args.source,
        nnunet_raw=args.nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
    )
    print(f"Converted TVUS dataset to: {output}")
