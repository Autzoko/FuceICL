"""生成分层抽样的点云/layout 正交投影质检图。"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from dev.pointnet.dataset import PointNetContextStore


def _stratified_indices(
    records: list[dict],
    count: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record["task"], record["phase"])].append(index)
    queues = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        queues.append(deque(groups[key]))
    selected = []
    while queues and len(selected) < count:
        remaining = []
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def _draw_projection(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    *,
    origin: tuple[int, int],
    axes: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    center_x, center_y = origin
    selected = points[:, axes]
    pixel_x = np.clip(center_x + selected[:, 0] * scale, center_x - 48, center_x + 48)
    pixel_y = np.clip(center_y - selected[:, 1] * scale, center_y - 42, center_y + 42)
    for x, y in zip(pixel_x.astype(int), pixel_y.astype(int), strict=True):
        draw.point((x, y), fill=color)


def visualize(
    *,
    root: Path,
    split: str,
    output: Path,
    count: int,
    seed: int,
) -> None:
    store = PointNetContextStore(root, split, cache_size=4)
    indices = _stratified_indices(store.records, count, seed)
    columns = 5
    rows = int(np.ceil(len(indices) / columns))
    card_width, card_height = 240, 145
    image = Image.new("RGB", (columns * card_width, rows * card_height), "white")
    draw = ImageDraw.Draw(image)
    scale = 210.0
    for position, index in enumerate(indices):
        record = store.records[index]
        context = store.get(record["chunk_id"])
        column, row = position % columns, position // columns
        left, top = column * card_width, row * card_height
        active = context["active_points"]
        target_relative = context["state"][6:9]
        target = context["target_points"] + target_relative
        eef = context["state"][12:15].reshape(1, 3)
        for projection, axes in enumerate(((0, 1), (0, 2))):
            origin = (left + 60 + projection * 116, top + 86)
            draw.rectangle(
                (origin[0] - 50, origin[1] - 44, origin[0] + 50, origin[1] + 44),
                outline=(215, 215, 215),
            )
            _draw_projection(
                draw,
                active,
                origin=origin,
                axes=axes,
                color=(31, 119, 180),
                scale=scale,
            )
            if bool(context["target_valid"]):
                _draw_projection(
                    draw,
                    target,
                    origin=origin,
                    axes=axes,
                    color=(255, 127, 14),
                    scale=scale,
                )
            _draw_projection(
                draw,
                eef,
                origin=origin,
                axes=axes,
                color=(214, 39, 40),
                scale=scale,
            )
        label = f"{record['task'][:22]} | {record['phase']}"
        detail = (
            f"ep{record['episode']} h{record['active_handle']} "
            f"conf={record['active_confidence']:.2f}"
        )
        draw.text((left + 5, top + 5), label, fill="black")
        draw.text((left + 5, top + 20), detail, fill=(70, 70, 70))
        draw.text((left + 38, top + 132), "XY", fill=(80, 80, 80))
        draw.text((left + 154, top + 132), "XZ", fill=(80, 80, 80))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    visualize(
        root=args.root,
        split=args.split,
        output=args.output,
        count=args.count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
