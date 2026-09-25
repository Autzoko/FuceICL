"""诊断 RLBench 离线 mask handle 在 EEF 局部系下的多帧轨迹。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from dev.pointnet.preprocess_rlbench import (
    PointNetPreprocessConfig,
    RLBenchPointNetPreprocessor,
    detect_event_anchors,
)


def _episode_tracks(
    preprocessor: RLBenchPointNetPreprocessor,
    archive: zipfile.ZipFile,
    member: str,
) -> list[dict[str, Any]]:
    demo, _, _, prefix = preprocessor.adapter._read_episode(archive, member)
    anchors = detect_event_anchors(
        [float(observation.gripper_open) for observation in demo]
    )
    contact_frame = next(
        (anchor.frame for anchor in anchors if anchor.phase == "contact"),
        anchors[0].frame,
    )
    count = min(preprocessor.config.robot_calibration_frames, contact_frame + 1)
    frames = sorted(
        set(np.linspace(0, contact_frame, count, dtype=np.int64).tolist())
    )
    segments = {
        frame: preprocessor.adapter._frame_segments(
            archive,
            prefix,
            demo[frame],
            frame,
        )
        for frame in frames
    }
    local_tracks: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame in frames:
        pose = np.asarray(demo[frame].gripper_pose, dtype=np.float64)
        rotation = preprocessor._rotation_matrix(pose[3:7])
        for handle, segment in segments[frame].items():
            local_tracks[handle].append(rotation.T @ (segment.center - pose[:3]))

    minimum = max(2, math.ceil(0.5 * len(frames)))
    rows = []
    for handle, track in local_tracks.items():
        values = np.stack(track)
        median = np.median(values, axis=0)
        distance = float(np.linalg.norm(median))
        spread = float(np.quantile(np.linalg.norm(values - median, axis=1), 0.90))
        rows.append(
            {
                "handle": handle,
                "observations": len(track),
                "calibration_frames": len(frames),
                "distance_m": distance,
                "spread_m": spread,
                "enough_observations": len(track) >= minimum,
                "classified_robot": (
                    len(track) >= minimum
                    and distance <= preprocessor.config.robot_handle_radius_m
                    and spread <= preprocessor.config.robot_handle_spread_m
                ),
            }
        )
    return rows


def diagnose(
    *,
    data_root: Path,
    split: str,
    config: PointNetPreprocessConfig,
    episode_limit: int,
) -> dict[str, Any]:
    preprocessor = RLBenchPointNetPreprocessor(config)
    tasks = sorted(path.stem for path in data_root.joinpath(split).glob("*.zip"))
    aggregate: dict[int, dict[str, Any]] = {}
    for task in tasks:
        with zipfile.ZipFile(data_root / split / f"{task}.zip") as archive:
            members = preprocessor.adapter._episode_members(archive)[:episode_limit]
            for episode, member in members:
                for row in _episode_tracks(preprocessor, archive, member):
                    handle = int(row["handle"])
                    item = aggregate.setdefault(
                        handle,
                        {
                            "handle": handle,
                            "tasks": set(),
                            "episodes": 0,
                            "classified_robot_episodes": 0,
                            "distances_m": [],
                            "spreads_m": [],
                        },
                    )
                    item["tasks"].add(task)
                    item["episodes"] += 1
                    item["classified_robot_episodes"] += int(
                        row["classified_robot"]
                    )
                    if row["enough_observations"]:
                        item["distances_m"].append(row["distance_m"])
                        item["spreads_m"].append(row["spread_m"])

    handles = []
    for item in aggregate.values():
        distances = item.pop("distances_m")
        spreads = item.pop("spreads_m")
        item["task_count"] = len(item["tasks"])
        item["tasks"] = sorted(item["tasks"])
        item["median_distance_m"] = (
            float(np.median(distances)) if distances else None
        )
        item["median_spread_m"] = float(np.median(spreads)) if spreads else None
        item["p95_spread_m"] = (
            float(np.quantile(spreads, 0.95)) if spreads else None
        )
        if item["task_count"] >= 2 or item["classified_robot_episodes"]:
            handles.append(item)
    handles.sort(
        key=lambda item: (
            -item["task_count"],
            -item["classified_robot_episodes"],
            item["handle"],
        )
    )
    return {
        "split": split,
        "task_count": len(tasks),
        "episode_limit": episode_limit,
        "config": {
            "robot_handle_radius_m": config.robot_handle_radius_m,
            "robot_handle_spread_m": config.robot_handle_spread_m,
            "robot_calibration_frames": config.robot_calibration_frames,
            "shared_handle_task_fraction": config.shared_handle_task_fraction,
            "shared_handle_episodes": config.shared_handle_episodes,
        },
        "handles": handles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--episode-limit", type=int, default=1)
    args = parser.parse_args()
    config = PointNetPreprocessConfig.from_json(args.config)
    report = diagnose(
        data_root=args.data_root,
        split=args.split,
        config=config,
        episode_limit=args.episode_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
