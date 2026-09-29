"""复制 source trajectories，并在副本中显式写入多视角 replay 配置。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


OPERATIONS = ("toward", "away")
ENV_IDS = {
    "toward": "RotatedLayoutSlideToward-v0",
    "away": "RotatedLayoutSlideAway-v0",
}
CAMERA_VARIANT = "dual-fixed-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _patched_metadata(
    metadata: dict[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
    env_info = metadata.get("env_info")
    if not isinstance(env_info, dict):
        raise ValueError("source metadata 缺少 env_info")
    if env_info.get("env_id") != ENV_IDS[operation]:
        raise ValueError(f"{operation} env_id 不匹配")
    env_kwargs = env_info.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise ValueError("source metadata 缺少 env_kwargs")
    if env_kwargs.get("operation") != operation:
        raise ValueError(f"{operation} operation metadata 不匹配")
    if "camera_variant" in env_kwargs:
        raise ValueError("source metadata 已含 camera_variant，拒绝二次修改")
    env_kwargs["camera_variant"] = CAMERA_VARIANT
    return metadata


def run(
    *,
    project_root: Path,
    source_root: Path,
    output_root: Path,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    temporary = output_root.with_name(
        f".{output_root.name}.incomplete-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"临时目录已存在：{temporary}")
    files: dict[str, dict[str, str]] = {}
    try:
        temporary.mkdir(parents=True)
        for operation in OPERATIONS:
            source_dir = source_root / operation
            target_dir = temporary / operation
            source_h5 = source_dir / "trajectory.h5"
            source_json = source_dir / "trajectory.json"
            if not source_h5.is_file() or not source_json.is_file():
                raise FileNotFoundError(f"{operation} source trajectory 不完整")
            target_dir.mkdir()
            target_h5 = target_dir / source_h5.name
            target_json = target_dir / source_json.name
            shutil.copy2(source_h5, target_h5)
            metadata = _patched_metadata(
                json.loads(source_json.read_text(encoding="utf-8")),
                operation=operation,
            )
            target_json.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            files[operation] = {
                "source_h5_sha256": _sha256(source_h5),
                "source_json_sha256": _sha256(source_json),
                "copied_h5_sha256": _sha256(target_h5),
                "patched_json_sha256": _sha256(target_json),
            }
            if files[operation]["source_h5_sha256"] != files[operation][
                "copied_h5_sha256"
            ]:
                raise ValueError(f"{operation} HDF5 copy hash 不匹配")
        report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(project_root),
            "camera_variant": CAMERA_VARIANT,
            "source_root": str(source_root),
            "files": files,
        }
        (temporary / "multiview_prepare_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        project_root=arguments.project_root.resolve(),
        source_root=arguments.source_root.resolve(),
        output_root=arguments.output_root.resolve(),
    )
