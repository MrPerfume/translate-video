from __future__ import annotations

import re
from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def validate_video_path(path: str | Path) -> Path:
    video_path = Path(path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"输入视频不存在：{video_path}")
    if not video_path.is_file():
        raise ValueError(f"输入路径不是文件：{video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("仅支持 mp4、mov、mkv、avi 视频文件")
    return video_path.resolve()


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError(f"输出目录无效：{output_dir}")
    return output_dir.resolve()


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem.strip() or "video"
    return re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", stem)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.with_suffix("")
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = Path(f"{base}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法生成唯一输出文件名：{path}")

