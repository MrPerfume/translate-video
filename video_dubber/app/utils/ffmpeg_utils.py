from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"未找到 {name}，请先安装并确保它在 PATH 中")
    return path


def check_ffmpeg_and_ffprobe() -> tuple[str, str]:
    return require_binary("ffmpeg"), require_binary("ffprobe")


def run_command(command: list[str], log_prefix: str | None = None) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        prefix = f"{log_prefix}：" if log_prefix else ""
        raise FFmpegError(f"{prefix}命令执行失败\n{detail}")


def probe_json(media_path: Path) -> dict:
    ffprobe = require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise FFmpegError(completed.stderr.strip() or f"无法读取媒体信息：{media_path}")
    return json.loads(completed.stdout)


def get_media_duration(media_path: Path) -> float:
    data = probe_json(media_path)
    duration = data.get("format", {}).get("duration")
    if duration is None:
        durations = [stream.get("duration") for stream in data.get("streams", []) if stream.get("duration")]
        duration = max(durations) if durations else 0
    return float(duration or 0)


def has_audio_stream(media_path: Path) -> bool:
    data = probe_json(media_path)
    return any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))

