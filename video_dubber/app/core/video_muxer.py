from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.utils.ffmpeg_utils import has_audio_stream, require_binary, run_command


class VideoMuxer:
    def mux(self, video_path: Path, voice_wav: Path, output_mp4: Path, keep_background: bool, log: Callable[[str], None]) -> Path:
        ffmpeg = require_binary("ffmpeg")
        output_mp4.parent.mkdir(parents=True, exist_ok=True)

        if keep_background and has_audio_stream(video_path):
            log("已启用背景音保留：当前 MVP 使用简单低音量混音，可能保留英文人声")
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(voice_wav),
                "-filter_complex",
                "[0:a]volume=0.18[bg];[1:a]volume=1.0[voice];[bg][voice]amix=inputs=2:duration=longest:dropout_transition=0[a]",
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_mp4),
            ]
        else:
            if keep_background:
                log("输入视频没有可用音轨，已改为仅合成中文配音")
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(voice_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_mp4),
            ]

        run_command(command, "合成中文配音视频")
        return output_mp4

