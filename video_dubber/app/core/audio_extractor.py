from __future__ import annotations

from pathlib import Path

from app.utils.ffmpeg_utils import require_binary, run_command


class AudioExtractor:
    def extract_wav(self, video_path: Path, output_wav: Path) -> Path:
        ffmpeg = require_binary("ffmpeg")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_wav),
        ]
        run_command(command, "提取音频")
        return output_wav

