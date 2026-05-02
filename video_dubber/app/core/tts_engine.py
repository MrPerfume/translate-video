from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from app.core.transcriber import SubtitleSegment
from app.utils.ffmpeg_utils import get_media_duration


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


class BaseTTSEngine:
    def synthesize_track(
        self,
        segments: list[SubtitleSegment],
        output_wav: Path,
        total_duration_seconds: float,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path:
        raise NotImplementedError


class EdgeTTSEngine(BaseTTSEngine):
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural") -> None:
        self.voice = voice

    def synthesize_track(
        self,
        segments: list[SubtitleSegment],
        output_wav: Path,
        total_duration_seconds: float,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("缺少 edge-tts，请先安装 requirements.txt 中的依赖") from exc

        duration_ms = max(int(total_duration_seconds * 1000), int(max(seg.end for seg in segments) * 1000) + 2000)
        track = AudioSegment.silent(duration=duration_ms, frame_rate=44100).set_channels(1)

        with tempfile.TemporaryDirectory(prefix="video_dubber_tts_") as tmp:
            tmp_dir = Path(tmp)
            for index, segment in enumerate(segments, start=1):
                if should_cancel and should_cancel():
                    raise RuntimeError("任务已取消")
                text = segment.chinese_text.strip()
                if not text:
                    continue
                mp3_path = tmp_dir / f"segment_{index:04d}.mp3"
                if on_log:
                    on_log(f"Edge TTS 生成第 {index}/{len(segments)} 段")
                asyncio.run(self._save_mp3(edge_tts, text, mp3_path))
                audio = AudioSegment.from_file(mp3_path)
                original_duration_ms = max(0, int((segment.end - segment.start) * 1000))
                if original_duration_ms and len(audio) > original_duration_ms and on_log:
                    over = (len(audio) - original_duration_ms) / 1000
                    on_log(f"第 {index} 段 TTS 超出原字幕时长 {over:.2f}s，已保留完整语音")
                track = track.overlay(audio, position=max(0, int(segment.start * 1000)))
                if on_progress:
                    on_progress(index, len(segments))

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        track.export(output_wav, format="wav")
        return output_wav

    async def _save_mp3(self, edge_tts_module, text: str, mp3_path: Path) -> None:
        communicate = edge_tts_module.Communicate(text=text, voice=self.voice)
        await communicate.save(str(mp3_path))


class PlaceholderTTSEngine(BaseTTSEngine):
    def __init__(self, service_name: str) -> None:
        self.service_name = service_name

    def synthesize_track(
        self,
        segments: list[SubtitleSegment],
        output_wav: Path,
        total_duration_seconds: float,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path:
        raise RuntimeError(f"{self.service_name} TTS 预留接口暂未实现，请先选择 Edge TTS")


def build_tts_engine(service: str, voice: str) -> BaseTTSEngine:
    normalized = service.lower()
    if normalized == "edge tts":
        return EdgeTTSEngine(voice=voice)
    if normalized == "openai tts":
        return PlaceholderTTSEngine("OpenAI")
    if normalized == "elevenlabs":
        return PlaceholderTTSEngine("ElevenLabs")
    raise RuntimeError(f"不支持的 TTS 服务：{service}")


def media_duration_or_fallback(video_path: Path, segments: list[SubtitleSegment]) -> float:
    duration = get_media_duration(video_path)
    if duration <= 0 and segments:
        duration = max(segment.end for segment in segments) + 2
    return duration
