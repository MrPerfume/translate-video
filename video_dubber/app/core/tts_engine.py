from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from app.core.transcriber import SubtitleSegment
from app.utils.ffmpeg_utils import get_media_duration, require_binary


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]

# 语速调整的边界：最多压缩到原时长的 80%（即最多加速 1.25x），避免语音失真
_MAX_SPEED_RATIO = 1.25
# atempo 滤镜单次最大值为 2.0，最小值为 0.5；超出需要串联
_ATEMPO_MAX = 2.0
_ATEMPO_MIN = 0.5


def _build_atempo_filter(speed: float) -> str:
    """将任意倍速分解为合法的 atempo 滤镜链。"""
    speed = max(_ATEMPO_MIN, min(speed, _ATEMPO_MAX * _ATEMPO_MAX))
    filters: list[str] = []
    remaining = speed
    while remaining > _ATEMPO_MAX + 1e-6:
        filters.append(f"atempo={_ATEMPO_MAX}")
        remaining /= _ATEMPO_MAX
    while remaining < _ATEMPO_MIN - 1e-6:
        filters.append(f"atempo={_ATEMPO_MIN}")
        remaining /= _ATEMPO_MIN
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def _speedup_audio(audio: AudioSegment, target_ms: int, log: LogCallback | None, index: int) -> AudioSegment:
    """
    若 audio 时长超过 target_ms，用 ffmpeg atempo 压缩到 target_ms。
    压缩比超过 _MAX_SPEED_RATIO 时放弃压缩并记录日志。
    """
    actual_ms = len(audio)
    if actual_ms <= target_ms or target_ms <= 0:
        return audio

    speed = actual_ms / target_ms
    if speed > _MAX_SPEED_RATIO:
        over = (actual_ms - target_ms) / 1000
        if log:
            log(
                f"第 {index} 段 TTS 超出原字幕时长 {over:.2f}s，"
                f"加速比 {speed:.2f}x 超过上限 {_MAX_SPEED_RATIO}x，保留完整语音"
            )
        return audio

    try:
        ffmpeg = require_binary("ffmpeg")
    except RuntimeError:
        over = (actual_ms - target_ms) / 1000
        if log:
            log(f"第 {index} 段 TTS 超出 {over:.2f}s，ffmpeg 不可用，跳过语速调整")
        return audio

    atempo = _build_atempo_filter(speed)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        tmp_in_path = Path(tmp_in.name)
        tmp_out_path = Path(tmp_out.name)

    try:
        audio.export(str(tmp_in_path), format="wav")
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(tmp_in_path), "-filter:a", atempo, str(tmp_out_path)],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            over = (actual_ms - target_ms) / 1000
            if log:
                log(f"第 {index} 段语速调整失败，保留原始语音（超出 {over:.2f}s）")
            return audio
        sped_up = AudioSegment.from_file(str(tmp_out_path), format="wav")
        if log:
            log(f"第 {index} 段语速调整 {speed:.2f}x，时长 {actual_ms}ms → {len(sped_up)}ms")
        return sped_up
    finally:
        tmp_in_path.unlink(missing_ok=True)
        tmp_out_path.unlink(missing_ok=True)


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
                # 自动语速调整：若 TTS 超出字幕时长则尝试压缩
                audio = _speedup_audio(audio, original_duration_ms, on_log, index)
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
    from app.utils.ffmpeg_utils import get_media_duration
    duration = get_media_duration(video_path)
    if duration <= 0 and segments:
        duration = max(segment.end for segment in segments) + 2
    return duration
