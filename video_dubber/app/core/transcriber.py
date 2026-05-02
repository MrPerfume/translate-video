from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleSegment:
    start: float
    end: float
    original_text: str
    chinese_text: str = ""


class WhisperTranscriber:
    def __init__(self, model_name: str, language: str = "en", model_path: str | None = None) -> None:
        self.model_name = model_name
        self.language = language
        self.model_path = model_path

    def transcribe(self, audio_path: Path) -> list[SubtitleSegment]:
        model_ref = self._resolve_model_ref()
        if self._is_openai_whisper_checkpoint(model_ref):
            return self._transcribe_openai_whisper(audio_path, model_ref)

        return self._transcribe_faster_whisper(audio_path, model_ref)

    def _transcribe_faster_whisper(self, audio_path: Path, model_ref: str) -> list[SubtitleSegment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("缺少 faster-whisper，请先安装 requirements.txt 中的依赖") from exc

        model = WhisperModel(model_ref, device="auto", compute_type="auto")
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,
            beam_size=5,
        )
        segments: list[SubtitleSegment] = []
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            segments.append(SubtitleSegment(start=float(segment.start), end=float(segment.end), original_text=text))
        if not segments:
            raise RuntimeError("Whisper 未识别到有效英文字幕")
        return segments

    def _transcribe_openai_whisper(self, audio_path: Path, checkpoint_path: str) -> list[SubtitleSegment]:
        try:
            import torch
            import whisper
        except ImportError as exc:
            raise RuntimeError("检测到 .pt 模型文件，但缺少 openai-whisper/torch，请重新安装 requirements.txt") from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(checkpoint_path, device=device)
        result = model.transcribe(str(audio_path), language=self.language, fp16=(device == "cuda"))
        segments: list[SubtitleSegment] = []
        for item in result.get("segments", []):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                SubtitleSegment(
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    original_text=text,
                )
            )
        if not segments:
            raise RuntimeError("Whisper 未识别到有效英文字幕")
        return segments

    def _resolve_model_ref(self) -> str:
        if not self.model_path:
            return self.model_name

        model_path = Path(self.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"本地 Whisper 模型路径不存在：{model_path}")
        if model_path.is_file():
            if model_path.suffix.lower() != ".pt":
                raise ValueError(f"本地 Whisper 模型文件只支持 .pt：{model_path}")
            return str(model_path)
        if not model_path.is_dir():
            raise ValueError(f"本地 Whisper 模型路径无效：{model_path}")

        snapshot_dir = self._latest_huggingface_snapshot(model_path)
        if snapshot_dir is not None:
            return str(snapshot_dir)
        return str(model_path)

    @staticmethod
    def _is_openai_whisper_checkpoint(model_ref: str) -> bool:
        return Path(model_ref).suffix.lower() == ".pt"

    @staticmethod
    def _latest_huggingface_snapshot(model_dir: Path) -> Path | None:
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None

        candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]
