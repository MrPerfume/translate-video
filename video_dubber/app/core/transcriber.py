from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
        # 仅在 _transcribe_direct 路径中使用，用于向调用方上报进度
        self._on_progress_direct: Callable[[int, int], None] | None = None

    def transcribe(
        self,
        audio_path: Path,
        should_cancel: Callable[[], bool] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SubtitleSegment]:
        if should_cancel is None and on_progress is None:
            return self._transcribe_direct(audio_path)
        return self._transcribe_in_subprocess(audio_path, should_cancel, on_progress)

    def _transcribe_direct(self, audio_path: Path, on_progress: Callable[[int, int], None] | None = None) -> list[SubtitleSegment]:
        self._on_progress_direct = on_progress
        model_ref = self._resolve_model_ref()
        if self._is_openai_whisper_checkpoint(model_ref):
            return self._transcribe_openai_whisper(audio_path, model_ref)
        return self._transcribe_faster_whisper(audio_path, model_ref)

    def _transcribe_in_subprocess(
        self,
        audio_path: Path,
        should_cancel: Callable[[], bool] | None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SubtitleSegment]:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_transcribe_process_entry,
            args=(str(audio_path), self.model_name, self.language, self.model_path, result_queue),
        )
        process.daemon = True
        process.start()

        while True:
            if should_cancel and should_cancel():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                raise RuntimeError("任务已取消")

            try:
                status, payload = result_queue.get(timeout=0.5)
            except queue.Empty:
                if not process.is_alive():
                    process.join()
                    raise RuntimeError(f"Whisper 识别进程异常退出，退出码：{process.exitcode}")
                continue

            process.join(timeout=2)
            if status == "ok":
                segments = [SubtitleSegment(**item) for item in payload]
                if on_progress:
                    on_progress(len(segments), len(segments))
                return segments
            raise RuntimeError(str(payload))

    def _transcribe_faster_whisper(self, audio_path: Path, model_ref: str) -> list[SubtitleSegment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("缺少 faster-whisper，请先安装 requirements.txt 中的依赖") from exc

        model = WhisperModel(model_ref, device="auto", compute_type="auto")
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,
            beam_size=5,
        )
        segments: list[SubtitleSegment] = []
        total_duration = info.duration if info.duration else 0.0
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            segments.append(SubtitleSegment(start=float(segment.start), end=float(segment.end), original_text=text))
            # 通过 on_progress 上报已处理时长占比（仅 direct 路径可用）
            if self._on_progress_direct and total_duration > 0:
                elapsed_pct = min(99, int(segment.end / total_duration * 100))
                self._on_progress_direct(elapsed_pct, 100)
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


def _transcribe_process_entry(
    audio_path: str,
    model_name: str,
    language: str,
    model_path: str | None,
    result_queue,
) -> None:
    try:
        segments = WhisperTranscriber(
            model_name=model_name,
            language=language,
            model_path=model_path,
        )._transcribe_direct(Path(audio_path))
        result_queue.put(("ok", [segment.__dict__ for segment in segments]))
    except BaseException as exc:
        result_queue.put(("error", str(exc) or exc.__class__.__name__))
