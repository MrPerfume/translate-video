from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.config.settings import DEFAULT_OUTPUT_DIR
from app.core.audio_extractor import AudioExtractor
from app.core.subtitle_writer import SubtitleWriter
from app.core.transcriber import WhisperTranscriber
from app.core.translator import SubtitleTranslator, TranslatorConfig
from app.core.tts_engine import build_tts_engine, media_duration_or_fallback
from app.core.video_muxer import VideoMuxer
from app.utils.ffmpeg_utils import check_ffmpeg_and_ffprobe
from app.utils.file_utils import ensure_output_dir, safe_stem, unique_path, validate_video_path
from app.utils.logger import TaskLogger
from app.utils.time_utils import seconds_to_srt_time


@dataclass(frozen=True)
class ProcessingOptions:
    video_path: str
    output_dir: str
    whisper_model: str = "small"
    whisper_model_path: str = ""
    source_language: str = "English"
    target_language: str = "Chinese Simplified"
    translation_service: str = "DeepSeek"
    translation_model: str = "deepseek-v4-flash"
    tts_service: str = "Edge TTS"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    keep_background_audio: bool = False
    generate_bilingual_subtitles: bool = True


@dataclass(frozen=True)
class ProcessingResult:
    output_dir: str
    dubbed_video: str
    english_srt: str
    chinese_srt: str
    bilingual_txt: str
    chinese_voice_wav: str
    log_file: str
    bilingual_srt: str | None = None


class VideoDubberWorker(QObject):
    log_message = Signal(str)
    progress_changed = Signal(int)
    step_changed = Signal(str)
    status_changed = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, options: ProcessingOptions) -> None:
        super().__init__()
        self.options = options
        self._cancel_requested = False
        self.logger = TaskLogger(self.log_message.emit)

    @Slot()
    def run(self) -> None:
        try:
            result = self._run_pipeline()
            if self._cancel_requested:
                self.canceled.emit()
            else:
                self.finished.emit(result.__dict__)
        except Exception as exc:
            if self._cancel_requested:
                self.canceled.emit()
            else:
                message = str(exc) or exc.__class__.__name__
                self.logger.error(message)
                self.failed.emit(message)

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True
        self.status_changed.emit("正在取消任务，正在停止当前处理...")
        self.logger.warning("收到取消请求，正在停止当前处理")

    def _check_cancel(self) -> None:
        if self._cancel_requested:
            raise RuntimeError("任务已取消")

    def _set_step(self, text: str, progress: int) -> None:
        self._check_cancel()
        self.step_changed.emit(text)
        self.progress_changed.emit(progress)
        self.status_changed.emit(text)
        self.logger.info(text)

    def _run_pipeline(self) -> ProcessingResult:
        options = self.options
        output_dir = ensure_output_dir(options.output_dir or DEFAULT_OUTPUT_DIR)

        self._set_step("步骤 1/7：环境检查", 5)
        ffmpeg, ffprobe = check_ffmpeg_and_ffprobe()
        self.logger.info(f"FFmpeg 可用：{ffmpeg}")
        self.logger.info(f"FFprobe 可用：{ffprobe}")

        video_path = validate_video_path(options.video_path)
        self.logger.info(f"输入视频：{video_path}")
        self.logger.info(f"输出目录：{output_dir}")
        if options.whisper_model_path.strip():
            model_path = Path(options.whisper_model_path).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"本地 Whisper 模型路径不存在：{model_path}")
            if model_path.is_file() and model_path.suffix.lower() != ".pt":
                raise ValueError(f"本地 Whisper 模型文件只支持 .pt：{model_path}")
            if not model_path.is_file() and not model_path.is_dir():
                raise ValueError(f"本地 Whisper 模型路径无效：{model_path}")
            model_kind = "OpenAI Whisper .pt 模型文件" if model_path.is_file() else "faster-whisper 模型文件夹"
            self.logger.info(f"使用本地 {model_kind}：{model_path}")
        else:
            self.logger.info(f"使用 Whisper 模型名：{options.whisper_model}")

        translator = SubtitleTranslator(
            TranslatorConfig(service=options.translation_service, model=options.translation_model)
        )
        translator.validate_credentials()
        tts_engine = build_tts_engine(options.tts_service, options.tts_voice)

        stem = safe_stem(video_path)
        extracted_wav = unique_path(output_dir / f"{stem}_source_16k.wav")
        english_srt = unique_path(output_dir / f"{stem}_en.srt")
        chinese_srt = unique_path(output_dir / f"{stem}_zh.srt")
        bilingual_txt = unique_path(output_dir / f"{stem}_bilingual.txt")
        bilingual_srt = unique_path(output_dir / f"{stem}_bilingual.srt") if options.generate_bilingual_subtitles else None
        voice_wav = unique_path(output_dir / f"{stem}_zh_voice.wav")
        dubbed_video = unique_path(output_dir / f"{stem}_zh_dubbed.mp4")
        log_file = unique_path(output_dir / f"{stem}_process.log")

        self._set_step("步骤 2/7：提取英文音频", 12)
        AudioExtractor().extract_wav(video_path, extracted_wav)
        self.logger.info(f"音频已提取：{extracted_wav}")

        self._set_step("步骤 3/7：识别英文字幕", 25)
        segments = WhisperTranscriber(
            model_name=options.whisper_model,
            language="en",
            model_path=options.whisper_model_path.strip() or None,
        ).transcribe(extracted_wav, should_cancel=lambda: self._cancel_requested)
        self.logger.info(f"识别完成：共 {len(segments)} 段字幕")

        self._set_step("步骤 4/7：翻译中文字幕", 45)
        segments = translator.translate_segments(segments)
        self.logger.info("翻译完成，已确保中文段数与英文段数一致")

        self._set_step("步骤 5/7：生成字幕与双语文本", 60)
        writer = SubtitleWriter()
        writer.write_srt(segments, english_srt, "en")
        writer.write_srt(segments, chinese_srt, "zh")
        writer.write_bilingual_txt(segments, bilingual_txt)
        if bilingual_srt is not None:
            self._write_bilingual_srt(segments, bilingual_srt)
        self.logger.info(f"英文 SRT：{english_srt}")
        self.logger.info(f"中文 SRT：{chinese_srt}")
        self.logger.info(f"双语 TXT：{bilingual_txt}")
        if bilingual_srt is not None:
            self.logger.info(f"双语 SRT：{bilingual_srt}")

        self._set_step("步骤 6/7：生成中文配音 WAV", 72)
        total_duration = media_duration_or_fallback(video_path, segments)
        tts_engine.synthesize_track(
            segments,
            voice_wav,
            total_duration,
            on_log=self.logger.info,
            should_cancel=lambda: self._cancel_requested,
        )
        self.logger.info(f"中文配音 WAV：{voice_wav}")

        self._set_step("步骤 7/7：合成中文配音视频", 90)
        VideoMuxer().mux(video_path, voice_wav, dubbed_video, options.keep_background_audio, self.logger.info)
        self.logger.info(f"中文配音视频：{dubbed_video}")

        self.progress_changed.emit(100)
        self.status_changed.emit("处理完成")
        self.logger.info("全部处理完成")
        self.logger.save(log_file)

        return ProcessingResult(
            output_dir=str(output_dir),
            dubbed_video=str(dubbed_video),
            english_srt=str(english_srt),
            chinese_srt=str(chinese_srt),
            bilingual_txt=str(bilingual_txt),
            chinese_voice_wav=str(voice_wav),
            log_file=str(log_file),
            bilingual_srt=str(bilingual_srt) if bilingual_srt else None,
        )

    @staticmethod
    def _write_bilingual_srt(segments, path: Path) -> Path:
        lines: list[str] = []
        for index, segment in enumerate(segments, start=1):
            lines.extend(
                [
                    str(index),
                    f"{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}",
                    segment.original_text.strip(),
                    segment.chinese_text.strip(),
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
