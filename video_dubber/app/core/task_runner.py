from __future__ import annotations

from dataclasses import dataclass, field
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
    cleanup_intermediate_files: bool = False


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
    step_progress_changed = Signal(int)
    step_changed = Signal(str)
    step_detail_changed = Signal(str)
    status_changed = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, options: ProcessingOptions) -> None:
        super().__init__()
        self.options = options
        self._cancel_requested = False
        self.logger = TaskLogger(self.log_message.emit)
        self._overall_step_start = 0
        self._overall_step_end = 100

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
        self.step_detail_changed.emit("正在取消当前环节，请稍候...")
        self.logger.warning("收到取消请求，正在停止当前处理")

    def _check_cancel(self) -> None:
        if self._cancel_requested:
            raise RuntimeError("任务已取消")

    def _begin_step(
        self,
        text: str,
        overall_start: int,
        overall_end: int,
        detail: str,
        indeterminate: bool = False,
    ) -> None:
        self._check_cancel()
        self._overall_step_start = overall_start
        self._overall_step_end = overall_end
        self.step_changed.emit(text)
        self.step_detail_changed.emit(detail)
        self.progress_changed.emit(overall_start)
        self.step_progress_changed.emit(-1 if indeterminate else 0)
        self.status_changed.emit(text)
        self.logger.info(text)

    def _update_step_progress(self, percent: int, detail: str | None = None) -> None:
        percent = max(0, min(100, int(percent)))
        if detail is not None:
            self.step_detail_changed.emit(detail)
        self.step_progress_changed.emit(percent)
        span = self._overall_step_end - self._overall_step_start
        self.progress_changed.emit(self._overall_step_start + round(span * percent / 100))

    def _finish_step(self, detail: str | None = None) -> None:
        self._update_step_progress(100, detail)

    def _run_pipeline(self) -> ProcessingResult:
        options = self.options
        output_dir = ensure_output_dir(options.output_dir or DEFAULT_OUTPUT_DIR)

        self._begin_step("步骤 1/7：环境检查", 0, 8, "检查 FFmpeg、FFprobe、输入视频、输出目录和 API Key")
        ffmpeg, ffprobe = check_ffmpeg_and_ffprobe()
        self.logger.info(f"FFmpeg 可用：{ffmpeg}")
        self.logger.info(f"FFprobe 可用：{ffprobe}")
        self._update_step_progress(25, "FFmpeg / FFprobe 检查通过")

        video_path = validate_video_path(options.video_path)
        self.logger.info(f"输入视频：{video_path}")
        self.logger.info(f"输出目录：{output_dir}")
        self._update_step_progress(45, "输入视频和输出目录检查通过")
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
        self._update_step_progress(75, "翻译服务 API Key 检查通过")
        tts_engine = build_tts_engine(options.tts_service, options.tts_voice)
        self._finish_step("环境检查完成")

        stem = safe_stem(video_path)
        extracted_wav = unique_path(output_dir / f"{stem}_source_16k.wav")
        english_srt = unique_path(output_dir / f"{stem}_en.srt")
        chinese_srt = unique_path(output_dir / f"{stem}_zh.srt")
        bilingual_txt = unique_path(output_dir / f"{stem}_bilingual.txt")
        bilingual_srt = unique_path(output_dir / f"{stem}_bilingual.srt") if options.generate_bilingual_subtitles else None
        voice_wav = unique_path(output_dir / f"{stem}_zh_voice.wav")
        dubbed_video = unique_path(output_dir / f"{stem}_zh_dubbed.mp4")
        log_file = unique_path(output_dir / f"{stem}_process.log")

        self._begin_step(
            "步骤 2/7：提取英文音频",
            8,
            18,
            "FFmpeg 正在提取 16kHz mono WAV 音频，当前环节无精确百分比",
            indeterminate=True,
        )
        AudioExtractor().extract_wav(video_path, extracted_wav)
        self.logger.info(f"音频已提取：{extracted_wav}")
        self._finish_step(f"音频已提取：{extracted_wav.name}")

        self._begin_step(
            "步骤 3/7：识别英文字幕",
            18,
            45,
            "Whisper 正在加载模型并识别字幕；长视频这一环节可能耗时较久",
            indeterminate=False,
        )
        segments = WhisperTranscriber(
            model_name=options.whisper_model,
            language="en",
            model_path=options.whisper_model_path.strip() or None,
        ).transcribe(
            extracted_wav,
            should_cancel=lambda: self._cancel_requested,
            on_progress=lambda done, total: self._update_step_progress(
                min(99, round(done * 100 / total)) if total else 0,
                f"识别进度：{done}/{total}",
            ),
        )
        self.logger.info(f"识别完成：共 {len(segments)} 段字幕")
        self._finish_step(f"识别完成：共 {len(segments)} 段字幕")

        self._begin_step("步骤 4/7：翻译中文字幕", 45, 62, "正在按批次翻译字幕", indeterminate=False)
        segments = translator.translate_segments(
            segments,
            on_progress=lambda done, total: self._update_step_progress(
                round(done * 100 / total),
                f"翻译批次：{done}/{total}",
            ),
            should_cancel=lambda: self._cancel_requested,
        )
        self.logger.info("翻译完成，已确保中文段数与英文段数一致")
        self._finish_step("翻译完成，字幕段数已校验")

        self._begin_step("步骤 5/7：生成字幕与双语文本", 62, 68, "正在写入字幕和文本文件")
        writer = SubtitleWriter()
        writer.write_srt(segments, english_srt, "en")
        self._update_step_progress(25, f"英文 SRT 已生成：{english_srt.name}")
        writer.write_srt(segments, chinese_srt, "zh")
        self._update_step_progress(50, f"中文 SRT 已生成：{chinese_srt.name}")
        writer.write_bilingual_txt(segments, bilingual_txt)
        self._update_step_progress(75, f"双语 TXT 已生成：{bilingual_txt.name}")
        if bilingual_srt is not None:
            self._write_bilingual_srt(segments, bilingual_srt)
        self.logger.info(f"英文 SRT：{english_srt}")
        self.logger.info(f"中文 SRT：{chinese_srt}")
        self.logger.info(f"双语 TXT：{bilingual_txt}")
        if bilingual_srt is not None:
            self.logger.info(f"双语 SRT：{bilingual_srt}")
        self._finish_step("字幕与双语文本生成完成")

        self._begin_step("步骤 6/7：生成中文配音 WAV", 68, 88, "正在按中文字幕段生成 Edge TTS 语音")
        total_duration = media_duration_or_fallback(video_path, segments)
        tts_engine.synthesize_track(
            segments,
            voice_wav,
            total_duration,
            on_log=self.logger.info,
            on_progress=lambda done, total: self._update_step_progress(
                round(done * 100 / total),
                f"TTS 分段：{done}/{total}",
            ),
            should_cancel=lambda: self._cancel_requested,
        )
        self.logger.info(f"中文配音 WAV：{voice_wav}")
        self._finish_step(f"中文配音 WAV 已生成：{voice_wav.name}")

        self._begin_step(
            "步骤 7/7：合成中文配音视频",
            88,
            100,
            "FFmpeg 正在合成最终 MP4，当前环节无精确百分比",
            indeterminate=True,
        )
        VideoMuxer().mux(video_path, voice_wav, dubbed_video, options.keep_background_audio, self.logger.info)
        self.logger.info(f"中文配音视频：{dubbed_video}")
        self._finish_step(f"中文配音视频已生成：{dubbed_video.name}")

        # 清理中间文件（可选）
        if options.cleanup_intermediate_files:
            for intermediate in (extracted_wav, voice_wav):
                try:
                    intermediate.unlink(missing_ok=True)
                    self.logger.info(f"已删除中间文件：{intermediate.name}")
                except OSError as exc:
                    self.logger.warning(f"删除中间文件失败：{intermediate.name}，{exc}")

        self.progress_changed.emit(100)
        self.step_progress_changed.emit(100)
        self.step_detail_changed.emit("全部处理完成")
        self.status_changed.emit("处理完成")
        self.logger.info("全部处理完成")
        self.logger.save(log_file)

        return ProcessingResult(
            output_dir=str(output_dir),
            dubbed_video=str(dubbed_video),
            english_srt=str(english_srt),
            chinese_srt=str(chinese_srt),
            bilingual_txt=str(bilingual_txt),
            chinese_voice_wav=str(voice_wav) if not options.cleanup_intermediate_files else "",
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


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchOptions:
    """批量处理选项：video_paths 为待处理视频路径列表，其余参数与 ProcessingOptions 相同。"""
    video_paths: tuple[str, ...]
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
    cleanup_intermediate_files: bool = False

    def to_single(self, video_path: str) -> ProcessingOptions:
        return ProcessingOptions(
            video_path=video_path,
            output_dir=self.output_dir,
            whisper_model=self.whisper_model,
            whisper_model_path=self.whisper_model_path,
            source_language=self.source_language,
            target_language=self.target_language,
            translation_service=self.translation_service,
            translation_model=self.translation_model,
            tts_service=self.tts_service,
            tts_voice=self.tts_voice,
            keep_background_audio=self.keep_background_audio,
            generate_bilingual_subtitles=self.generate_bilingual_subtitles,
            cleanup_intermediate_files=self.cleanup_intermediate_files,
        )


@dataclass
class BatchItemResult:
    index: int          # 0-based
    video_path: str
    success: bool
    result: ProcessingResult | None = None
    error: str = ""


class BatchWorker(QObject):
    """串行处理多个视频，每个视频复用 VideoDubberWorker 的 pipeline 逻辑。

    信号说明：
      log_message(str)              — 日志行（含视频文件名前缀）
      progress_changed(int)         — 当前视频的单视频总进度 0-100
      step_progress_changed(int)    — 当前视频的当前步骤进度 0-100（-1 表示不确定）
      step_changed(str)             — 当前步骤描述
      step_detail_changed(str)      — 当前步骤细节
      status_changed(str)           — 状态文字
      batch_progress_changed(int, int)  — (已完成数, 总数)
      item_finished(int, dict)      — (index, result_dict) 单个视频完成
      item_failed(int, str, str)    — (index, video_path, error) 单个视频失败
      finished(list)                — 全部完成，payload 为 BatchItemResult.__dict__ 列表
      canceled()                    — 任务被取消
    """

    log_message = Signal(str)
    progress_changed = Signal(int)
    step_progress_changed = Signal(int)
    step_changed = Signal(str)
    step_detail_changed = Signal(str)
    status_changed = Signal(str)
    batch_progress_changed = Signal(int, int)   # (done, total)
    item_finished = Signal(int, dict)           # (index, result_dict)
    item_failed = Signal(int, str, str)         # (index, video_path, error)
    finished = Signal(list)
    canceled = Signal()

    def __init__(self, options: BatchOptions) -> None:
        super().__init__()
        self.options = options
        self._cancel_requested = False

    @Slot()
    def run(self) -> None:
        total = len(self.options.video_paths)
        results: list[BatchItemResult] = []

        for index, video_path in enumerate(self.options.video_paths):
            if self._cancel_requested:
                self.canceled.emit()
                return

            video_name = Path(video_path).name
            self.status_changed.emit(f"处理第 {index + 1}/{total} 个：{video_name}")
            self.batch_progress_changed.emit(index, total)
            self.log_message.emit(f"{'=' * 60}")
            self.log_message.emit(f"[批量 {index + 1}/{total}] 开始处理：{video_name}")

            # 构造一个内嵌 worker，把它的信号桥接到 BatchWorker 的信号上
            single_options = self.options.to_single(video_path)
            inner = _InlinePipelineRunner(single_options, lambda: self._cancel_requested)

            # 桥接信号
            inner.log_message = lambda line, vn=video_name: self.log_message.emit(f"[{vn}] {line}")
            inner.progress_changed = self.progress_changed.emit
            inner.step_progress_changed = self.step_progress_changed.emit
            inner.step_changed = self.step_changed.emit
            inner.step_detail_changed = self.step_detail_changed.emit

            try:
                result = inner.run_pipeline()
                item = BatchItemResult(index=index, video_path=video_path, success=True, result=result)
                results.append(item)
                self.item_finished.emit(index, result.__dict__)
                self.log_message.emit(f"[批量 {index + 1}/{total}] 完成：{video_name}")
            except RuntimeError as exc:
                msg = str(exc)
                if self._cancel_requested:
                    self.canceled.emit()
                    return
                item = BatchItemResult(index=index, video_path=video_path, success=False, error=msg)
                results.append(item)
                self.item_failed.emit(index, video_path, msg)
                self.log_message.emit(f"[批量 {index + 1}/{total}] 失败：{video_name} — {msg}")
                # 失败后继续处理下一个

        self.batch_progress_changed.emit(total, total)
        self.status_changed.emit("批量处理完成")
        self.finished.emit([r.__dict__ for r in results])

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True
        self.status_changed.emit("正在取消批量任务...")
        self.log_message.emit("收到取消请求，将在当前视频处理完成后停止")


class _InlinePipelineRunner:
    """不依赖 QThread/Signal，直接在调用线程中同步执行单视频 pipeline。
    回调通过普通 callable 注入，供 BatchWorker 桥接使用。
    """

    def __init__(self, options: ProcessingOptions, should_cancel: object) -> None:
        self.options = options
        self._should_cancel = should_cancel
        # 以下由 BatchWorker 在调用前赋值
        self.log_message = lambda line: None
        self.progress_changed = lambda v: None
        self.step_progress_changed = lambda v: None
        self.step_changed = lambda t: None
        self.step_detail_changed = lambda t: None
        self._overall_step_start = 0
        self._overall_step_end = 100

    def _check_cancel(self) -> None:
        if self._should_cancel():
            raise RuntimeError("任务已取消")

    def _begin_step(self, text: str, overall_start: int, overall_end: int, detail: str, indeterminate: bool = False) -> None:
        self._check_cancel()
        self._overall_step_start = overall_start
        self._overall_step_end = overall_end
        self.step_changed(text)
        self.step_detail_changed(detail)
        self.progress_changed(overall_start)
        self.step_progress_changed(-1 if indeterminate else 0)
        self.log_message(text)

    def _update_step_progress(self, percent: int, detail: str | None = None) -> None:
        percent = max(0, min(100, int(percent)))
        if detail is not None:
            self.step_detail_changed(detail)
        self.step_progress_changed(percent)
        span = self._overall_step_end - self._overall_step_start
        self.progress_changed(self._overall_step_start + round(span * percent / 100))

    def _finish_step(self, detail: str | None = None) -> None:
        self._update_step_progress(100, detail)

    def run_pipeline(self) -> ProcessingResult:
        """与 VideoDubberWorker._run_pipeline 逻辑完全一致，但使用 callable 回调。"""
        options = self.options
        output_dir = ensure_output_dir(options.output_dir or DEFAULT_OUTPUT_DIR)

        self._begin_step("步骤 1/7：环境检查", 0, 8, "检查 FFmpeg、FFprobe、输入视频、输出目录和 API Key")
        ffmpeg, ffprobe = check_ffmpeg_and_ffprobe()
        self.log_message(f"FFmpeg 可用：{ffmpeg}")
        self.log_message(f"FFprobe 可用：{ffprobe}")
        self._update_step_progress(25, "FFmpeg / FFprobe 检查通过")

        video_path = validate_video_path(options.video_path)
        self.log_message(f"输入视频：{video_path}")
        self.log_message(f"输出目录：{output_dir}")
        self._update_step_progress(45, "输入视频和输出目录检查通过")

        if options.whisper_model_path.strip():
            model_path = Path(options.whisper_model_path).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"本地 Whisper 模型路径不存在：{model_path}")
            if model_path.is_file() and model_path.suffix.lower() != ".pt":
                raise ValueError(f"本地 Whisper 模型文件只支持 .pt：{model_path}")
            if not model_path.is_file() and not model_path.is_dir():
                raise ValueError(f"本地 Whisper 模型路径无效：{model_path}")
            model_kind = "OpenAI Whisper .pt 模型文件" if model_path.is_file() else "faster-whisper 模型文件夹"
            self.log_message(f"使用本地 {model_kind}：{model_path}")
        else:
            self.log_message(f"使用 Whisper 模型名：{options.whisper_model}")

        translator = SubtitleTranslator(
            TranslatorConfig(service=options.translation_service, model=options.translation_model)
        )
        translator.validate_credentials()
        self._update_step_progress(75, "翻译服务 API Key 检查通过")
        tts_engine = build_tts_engine(options.tts_service, options.tts_voice)
        self._finish_step("环境检查完成")

        stem = safe_stem(video_path)
        extracted_wav = unique_path(output_dir / f"{stem}_source_16k.wav")
        english_srt = unique_path(output_dir / f"{stem}_en.srt")
        chinese_srt = unique_path(output_dir / f"{stem}_zh.srt")
        bilingual_txt = unique_path(output_dir / f"{stem}_bilingual.txt")
        bilingual_srt = unique_path(output_dir / f"{stem}_bilingual.srt") if options.generate_bilingual_subtitles else None
        voice_wav = unique_path(output_dir / f"{stem}_zh_voice.wav")
        dubbed_video = unique_path(output_dir / f"{stem}_zh_dubbed.mp4")
        log_file = unique_path(output_dir / f"{stem}_process.log")

        self._begin_step("步骤 2/7：提取英文音频", 8, 18, "FFmpeg 正在提取 16kHz mono WAV 音频", indeterminate=True)
        AudioExtractor().extract_wav(video_path, extracted_wav)
        self.log_message(f"音频已提取：{extracted_wav}")
        self._finish_step(f"音频已提取：{extracted_wav.name}")

        self._begin_step("步骤 3/7：识别英文字幕", 18, 45, "Whisper 正在加载模型并识别字幕", indeterminate=False)
        segments = WhisperTranscriber(
            model_name=options.whisper_model,
            language="en",
            model_path=options.whisper_model_path.strip() or None,
        ).transcribe(
            extracted_wav,
            should_cancel=self._should_cancel,
            on_progress=lambda done, total: self._update_step_progress(
                min(99, round(done * 100 / total)) if total else 0,
                f"识别进度：{done}/{total}",
            ),
        )
        self.log_message(f"识别完成：共 {len(segments)} 段字幕")
        self._finish_step(f"识别完成：共 {len(segments)} 段字幕")

        self._begin_step("步骤 4/7：翻译中文字幕", 45, 62, "正在按批次翻译字幕")
        segments = translator.translate_segments(
            segments,
            on_progress=lambda done, total: self._update_step_progress(
                round(done * 100 / total), f"翻译批次：{done}/{total}"
            ),
            should_cancel=self._should_cancel,
        )
        self.log_message("翻译完成，已确保中文段数与英文段数一致")
        self._finish_step("翻译完成，字幕段数已校验")

        self._begin_step("步骤 5/7：生成字幕与双语文本", 62, 68, "正在写入字幕和文本文件")
        writer = SubtitleWriter()
        writer.write_srt(segments, english_srt, "en")
        self._update_step_progress(25, f"英文 SRT 已生成：{english_srt.name}")
        writer.write_srt(segments, chinese_srt, "zh")
        self._update_step_progress(50, f"中文 SRT 已生成：{chinese_srt.name}")
        writer.write_bilingual_txt(segments, bilingual_txt)
        self._update_step_progress(75, f"双语 TXT 已生成：{bilingual_txt.name}")
        if bilingual_srt is not None:
            _write_bilingual_srt(segments, bilingual_srt)
        self._finish_step("字幕与双语文本生成完成")

        self._begin_step("步骤 6/7：生成中文配音 WAV", 68, 88, "正在按中文字幕段生成 Edge TTS 语音")
        total_duration = media_duration_or_fallback(video_path, segments)
        tts_engine.synthesize_track(
            segments,
            voice_wav,
            total_duration,
            on_log=self.log_message,
            on_progress=lambda done, total: self._update_step_progress(
                round(done * 100 / total), f"TTS 分段：{done}/{total}"
            ),
            should_cancel=self._should_cancel,
        )
        self.log_message(f"中文配音 WAV：{voice_wav}")
        self._finish_step(f"中文配音 WAV 已生成：{voice_wav.name}")

        self._begin_step("步骤 7/7：合成中文配音视频", 88, 100, "FFmpeg 正在合成最终 MP4", indeterminate=True)
        VideoMuxer().mux(video_path, voice_wav, dubbed_video, options.keep_background_audio, self.log_message)
        self.log_message(f"中文配音视频：{dubbed_video}")
        self._finish_step(f"中文配音视频已生成：{dubbed_video.name}")

        if options.cleanup_intermediate_files:
            for intermediate in (extracted_wav, voice_wav):
                try:
                    intermediate.unlink(missing_ok=True)
                    self.log_message(f"已删除中间文件：{intermediate.name}")
                except OSError as exc:
                    self.log_message(f"删除中间文件失败：{intermediate.name}，{exc}")

        self.progress_changed(100)
        self.step_progress_changed(100)
        self.step_detail_changed("全部处理完成")
        self.log_message("全部处理完成")

        # 保存日志
        logger_lines: list[str] = []
        # （inline runner 不维护独立 TaskLogger，日志已通过 log_message 回调输出）
        log_file.write_text("（日志已输出到主窗口日志区）\n", encoding="utf-8")

        return ProcessingResult(
            output_dir=str(output_dir),
            dubbed_video=str(dubbed_video),
            english_srt=str(english_srt),
            chinese_srt=str(chinese_srt),
            bilingual_txt=str(bilingual_txt),
            chinese_voice_wav=str(voice_wav) if not options.cleanup_intermediate_files else "",
            log_file=str(log_file),
            bilingual_srt=str(bilingual_srt) if bilingual_srt else None,
        )


def _write_bilingual_srt(segments, path: Path) -> Path:
    """模块级辅助函数，供 _InlinePipelineRunner 使用。"""
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.extend([
            str(index),
            f"{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}",
            segment.original_text.strip(),
            segment.chinese_text.strip(),
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
