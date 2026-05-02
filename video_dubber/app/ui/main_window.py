from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import DEFAULT_OUTPUT_DIR
from app.core.task_runner import ProcessingOptions, VideoDubberWorker
from app.ui.widgets import DropVideoLabel
from app.utils.file_utils import SUPPORTED_VIDEO_EXTENSIONS


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("英文视频转中文配音工具")
        self.resize(1280, 900)
        self.setMinimumSize(980, 760)
        self.video_path: str = ""
        self.result_paths: dict[str, str] = {}
        self.thread: QThread | None = None
        self.worker: VideoDubberWorker | None = None

        self._build_ui()
        self._connect_signals()
        self._set_running(False)
        self._set_result_buttons_enabled(False)

    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._build_video_group())
        main_layout.addWidget(self._build_settings_group())
        main_layout.addWidget(self._build_execution_group())
        main_layout.addWidget(self._build_log_group(), stretch=1)
        main_layout.addWidget(self._build_result_group())

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

        save_log_action = QAction("保存日志", self)
        save_log_action.triggered.connect(self.save_log)
        self.menuBar().addAction(save_log_action)

    def _build_video_group(self) -> QGroupBox:
        group = QGroupBox("视频选择")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.choose_video_button = QPushButton("选择视频")
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("支持 mp4、mov、mkv、avi")
        row.addWidget(self.choose_video_button)
        row.addWidget(self.video_path_edit, stretch=1)

        self.drop_label = DropVideoLabel()
        layout.addLayout(row)
        layout.addWidget(self.drop_label)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("参数设置")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.whisper_model_combo.setCurrentText("small")
        self.whisper_model_combo.setMinimumWidth(180)
        self.whisper_model_path_edit = QLineEdit()
        self.whisper_model_path_edit.setPlaceholderText("可选：选择本地 faster-whisper 模型文件夹，或 OpenAI Whisper .pt 模型文件")
        self.whisper_model_path_edit.setMinimumWidth(620)
        self.choose_whisper_model_path_button = QPushButton("选择本地模型")

        self.source_language_combo = QComboBox()
        self.source_language_combo.addItems(["English"])
        self.source_language_combo.setMinimumWidth(180)
        self.target_language_combo = QComboBox()
        self.target_language_combo.addItems(["Chinese Simplified"])
        self.target_language_combo.setMinimumWidth(220)

        self.translation_service_combo = QComboBox()
        self.translation_service_combo.addItems(["DeepSeek", "OpenAI"])
        self.translation_service_combo.setCurrentText("DeepSeek")
        self.translation_service_combo.setMinimumWidth(180)

        self.translation_model_edit = QLineEdit("deepseek-v4-flash")
        self.translation_model_edit.setMinimumWidth(260)
        self.tts_service_combo = QComboBox()
        self.tts_service_combo.addItems(["Edge TTS", "OpenAI TTS", "ElevenLabs"])
        self.tts_service_combo.setCurrentText("Edge TTS")
        self.tts_service_combo.setMinimumWidth(180)

        self.tts_voice_edit = QLineEdit("zh-CN-XiaoxiaoNeural")
        self.tts_voice_edit.setMinimumWidth(360)
        self.keep_background_checkbox = QCheckBox("保留原视频背景音")
        self.bilingual_checkbox = QCheckBox("生成双语字幕")
        self.bilingual_checkbox.setChecked(True)

        self.output_dir_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self.output_dir_edit.setMinimumWidth(620)
        self.choose_output_button = QPushButton("选择输出目录")

        whisper_form = self._make_form()
        whisper_form.addRow("Whisper 模型", self.whisper_model_combo)
        model_path_row = QHBoxLayout()
        model_path_row.setSpacing(8)
        model_path_row.addWidget(self.whisper_model_path_edit, stretch=1)
        self.choose_whisper_model_file_button = QPushButton("选择模型文件")
        self.choose_whisper_model_path_button.setText("选择模型文件夹")
        self.choose_whisper_model_file_button.setFixedWidth(120)
        self.choose_whisper_model_path_button.setFixedWidth(132)
        model_path_row.addWidget(self.choose_whisper_model_file_button)
        model_path_row.addWidget(self.choose_whisper_model_path_button)
        whisper_form.addRow("本地模型路径", model_path_row)

        translation_form = self._make_form()
        language_row = QHBoxLayout()
        language_row.setSpacing(16)
        language_row.addWidget(QLabel("源语言"))
        language_row.addWidget(self.source_language_combo)
        language_row.addSpacing(20)
        language_row.addWidget(QLabel("目标语言"))
        language_row.addWidget(self.target_language_combo)
        language_row.addStretch(1)
        translation_form.addRow("语言", language_row)
        translation_row = QHBoxLayout()
        translation_row.setSpacing(16)
        translation_row.addWidget(QLabel("服务"))
        translation_row.addWidget(self.translation_service_combo)
        translation_row.addSpacing(20)
        translation_row.addWidget(QLabel("模型"))
        translation_row.addWidget(self.translation_model_edit, stretch=1)
        translation_form.addRow("翻译", translation_row)

        tts_form = self._make_form()
        tts_row = QHBoxLayout()
        tts_row.setSpacing(16)
        tts_row.addWidget(QLabel("服务"))
        tts_row.addWidget(self.tts_service_combo)
        tts_row.addSpacing(20)
        tts_row.addWidget(QLabel("中文声音"))
        tts_row.addWidget(self.tts_voice_edit, stretch=1)
        tts_form.addRow("TTS", tts_row)
        options_row = QHBoxLayout()
        options_row.setSpacing(24)
        options_row.addWidget(self.keep_background_checkbox)
        options_row.addWidget(self.bilingual_checkbox)
        options_row.addStretch(1)
        tts_form.addRow("选项", options_row)

        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        output_row.addWidget(self.output_dir_edit, stretch=1)
        self.choose_output_button.setFixedWidth(132)
        output_row.addWidget(self.choose_output_button)
        output_form = self._make_form()
        output_form.addRow("输出目录", output_row)

        layout.addLayout(whisper_form)
        layout.addLayout(translation_form)
        layout.addLayout(tts_form)
        layout.addLayout(output_form)
        return group

    def _make_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        return form

    def _build_execution_group(self) -> QGroupBox:
        group = QGroupBox("执行")
        layout = QVBoxLayout(group)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("开始处理")
        self.cancel_button = QPushButton("取消任务")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.step_label = QLabel("当前步骤：未开始")
        self.status_label = QLabel("状态：就绪")
        self.status_label.setWordWrap(True)

        layout.addLayout(button_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.step_label)
        layout.addWidget(self.status_label)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("日志")
        layout = QVBoxLayout(group)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        button_row = QHBoxLayout()
        self.copy_log_button = QPushButton("复制日志")
        self.save_log_button = QPushButton("保存日志")
        button_row.addWidget(self.copy_log_button)
        button_row.addWidget(self.save_log_button)
        button_row.addStretch(1)

        layout.addWidget(self.log_text)
        layout.addLayout(button_row)
        return group

    def _build_result_group(self) -> QGroupBox:
        group = QGroupBox("结果")
        layout = QHBoxLayout(group)
        self.open_output_button = QPushButton("打开输出文件夹")
        self.play_video_button = QPushButton("播放生成视频")
        self.view_en_srt_button = QPushButton("查看英文字幕")
        self.view_zh_srt_button = QPushButton("查看中文字幕")
        self.view_bilingual_button = QPushButton("查看双语文本")
        for button in (
            self.open_output_button,
            self.play_video_button,
            self.view_en_srt_button,
            self.view_zh_srt_button,
            self.view_bilingual_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)
        return group

    def _connect_signals(self) -> None:
        self.choose_video_button.clicked.connect(self.choose_video)
        self.drop_label.video_dropped.connect(self.set_video_path)
        self.choose_whisper_model_file_button.clicked.connect(self.choose_whisper_model_file)
        self.choose_whisper_model_path_button.clicked.connect(self.choose_whisper_model_path)
        self.choose_output_button.clicked.connect(self.choose_output_dir)
        self.translation_service_combo.currentTextChanged.connect(self._sync_default_model)
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.copy_log_button.clicked.connect(self.copy_log)
        self.save_log_button.clicked.connect(self.save_log)
        self.open_output_button.clicked.connect(lambda: self.open_path(self.result_paths.get("output_dir")))
        self.play_video_button.clicked.connect(lambda: self.open_path(self.result_paths.get("dubbed_video")))
        self.view_en_srt_button.clicked.connect(lambda: self.open_path(self.result_paths.get("english_srt")))
        self.view_zh_srt_button.clicked.connect(lambda: self.open_path(self.result_paths.get("chinese_srt")))
        self.view_bilingual_button.clicked.connect(lambda: self.open_path(self.result_paths.get("bilingual_txt")))

    def choose_video(self) -> None:
        filters = "视频文件 (*.mp4 *.mov *.mkv *.avi)"
        path, _ = QFileDialog.getOpenFileName(self, "选择英文视频", "", filters)
        if path:
            self.set_video_path(path)

    def set_video_path(self, path: str) -> None:
        suffix = Path(path).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            QMessageBox.warning(self, "视频格式不支持", "仅支持 mp4、mov、mkv、avi 视频文件")
            return
        self.video_path = path
        self.video_path_edit.setText(path)
        self.drop_label.setText(Path(path).name)

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)

    def choose_whisper_model_path(self) -> None:
        start_dir = self.whisper_model_path_edit.text().strip() or str(Path.home() / ".cache" / "huggingface" / "hub")
        path = QFileDialog.getExistingDirectory(self, "选择本地 Whisper 模型文件夹", start_dir)
        if path:
            self.whisper_model_path_edit.setText(path)

    def choose_whisper_model_file(self) -> None:
        start_dir = self.whisper_model_path_edit.text().strip() or str(Path.cwd() / "models")
        path, _ = QFileDialog.getOpenFileName(self, "选择 OpenAI Whisper .pt 模型文件", start_dir, "Whisper 模型 (*.pt)")
        if path:
            self.whisper_model_path_edit.setText(path)

    def _sync_default_model(self, service: str) -> None:
        if service == "DeepSeek":
            self.translation_model_edit.setText("deepseek-v4-flash")
        elif service == "OpenAI":
            self.translation_model_edit.setText("gpt-4o-mini")

    def start_processing(self) -> None:
        if self.thread is not None:
            QMessageBox.information(self, "任务运行中", "当前已有任务正在运行")
            return
        if not self.video_path:
            QMessageBox.warning(self, "缺少视频", "请先选择或拖拽一个视频文件")
            return

        self.log_text.clear()
        self.result_paths = {}
        self._set_result_buttons_enabled(False)
        self.progress_bar.setValue(0)

        options = ProcessingOptions(
            video_path=self.video_path,
            output_dir=self.output_dir_edit.text().strip(),
            whisper_model=self.whisper_model_combo.currentText(),
            whisper_model_path=self.whisper_model_path_edit.text().strip(),
            source_language=self.source_language_combo.currentText(),
            target_language=self.target_language_combo.currentText(),
            translation_service=self.translation_service_combo.currentText(),
            translation_model=self.translation_model_edit.text().strip() or "deepseek-v4-flash",
            tts_service=self.tts_service_combo.currentText(),
            tts_voice=self.tts_voice_edit.text().strip() or "zh-CN-XiaoxiaoNeural",
            keep_background_audio=self.keep_background_checkbox.isChecked(),
            generate_bilingual_subtitles=self.bilingual_checkbox.isChecked(),
        )

        self.thread = QThread(self)
        self.worker = VideoDubberWorker(options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.append_log)
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.step_changed.connect(lambda text: self.step_label.setText(f"当前步骤：{text}"))
        self.worker.status_changed.connect(lambda text: self.status_label.setText(f"状态：{text}"))
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.canceled.connect(self.on_canceled)
        self.worker.finished.connect(lambda _result: self._cleanup_thread())
        self.worker.failed.connect(lambda _message: self._cleanup_thread())
        self.worker.canceled.connect(self._cleanup_thread)
        self.thread.start()
        self._set_running(True)

    def cancel_processing(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def append_log(self, line: str) -> None:
        self.log_text.appendPlainText(line)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_finished(self, result: dict) -> None:
        self.result_paths = result
        self._set_result_buttons_enabled(True)
        self.statusBar().showMessage("处理完成")
        QMessageBox.information(self, "处理完成", "中文配音视频和相关文件已生成")

    def on_failed(self, message: str) -> None:
        self.statusBar().showMessage("处理失败")
        QMessageBox.critical(self, "处理失败", message)

    def on_canceled(self) -> None:
        self.append_log("任务已取消")
        self.statusBar().showMessage("任务已取消")
        QMessageBox.information(self, "任务已取消", "任务已取消，已生成的中间文件会保留在输出目录")

    def _cleanup_thread(self) -> None:
        self._set_running(False)
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
        self.thread = None
        self.worker = None

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.choose_video_button.setEnabled(not running)
        self.choose_whisper_model_file_button.setEnabled(not running)
        self.choose_whisper_model_path_button.setEnabled(not running)
        self.choose_output_button.setEnabled(not running)

    def _set_result_buttons_enabled(self, enabled: bool) -> None:
        for button in (
            self.open_output_button,
            self.play_video_button,
            self.view_en_srt_button,
            self.view_zh_srt_button,
            self.view_bilingual_button,
        ):
            button.setEnabled(enabled)

    def copy_log(self) -> None:
        QApplication.clipboard().setText(self.log_text.toPlainText())
        self.statusBar().showMessage("日志已复制")

    def save_log(self) -> None:
        default_path = str(Path(self.output_dir_edit.text() or DEFAULT_OUTPUT_DIR) / "manual_log.txt")
        path, _ = QFileDialog.getSaveFileName(self, "保存日志", default_path, "文本文件 (*.txt *.log)")
        if path:
            Path(path).write_text(self.log_text.toPlainText(), encoding="utf-8")
            self.statusBar().showMessage(f"日志已保存：{path}")

    def open_path(self, path: str | None) -> None:
        if not path:
            QMessageBox.information(self, "暂无结果", "请先完成一次处理")
            return
        target = Path(path)
        if not target.exists():
            QMessageBox.warning(self, "文件不存在", f"找不到文件：{target}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
