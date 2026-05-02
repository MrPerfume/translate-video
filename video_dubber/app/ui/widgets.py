from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QLabel

from app.utils.file_utils import SUPPORTED_VIDEO_EXTENSIONS


class DropVideoLabel(QLabel):
    video_dropped = Signal(str)  # 每个有效视频文件触发一次

    def __init__(self, text: str = "拖拽视频文件或文件夹到此处批量添加") -> None:
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(54)
        self.setStyleSheet(
            """
            QLabel {
                border: 1px dashed #8a8f98;
                border-radius: 6px;
                color: #4a4f57;
                background: #f8fafc;
                padding: 10px;
            }
            """
        )

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.is_dir() or p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        accepted = False
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                for child in sorted(p.iterdir()):
                    if child.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                        self.video_dropped.emit(str(child))
                        accepted = True
            elif p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                self.video_dropped.emit(str(p))
                accepted = True
        if accepted:
            event.acceptProposedAction()
        else:
            event.ignore()

