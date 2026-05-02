from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QLabel

from app.utils.file_utils import SUPPORTED_VIDEO_EXTENSIONS


class DropVideoLabel(QLabel):
    video_dropped = Signal(str)

    def __init__(self, text: str = "拖拽视频文件到这里") -> None:
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
                if Path(url.toLocalFile()).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                self.video_dropped.emit(str(path))
                event.acceptProposedAction()
                return
        event.ignore()

