from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable


class TaskLogger:
    def __init__(self, sink: Callable[[str], None] | None = None) -> None:
        self._sink = sink
        self._lines: list[str] = []
        self._lock = Lock()

    @property
    def text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def _write(self, level: str, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}"
        with self._lock:
            self._lines.append(line)
        if self._sink:
            self._sink(line)

    def save(self, path: Path) -> Path:
        path.write_text(self.text + "\n", encoding="utf-8")
        return path

