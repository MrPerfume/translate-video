from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "video_dubber"
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(PROJECT_DIR))

from app.main import main  # noqa: E402


def self_check() -> int:
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow
    from app.utils.ffmpeg_utils import check_ffmpeg_and_ffprobe

    ffmpeg, ffprobe = check_ffmpeg_and_ffprobe()
    app = QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "英文视频转中文配音工具"
    app.quit()
    print("self-check ok")
    print(f"ffmpeg: {ffmpeg}")
    print(f"ffprobe: {ffprobe}")
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(self_check())
    raise SystemExit(main())
