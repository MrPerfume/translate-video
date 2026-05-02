from __future__ import annotations

from pathlib import Path

from app.core.transcriber import SubtitleSegment
from app.utils.time_utils import seconds_to_srt_time


class SubtitleWriter:
    def write_srt(self, segments: list[SubtitleSegment], path: Path, language: str) -> Path:
        lines: list[str] = []
        for index, segment in enumerate(segments, start=1):
            text = segment.original_text if language == "en" else segment.chinese_text
            lines.extend(
                [
                    str(index),
                    f"{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}",
                    text.strip(),
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def write_bilingual_txt(self, segments: list[SubtitleSegment], path: Path) -> Path:
        lines: list[str] = []
        for index, segment in enumerate(segments, start=1):
            lines.append(f"{index}. {seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}")
            lines.append(f"EN: {segment.original_text.strip()}")
            lines.append(f"ZH: {segment.chinese_text.strip()}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

