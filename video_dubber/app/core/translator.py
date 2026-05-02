from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

import httpx

from app.config.settings import load_settings
from app.core.transcriber import SubtitleSegment


TRANSLATION_SYSTEM_PROMPT = """你是专业影视字幕翻译。请把英文字幕翻译成自然、口语化、适合中文观众观看的简体中文。要求：
1. 保留原意
2. 不要逐字硬翻
3. 尽量简洁
4. 适合配音朗读
5. 不要解释
6. 不要添加编号
7. 不要输出多余内容

请严格输出 JSON object，格式示例：
{"translations":[{"translated_text":"你好"}]}"""


@dataclass(frozen=True)
class TranslatorConfig:
    service: str
    model: str
    batch_size: int = 20


class SubtitleTranslator:
    def __init__(self, config: TranslatorConfig) -> None:
        self.config = config
        self.settings = load_settings()

    def validate_credentials(self) -> None:
        service = self.config.service.lower()
        if service == "openai" and not self.settings.openai_api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 中配置")
        if service == "deepseek" and not self.settings.deepseek_api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置")
        if service not in {"openai", "deepseek"}:
            raise RuntimeError(f"不支持的翻译服务：{self.config.service}")

    def translate_segments(
        self,
        segments: list[SubtitleSegment],
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[SubtitleSegment]:
        self.validate_credentials()
        total_batches = max(1, (len(segments) + self.config.batch_size - 1) // self.config.batch_size)
        completed_batches = 0
        for start in range(0, len(segments), self.config.batch_size):
            if should_cancel and should_cancel():
                raise RuntimeError("任务已取消")
            batch = segments[start : start + self.config.batch_size]
            translations = self._translate_batch(batch, start)
            if len(translations) != len(batch):
                raise RuntimeError(
                    f"翻译返回数量不匹配：输入 {len(batch)} 条，返回 {len(translations)} 条。请稍后重试或减小批量。"
                )
            for segment, translated in zip(batch, translations):
                segment.chinese_text = str(translated).strip()
                if not segment.chinese_text:
                    raise RuntimeError("翻译结果包含空字幕")
            completed_batches += 1
            if on_progress:
                on_progress(completed_batches, total_batches)
        return segments

    def _translate_batch(self, batch: list[SubtitleSegment], offset: int) -> list[str]:
        payload_items = [
            {
                "index": offset + idx,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.original_text,
            }
            for idx, segment in enumerate(batch)
        ]
        user_prompt = (
            "请翻译下面 JSON 数组里的 text 字段，并只返回 JSON object。"
            "返回格式必须是 {\"translations\":[{\"translated_text\":\"...\"}]}。"
            "translations 数组必须与输入数组长度和顺序完全一致。\n\n"
            f"{json.dumps(payload_items, ensure_ascii=False)}"
        )

        content = self._chat_completion(user_prompt)
        parsed = self._parse_json_array(content)
        translations: list[str] = []
        for item in parsed:
            if isinstance(item, dict):
                value = item.get("translated_text") or item.get("text") or item.get("translation")
            else:
                value = item
            translations.append(str(value).strip())
        return translations

    def _chat_completion(self, user_prompt: str) -> str:
        service = self.config.service.lower()
        if service == "openai":
            base_url = self.settings.openai_base_url
            api_key = self.settings.openai_api_key
        elif service == "deepseek":
            base_url = self.settings.deepseek_base_url
            api_key = self.settings.deepseek_api_key
        else:
            raise RuntimeError(f"不支持的翻译服务：{self.config.service}")

        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": self._estimate_max_tokens(user_prompt),
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, headers=headers, json=body)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:500] if response.text else str(exc)
                raise RuntimeError(f"翻译 API 请求失败：HTTP {response.status_code}，{detail}") from exc
            data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("翻译 API 返回格式异常") from exc

    @staticmethod
    def _parse_json_array(content: str) -> list:
        text = content.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if not array_match:
                raise RuntimeError("翻译结果不是有效 JSON 数组")
            parsed = json.loads(array_match.group(0))
        if isinstance(parsed, dict):
            for key in ("translations", "items", "result", "data"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            raise RuntimeError("翻译结果不是 JSON 数组")
        return parsed

    @staticmethod
    def _estimate_max_tokens(prompt: str) -> int:
        return min(16_000, max(1_024, len(prompt) * 2))
