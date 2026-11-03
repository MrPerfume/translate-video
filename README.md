# 英文视频转中文配音工具

导入英文视频，依次完成语音识别、翻译、中文配音和视频合成，输出配音视频、字幕与处理日志。适合把英文视频整理成可听、可对照阅读的中文材料。

## 使用示例

选择一个示例视频 `lesson.mp4`，设置翻译服务与输出目录，点击“开始处理”。完成后可查看：

| 输出 | 用途 |
| --- | --- |
| `lesson_zh_dubbed.mp4` | 中文配音视频 |
| `lesson_en.srt`、`lesson_zh.srt` | 英文、中文字幕 |
| `lesson_bilingual.txt` | 双语对照文本 |
| `lesson_bilingual.srt` | 勾选“生成双语字幕”时输出的双语字幕 |
| `lesson_process.log` | 处理过程与错误记录 |

## 首次使用

1. 准备 Python 3.11+ 和 FFmpeg，按[完整安装说明](video_dubber/README.md#创建虚拟环境)创建虚拟环境、安装依赖。
2. 将 `video_dubber/.env.example` 复制为 `video_dubber/.env`，填写所选翻译服务的 API Key。默认使用 DeepSeek；切换到 OpenAI 时需配置对应 Key。
3. 从仓库根目录启动：

```bash
./start_macos.command
```

上面的入口用于 macOS。也可以从仓库根目录运行：

```bash
python run.py
```

4. 选择或拖入视频，确认模型、翻译服务和输出目录，再开始处理。首次使用 faster-whisper 会下载模型，耗时和资源占用随模型大小变化。

## 功能与限制

- 支持导入 `mp4`、`mov`、`mkv`、`avi`；串联 FFmpeg、faster-whisper、DeepSeek / OpenAI 翻译与 Edge TTS。
- 提供进度与日志、声音试听、语速调整和输出目录入口。
- Edge TTS 为已实现的配音路径；OpenAI TTS / ElevenLabs 仅预留接口。
- “保留背景音”采用简单混音，可能保留英文人声；不包含人声分离、口型同步、字幕烧录和批量处理。
- TTS 语速调整上限为 1.25x，超出时保留完整语音并记录日志；识别、翻译和配音效果需自行检查。

## 开发说明

环境与窗口自检：

```bash
python run.py --self-check
```

完整的 macOS / Windows 安装、翻译服务配置、操作步骤与打包说明见 [video_dubber/README.md](video_dubber/README.md)。API Key 留在本机 `.env` 或环境变量中，视频、字幕、日志和模型缓存不纳入 Git。
