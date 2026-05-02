# 英文视频转中文配音工具

一个 Python 3.11+ 桌面 GUI MVP：导入英文视频后，自动提取音频、识别英文字幕、翻译为自然简体中文、生成中文配音，并合成为中文配音视频。

## 功能

- 选择或拖拽 `mp4`、`mov`、`mkv`、`avi` 视频
- FFmpeg 提取 16kHz mono WAV
- faster-whisper 英文语音识别
- DeepSeek 或 OpenAI Chat Completions 翻译字幕
- Edge TTS 生成中文配音 WAV
- FFmpeg 合成中文配音 MP4
- 输出英文 SRT、中文 SRT、双语 TXT、处理日志
- GUI 使用 QThread 执行耗时任务，处理时窗口不会冻结

## 创建虚拟环境

```bash
cd video_dubber
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
cd video_dubber
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 安装 Python 依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

首次使用 faster-whisper 会下载模型文件，请保持网络可用。模型越大，识别更慢、资源占用更高。

## 安装 FFmpeg

macOS 推荐 Homebrew:

```bash
brew install ffmpeg
```

Windows 推荐使用 winget:

```powershell
winget install Gyan.FFmpeg
```

也可以从 FFmpeg 官网下载静态构建，并把 `ffmpeg`、`ffprobe` 所在目录加入 `PATH`。

验证：

```bash
ffmpeg -version
ffprobe -version
```

## 配置 .env

复制示例文件：

```bash
cp .env.example .env
```

按你选择的翻译服务填写 API Key：

```dotenv
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
ELEVENLABS_API_KEY=
OPENAI_BASE_URL=
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

默认翻译服务是 DeepSeek，至少需要填写 `DEEPSEEK_API_KEY`。如果切换到 OpenAI，则需要填写 `OPENAI_API_KEY`。程序不会在日志中打印 API Key。

## 启动软件

从仓库根目录运行：

```bash
python run.py
```

或进入项目目录运行：

```bash
cd video_dubber
python run.py
```

环境自检：

```bash
python run.py --self-check
```

## GUI 使用方式

1. 点击“选择视频”，或把视频文件拖到窗口中。
2. 选择 Whisper 模型，默认 `small`。
3. 如果已有本地模型，可在“本地模型路径”选择模型。faster-whisper 模型请选择文件夹；OpenAI Whisper 的 `.pt` 模型请选择模型文件，例如 `models/large-v3.pt`。
4. 选择翻译服务，默认 `DeepSeek`，模型默认 `deepseek-v4-flash`。
5. TTS 默认 `Edge TTS`，中文声音默认 `zh-CN-XiaoxiaoNeural`。
6. 选择输出目录，默认 `video_dubber/output`。
7. 点击“开始处理”，等待日志和进度条更新。
8. 处理完成后，可打开输出目录、播放生成视频、查看字幕和双语文本。

## 输出文件

以输入视频文件名为前缀输出：

- `{stem}_zh_dubbed.mp4`：中文配音视频
- `{stem}_en.srt`：英文字幕
- `{stem}_zh.srt`：中文字幕
- `{stem}_bilingual.txt`：双语对照文本
- `{stem}_bilingual.srt`：勾选“生成双语字幕”时输出
- `{stem}_zh_voice.wav`：中文配音 WAV
- `{stem}_process.log`：处理日志

## 打包 macOS App

先确认在虚拟环境中可正常运行：

```bash
python run.py
```

再打包：

```bash
pyinstaller --name "英文视频转中文配音工具" --windowed --onefile run.py
```

生成结果在 `dist/`。打包后的应用仍需要系统可访问 `ffmpeg` 和 `ffprobe`，也需要 `.env` 或运行环境中配置 API Key。

## 打包 Windows exe

在 Windows 虚拟环境中执行：

```powershell
pyinstaller --name "英文视频转中文配音工具" --windowed --onefile run.py
```

生成结果在 `dist\`。请确保 FFmpeg 已加入 `PATH`，并在 exe 同目录或工作目录提供 `.env`。

## 当前 MVP 限制

- Edge TTS 是默认且完整实现的 TTS 路径。
- OpenAI TTS / ElevenLabs 目前只预留接口，选择后会给出明确错误。
- “保留背景音”是简单低音量混音，不做人声分离，可能保留英文人声。
- 不包含口型同步、字幕烧录、批量处理。
- TTS 超出原字幕时长时会保留完整语音并写入日志，暂不做自动变速。
