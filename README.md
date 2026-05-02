# 英文视频转中文配音工具

成品入口：

```bash
./start_macos.command
```

或：

```bash
cd video_dubber
.venv/bin/python run.py
```

自检：

```bash
python run.py --self-check
```

首次使用前，打开 `video_dubber/.env`，填写你选择的翻译服务 API Key；字段示例见 [video_dubber/.env.example](video_dubber/.env.example)。

完整安装、使用和打包说明见 [video_dubber/README.md](video_dubber/README.md)。
