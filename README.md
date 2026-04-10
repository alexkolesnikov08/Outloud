# OutLoud

A simple CLI tool to turn audio recordings into text and summaries. It works both locally and in the cloud.

## 1. Fast & Optimized
Designed to run on standard hardware without breaking a sweat. 
* Uses **less than 1GB of RAM**.
* Processes 45 minutes of audio in about 66 seconds (Cloud mode).
* Smart chunking: automatically splits long files so it won't crash your system.

## 2. Extremely Simple
No complex menus or heavy GUIs. Just a few commands in your terminal:
```bash
pip install -e .
outloud setup          # download models
outloud record --cloud # start recording and get a summary
```

## 3. Work in Progress
This is my very first project on GitHub. It's basically a "built on my knee" prototype that actually works. I am 17 and just starting my journey, so the code might be messy. I plan to improve and refine it as I learn more, but for now, it gets the job done!

---

## Quick Start

| Command | What it does |
|---------|--------------|
| `outloud record` | Record voice -> text + summary |
| `outloud file audio.m4a` | Process an existing file |
| `outloud yt "link"` | Summarize a YouTube video |

## Models used
* **Local:** Vosk (speech) + Qwen 0.8B (summary)
* **Cloud:** Whisper Turbo (speech) + GPT-OSS (summary) via Groq API
