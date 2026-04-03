<p align="center">
  <img src="assets/logo.png" alt="Font Search logo" width="120"/>
</p>

<h1 align="center">Font Search</h1>

<p align="center">
  <img alt="Beta" src="https://img.shields.io/badge/status-beta-orange"/>
  <img alt="Windows" src="https://img.shields.io/badge/platform-Windows%2010%2F11-blue"/>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green"/>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10--3.13-blue"/>
</p>

Find which system font best matches text in a screenshot or image. Crop the text region, and Font Search ranks your installed fonts by visual similarity.

> **Beta** — Tested on Windows 10/11. Currently supports **Khmer** and **English** font matching. Other scripts may work but are untested.

---

## Demo

<p align="center" width="100%">
<video src="https://github.com/user-attachments/assets/b2f58892-d1a1-471b-8383-00dcd2dd8586" width="80%" controls title="Font Search demo"></video>
</p>
---

## Features

- Drag-and-drop or open an image, then crop the text area
- Scores all installed system fonts using SSIM, HOG descriptors, stroke width, and projection profiles
- Correct rendering for complex scripts — Khmer shaping via Qt HarfBuzz (subscripts and stacked glyphs render correctly)
- Reads fonts from both system and user font directories (including `%LOCALAPPDATA%\Microsoft\Windows\Fonts`)
- Cancel a running search at any time

## Requirements

- Windows 10 or 11 (64-bit)
- Python 3.10–3.13 with [uv](https://github.com/astral-sh/uv) — for running from source

## Download

Download the latest pre-built release from the [Releases](../../releases) page and run `FontSearch.exe` — no installation required.

## Run from source

```bash
git clone <repo-url>
cd font-search
uv sync
uv run python main.py
```

## Build a standalone .exe

Requires the project dependencies to be installed (run `uv sync` first).

```bash
uv run python build.py
```

The output is `dist/FontSearch/FontSearch.exe`. Distribute the entire `dist/FontSearch/` folder.

## Usage

1. Launch the app and open (or drag-and-drop) an image containing text
2. Draw a crop rectangle around the text you want to identify
3. Type the text shown in the image into the **Text in image** field
4. Click **Find Matching Fonts** — results are ranked by visual similarity score

## Language support

| Language | Status  |
|----------|---------|
| Khmer    | ✅ Tested |
| English  | ✅ Tested |
| Others   | Untested (may work) |

## Project structure

```
font-search/
├── main.py              # PySide6 UI
├── app/
│   └── font_engine.py   # font discovery, rendering, and scoring
├── assets/
│   └── logo.png         # app logo
├── build.py             # PyInstaller build script
├── pyproject.toml
└── LICENSE
```

## Contributing

Contributions are welcome! Please open an issue or pull request. When adding support for a new script, include a screenshot showing correct rendering.

## License

[MIT](LICENSE) — free to use, modify, and distribute.
