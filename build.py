"""Build FontSearch into a standalone Windows executable.

Usage:
    uv run python build.py

Output: dist/FontSearch/FontSearch.exe
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PNG = ASSETS / "logo.png"
LOGO_ICO = ASSETS / "logo.ico"


def _make_icon() -> None:
    """Convert assets/logo.png to a multi-resolution .ico file."""
    from PIL import Image  # already a project dependency

    img = Image.open(LOGO_PNG).convert("RGBA")
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    resized = [img.resize(s, Image.LANCZOS) for s in sizes]
    resized[0].save(LOGO_ICO, format="ICO", sizes=sizes, append_images=resized[1:])
    print(f"    icon → {LOGO_ICO}")


PYINSTALLER_ARGS = [
    "--name", "FontSearch",
    "--windowed",       # no console window
    "--onedir",         # folder bundle — faster startup than --onefile
    "--noconfirm",      # overwrite dist/ without asking
    "--clean",          # clean PyInstaller cache before build
    # App icon (converted from assets/logo.png by _make_icon())
    "--icon", str(LOGO_ICO),
    # Bundle the assets folder so the logo is available at runtime
    "--add-data", f"{ASSETS};assets",
    # Hidden imports that static analysis may miss
    "--hidden-import", "cv2",
    "--hidden-import", "skimage.feature",
    "--hidden-import", "skimage.filters",
    "--hidden-import", "skimage.metrics",
    "--hidden-import", "skimage.transform",
    "--hidden-import", "fonttools",
    "--hidden-import", "fonttools.ttLib",
    str(ROOT / "main.py"),
]


def main() -> None:
    print("==> Converting icon ...")
    _make_icon()

    print("\n==> Building FontSearch ...")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", *PYINSTALLER_ARGS],
        check=True,
    )

    exe = ROOT / "dist" / "FontSearch" / "FontSearch.exe"
    if exe.exists():
        print(f"\n[OK] Build complete: {exe}")
        print("     Distribute the entire dist/FontSearch/ folder.")
    else:
        print("\n[!] Build may have failed — check the output above.")


if __name__ == "__main__":
    main()
