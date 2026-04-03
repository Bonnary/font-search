"""Font discovery, rendering, and similarity comparison engine (Windows)."""

from __future__ import annotations

import os
import threading
import winreg
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QPainter,
)
from skimage.feature import hog
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim

# ── constants ──────────────────────────────────────────────────────────────────

_FONTS_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_USER_FONTS_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"

# Stretch canvas: both images are resized to exactly this size (ignoring aspect
# ratio) before comparison.  This removes width-mismatch misalignment that
# plagued the old "center on a fixed canvas" approach and lets SSIM/pixel/HOG
# work on pixel-aligned glyphs regardless of render scale.
_STRETCH_W = 320
_STRETCH_H = 64

# Square canvas is kept for HOG / Hu / contour on the glyph block.
# We target a taller content region so multi-character words don't get
# squashed to a useless 12 px strip.
_NORM_SIZE = 96
_NORM_MARGIN = 4  # smaller margin → more canvas used for content

_STRETCH_SSIM_WEIGHT = 0.22  # SSIM on stretch-normalised image pair
_STRETCH_PIXEL_WEIGHT = 0.18  # MSE+IoU on stretch-normalised pair
_STRETCH_HOG_WEIGHT = 0.20  # HOG cosine on stretch canvas (aspect-invariant)
_HOG_WEIGHT = 0.16  # HOG cosine on square canvas
_STROKE_WEIGHT = 0.06  # stroke-width ratio
_DENSITY_WEIGHT = 0.06  # ink density ratio
_ASPECT_WEIGHT = 0.02  # aspect-ratio log-ratio
_CONTOUR_WEIGHT = 0.06  # contour IoU on square canvas
_HU_WEIGHT = 0.02  # Hu moments
_PROJECTION_WEIGHT = 0.02  # row/col projections on square canvas


@dataclass(frozen=True)
class _GlyphFeatures:
    mask: np.ndarray  # 96×96 square, aspect-preserving (for HOG/Hu/contour)
    stretch_mask: (
        np.ndarray
    )  # STRETCH_H×STRETCH_W, ignores aspect ratio (for SSIM/pixel)
    contour_mask: np.ndarray
    projections: np.ndarray
    hog_vector: np.ndarray  # HOG on square canvas
    stretch_hog_vector: np.ndarray  # HOG on stretch canvas (aspect-invariant)
    hu_moments: np.ndarray
    aspect_ratio: float
    ink_density: float
    stroke_width: float


# Invisible/format Unicode characters that should not disqualify a font from
# the coverage check.  These characters carry no visible glyph, so a font
# that covers all *visible* glyphs is a valid candidate even if it lacks them.
# Khmer text commonly uses U+200B (ZERO WIDTH SPACE) as a word separator, and
# most built-in fonts (e.g. Windows "Khmer") do not include that codepoint.
_INVISIBLE_CODEPOINTS: frozenset[int] = frozenset({
    0x00AD,  # SOFT HYPHEN
    0x200B,  # ZERO WIDTH SPACE (common Khmer word separator)
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x2060,  # WORD JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
})


def _text_codepoints(text: str) -> set[int]:
    """Return the visible Unicode code points required to render *text*.

    Invisible format characters (zero-width spaces, directional marks, etc.)
    are excluded so that fonts which cover all *visible* glyphs are not
    incorrectly filtered out by the coverage check.
    """
    return {
        ord(ch)
        for ch in text
        if not ch.isspace() and ord(ch) not in _INVISIBLE_CODEPOINTS
    }


# ── font discovery ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=2048)
def _font_display_name(path: str) -> str:
    """
    Read the human-friendly full font name from the file using fonttools.
    Falls back to the filename stem on any error.
    """
    try:
        from fontTools.ttLib import TTFont  # deferred to keep import overhead low

        tt = TTFont(path, lazy=True)
        nt = tt["name"]
        for pid, eid, lid in [(3, 1, 0x0409), (1, 0, 0)]:
            rec = nt.getName(nameID=4, platformID=pid, platEncID=eid, langID=lid)
            if rec:
                return rec.toUnicode()
            rec = nt.getName(nameID=1, platformID=pid, platEncID=eid, langID=lid)
            if rec:
                return rec.toUnicode()
    except Exception:
        pass
    return Path(path).stem


@lru_cache(maxsize=4096)
def _font_codepoints(path: str) -> frozenset[int]:
    """Return the set of Unicode code points advertised by *path*."""
    try:
        from fontTools.ttLib import TTFont  # deferred to keep import overhead low

        tt = TTFont(path, lazy=True)
        codepoints: set[int] = set()
        for table in tt["cmap"].tables:
            codepoints.update(table.cmap)
        return frozenset(codepoints)
    except Exception:
        return frozenset()


@lru_cache(maxsize=4096)
def _font_family_style(path: str) -> tuple[str, str]:
    """Return the preferred family and style names advertised by *path*."""
    fallback = Path(path).stem, ""
    try:
        from fontTools.ttLib import TTFont  # deferred to keep import overhead low

        tt = TTFont(path, lazy=True)
        nt = tt["name"]

        def _name(*name_ids: int) -> str | None:
            for name_id in name_ids:
                for pid, eid, lid in [(3, 1, 0x0409), (1, 0, 0)]:
                    rec = nt.getName(
                        nameID=name_id,
                        platformID=pid,
                        platEncID=eid,
                        langID=lid,
                    )
                    if rec:
                        return rec.toUnicode()
            return None

        family = _name(16, 1) or fallback[0]
        style = _name(17, 2) or ""
        return family, style
    except Exception:
        return fallback


@lru_cache(maxsize=4096)
def _qt_font_family(path: str) -> str | None:
    """Load *path* into Qt and return a family name that can be instantiated."""
    if QGuiApplication.instance() is None:
        return None
    font_id = QFontDatabase.addApplicationFont(path)
    if font_id == -1:
        return None
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        return None
    preferred_family, _ = _font_family_style(path)
    for family in families:
        if family.casefold() == preferred_family.casefold():
            return family
    return families[0]


def get_system_fonts() -> dict[str, str]:
    """
    Return ``{font_display_name: font_file_path}`` for installed fonts on Windows.

    Uses the Windows registry for discovery and fonttools for name resolution.
    Duplicate paths are skipped. When multiple files resolve to the same display
    name, the filename stem is appended to keep every face searchable.
    """
    candidates: list[str] = []
    seen_paths: set[str] = set()

    def _add_font_path(path: Path) -> None:
        norm = str(path)
        if (
            path.suffix.lower() in {".ttf", ".otf"}
            and path.exists()
            and norm not in seen_paths
        ):
            seen_paths.add(norm)
            candidates.append(norm)

    def _read_registry_fonts(root: int, subkey: str, base_dir: Path) -> None:
        try:
            with winreg.OpenKey(root, subkey) as key:
                idx = 0
                while True:
                    try:
                        _, value, _ = winreg.EnumValue(key, idx)
                        idx += 1
                    except OSError:
                        break
                    path = Path(value) if os.path.isabs(value) else base_dir / value
                    _add_font_path(path)
        except OSError:
            return

    _read_registry_fonts(
        winreg.HKEY_LOCAL_MACHINE,
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        _FONTS_DIR,
    )
    _read_registry_fonts(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows NT\CurrentVersion\Fonts",
        _USER_FONTS_DIR,
    )

    for font_dir in (_FONTS_DIR, _USER_FONTS_DIR):
        if not font_dir.exists():
            continue
        for ext in ("*.ttf", "*.otf"):
            for path in font_dir.glob(ext):
                _add_font_path(path)

    fonts: dict[str, str] = {}
    for path in candidates:
        display_name = _font_display_name(path)
        key = display_name
        if key in fonts:
            key = f"{display_name} [{Path(path).stem}]"
        while key in fonts:
            key = f"{key}*"
        fonts[key] = path
    return fonts


def font_supports_text(font_path: str, text: str) -> bool:
    """Return True when *font_path* covers every non-whitespace character in *text*."""
    needed = _text_codepoints(text)
    if not needed:
        return True
    return needed.issubset(_font_codepoints(font_path))


# ── text rendering ─────────────────────────────────────────────────────────────


def _render_with_qt(text: str, font_path: str) -> Image.Image | None:
    """Render *text* with Qt so complex scripts use the platform shaper."""
    family = _qt_font_family(font_path)
    if family is None:
        return None

    _, style = _font_family_style(font_path)

    def _make_font(pixel_size: int) -> QFont:
        font = QFont(family)
        if style:
            font.setStyleName(style)
        font.setPixelSize(pixel_size)
        return font

    try:
        lo, hi, best = 8, 600, 8
        for _ in range(12):
            mid = (lo + hi) // 2
            fm_mid = QFontMetricsF(_make_font(mid))
            bounds = fm_mid.tightBoundingRect(text)
            # Use the larger of tightBoundingRect height and line height as the
            # effective height, because tightBoundingRect underestimates for
            # complex scripts (Khmer, Thai) that have above/below-baseline marks.
            effective_h = max(bounds.height(), fm_mid.ascent() + fm_mid.descent())
            if effective_h <= 0:
                lo = mid + 1
                continue
            if effective_h < (_NORM_SIZE - 2 * _NORM_MARGIN):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        font = _make_font(best)
        fm = QFontMetricsF(font)
        bounds = fm.tightBoundingRect(text)
        w = max(1, int(np.ceil(bounds.width())))
        tight_h = max(1, int(np.ceil(bounds.height())))
        # tightBoundingRect is documented to underestimate for complex scripts
        # (Khmer, Thai, Devanagari, etc.) that have above/below-baseline marks.
        # Use the font's ascent+descent as a safe floor for the canvas height.
        line_h = max(1, int(np.ceil(fm.ascent() + fm.descent())))
        h = max(tight_h, line_h)
        # Generous margin so glyphs that extend beyond the reported bounds
        # (common for Khmer subscripts and vowel signs) are never clipped.
        # _crop_foreground in _normalise will remove the extra whitespace.
        margin_x = max(8, w // 6)
        margin_y = max(8, h // 4)

        image = QImage(w + 2 * margin_x, h + 2 * margin_y, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)

        painter = QPainter(image)
        painter.setPen(Qt.GlobalColor.black)
        painter.setFont(font)
        painter.drawText(QPointF(margin_x - bounds.left(), margin_y - bounds.top()), text)
        painter.end()

        from PIL.ImageQt import fromqimage

        return fromqimage(image).convert("L")
    except Exception:
        return None


def _render_with_pillow(text: str, font_path: str) -> Image.Image | None:
    """Fallback renderer when Qt font layout is unavailable."""
    try:
        dummy_img = Image.new("L", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        lo, hi, best = 8, 600, 8
        for _ in range(12):
            mid = (lo + hi) // 2
            font = ImageFont.truetype(font_path, mid)
            bb = dummy_draw.textbbox((0, 0), text, font=font)
            h = bb[3] - bb[1]
            if h <= 0:
                lo = mid + 1
                continue
            if h < (_NORM_SIZE - 2 * _NORM_MARGIN):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        font = ImageFont.truetype(font_path, best)
        bb = dummy_draw.textbbox((0, 0), text, font=font)
        w = max(1, bb[2] - bb[0])
        h = max(1, bb[3] - bb[1])
        img = Image.new("L", (int(w) + 4, int(h) + 4), 255)
        ImageDraw.Draw(img).text((2 - bb[0], 2 - bb[1]), text, font=font, fill=0)
        return img
    except Exception:
        return None


def _render(text: str, font_path: str) -> Image.Image | None:
    """
    Render *text* with *font_path* at a size that makes the glyph block
    approximately ``_NORM_H`` pixels tall.

    Returns a white-background greyscale PIL image, or None on failure.
    """
    rendered = _render_with_qt(text, font_path)
    if rendered is not None:
        return rendered
    return _render_with_pillow(text, font_path)


# ── normalisation & similarity ─────────────────────────────────────────────────


def _threshold_mask(img: Image.Image) -> np.ndarray:
    """Return a binary mask where glyph pixels are 255 and background is 0."""
    gray = np.array(img.convert("L"), dtype=np.uint8)
    try:
        thresh = threshold_otsu(gray)
    except Exception:
        thresh = 127
    mask = (gray < thresh).astype(np.uint8) * 255
    # If more than half the pixels are classified as ink the image has a dark
    # background (e.g. a dark-mode screenshot with light-coloured text).  Invert
    # so that foreground (text) is always 255 and background is always 0.
    if np.mean(mask > 0) > 0.5:
        mask = 255 - mask
    return mask


def _crop_foreground(mask: np.ndarray) -> np.ndarray:
    """Return the tight foreground crop for *mask*."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((1, 1), dtype=np.uint8)

    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = int(r_idx[0]), int(r_idx[-1])
    c0, c1 = int(c_idx[0]), int(c_idx[-1])
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _fit_mask_to_canvas(mask: np.ndarray) -> np.ndarray:
    """Crop the foreground and center it on a fixed square canvas."""
    cropped = _crop_foreground(mask)
    if not np.any(cropped):
        return np.zeros((_NORM_SIZE, _NORM_SIZE), dtype=np.uint8)

    h, w = cropped.shape
    target = _NORM_SIZE - 2 * _NORM_MARGIN
    scale = min(target / max(h, 1), target / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((_NORM_SIZE, _NORM_SIZE), dtype=np.uint8)
    top = (_NORM_SIZE - new_h) // 2
    left = (_NORM_SIZE - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _fit_mask_to_stretch_canvas(mask: np.ndarray) -> np.ndarray:
    """
    Crop the foreground and resize it to (_STRETCH_H, _STRETCH_W) IGNORING
    aspect ratio.

    Stretching both the input image and each rendered candidate to the same
    fixed rectangle removes horizontal-misalignment artifacts that arise when
    fonts render the same text at slightly different widths.  The comparison
    becomes purely about shape/stroke style, not about scale.
    """
    cropped = _crop_foreground(mask)
    if not np.any(cropped):
        return np.zeros((_STRETCH_H, _STRETCH_W), dtype=np.uint8)
    resized = cv2.resize(
        cropped, (_STRETCH_W, _STRETCH_H), interpolation=cv2.INTER_AREA
    )
    # Re-binarize after the resize blurs sub-pixel stroke edges
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def _contour_mask(mask: np.ndarray) -> np.ndarray:
    """Draw extracted contours to emphasize outlines for shape matching."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros_like(mask)
    if contours:
        cv2.drawContours(outline, contours, -1, color=255, thickness=1)
    return outline


def _projection_features(mask: np.ndarray) -> np.ndarray:
    """Return normalized horizontal and vertical ink distributions."""
    ink = mask.astype(np.float32) / 255.0
    row_proj = ink.sum(axis=1)
    col_proj = ink.sum(axis=0)
    row_total = float(row_proj.sum())
    col_total = float(col_proj.sum())
    if row_total > 0:
        row_proj /= row_total
    if col_total > 0:
        col_proj /= col_total
    return np.concatenate((row_proj, col_proj)).astype(np.float32)


def _hu_moments(mask: np.ndarray) -> np.ndarray:
    """Return log-scaled Hu moments for the full glyph mask."""
    moments = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    return np.sign(hu) * np.log1p(np.abs(hu))


def _aspect_ratio(mask: np.ndarray) -> float:
    """Return the width/height ratio for the foreground crop."""
    h, w = _crop_foreground(mask).shape
    return float(w / max(h, 1))


def _ink_density(mask: np.ndarray) -> float:
    """Return the fraction of foreground pixels inside the tight crop."""
    cropped = _crop_foreground(mask)
    return float(np.mean(cropped > 0))


def _stroke_width(mask: np.ndarray) -> float:
    """
    Estimate the average stroke width using the distance transform.

    For each foreground pixel, the distance transform gives the distance to
    the nearest background pixel. The median of these values is a robust
    estimate of half the stroke width. Multiply by 2 and normalize by the
    image diagonal so the value is scale-independent.
    """
    cropped = _crop_foreground(mask)
    if not np.any(cropped):
        return 0.0
    fg = (cropped > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    fg_dist = dist[fg > 0].astype(np.float32)
    if len(fg_dist) == 0:
        return 0.0
    h, w = cropped.shape
    diag = float(np.hypot(h, w))
    # Median of the distance values (half stroke width), normalized
    return float(np.median(fg_dist) * 2.0 / max(diag, 1.0))


def _hog_vector(mask: np.ndarray) -> np.ndarray:
    """Return a compact gradient descriptor for the normalized glyph."""
    values = hog(
        mask.astype(np.float32) / 255.0,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return values.astype(np.float32)


def _normalise(img: Image.Image) -> _GlyphFeatures:
    """Extract centered raster and shape descriptors for a glyph image."""
    raw_mask = _threshold_mask(img)
    mask = _fit_mask_to_canvas(raw_mask)
    stretch = _fit_mask_to_stretch_canvas(raw_mask)
    contour = _contour_mask(mask)
    return _GlyphFeatures(
        mask=mask,
        stretch_mask=stretch,
        contour_mask=contour,
        projections=_projection_features(mask),
        hog_vector=_hog_vector(mask),
        stretch_hog_vector=_hog_vector(stretch),  # HOG on aspect-invariant canvas
        hu_moments=_hu_moments(mask),
        aspect_ratio=_aspect_ratio(raw_mask),
        ink_density=_ink_density(raw_mask),
        stroke_width=_stroke_width(raw_mask),
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return cosine similarity in [0, 1].

    HOG descriptors are non-negative histograms, so cosine similarity is
    already in [0, 1]. The previous (cos+1)/2 shift compressed the range
    to [0.5, 1.0], masking differences between poor and good matches.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, cosine))


def _mse_score(a: np.ndarray, b: np.ndarray) -> float:
    """Return a normalized pixel similarity derived from MSE."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return max(0.0, 1.0 - mse / (255.0 * 255.0))


def _iou_score(a: np.ndarray, b: np.ndarray) -> float:
    """Return intersection-over-union for binary glyph masks."""
    a_fg = a > 0
    b_fg = b > 0
    union = int(np.logical_or(a_fg, b_fg).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(a_fg, b_fg).sum())
    return intersection / union


def _projection_score(a: np.ndarray, b: np.ndarray) -> float:
    """Return similarity for row/column ink distributions."""
    return max(0.0, 1.0 - float(np.mean(np.abs(a - b))))


def _hu_score(a: np.ndarray, b: np.ndarray) -> float:
    """Return a stable similarity based on Hu moments distance."""
    distance = float(np.mean(np.abs(a - b)))
    return 1.0 / (1.0 + distance)


def _aspect_score(a: float, b: float) -> float:
    """Return similarity for foreground width/height ratio."""
    return 1.0 / (1.0 + 1.0 * abs(np.log((a + 1e-6) / (b + 1e-6))))


def _density_score(a: float, b: float) -> float:
    """Return similarity for ink coverage inside the tight crop.

    Screenshots typically have lower ink density than clean renders due to
    antialiasing and background bleed, so the penalty is kept gentle.
    """
    return max(0.0, 1.0 - min(1.0, abs(a - b) * 2.0))


def _stroke_score(a: float, b: float) -> float:
    """
    Return similarity for normalized stroke width estimates.

    Uses a log-ratio penalty so doubling the stroke width halves the score,
    regardless of absolute scale.  The coefficient is kept gentle (1.5) because
    screenshots systematically have thinner strokes than clean renders due to
    antialiasing and display rendering differences.
    """
    return 1.0 / (1.0 + 1.5 * abs(np.log((a + 1e-6) / (b + 1e-6))))


def _similarity(a: _GlyphFeatures, b: _GlyphFeatures) -> float:
    """Weighted glyph similarity using raster, contour, and feature signals."""
    # ── Stretch-canvas comparison (width-normalised, aspect ignored) ─────────
    # Both images are stretched to the same (W×H) rectangle, so the comparison
    # is purely about shape/stroke style, not about scale or width differences.
    # This is the most reliable signal when the input is a screenshot crop,
    # because glyph height varies with subscript/superscript extent.
    stretch_ssim_score = max(
        0.0, cast(float, ssim(a.stretch_mask, b.stretch_mask, data_range=255))
    )
    stretch_pixel_score = 0.5 * _mse_score(
        a.stretch_mask, b.stretch_mask
    ) + 0.5 * _iou_score(a.stretch_mask, b.stretch_mask)

    # HOG on stretch canvas is aspect-invariant — critical for complex scripts
    # like Khmer where subscript characters change the effective glyph height,
    # causing the square-canvas HOG to compare at different scales.
    stretch_hog_score = _cosine_similarity(a.stretch_hog_vector, b.stretch_hog_vector)

    # ── Square-canvas signals ────────────────────────────────────────────────
    hog_score = _cosine_similarity(a.hog_vector, b.hog_vector)
    contour_score = _iou_score(a.contour_mask, b.contour_mask)
    hu_score = _hu_score(a.hu_moments, b.hu_moments)
    projection_score = _projection_score(a.projections, b.projections)

    # ── Scalar feature signals ───────────────────────────────────────────────
    aspect_score = _aspect_score(a.aspect_ratio, b.aspect_ratio)
    density_score = _density_score(a.ink_density, b.ink_density)
    stroke_score = _stroke_score(a.stroke_width, b.stroke_width)

    score = (
        _STRETCH_SSIM_WEIGHT * stretch_ssim_score
        + _STRETCH_PIXEL_WEIGHT * stretch_pixel_score
        + _STRETCH_HOG_WEIGHT * stretch_hog_score
        + _HOG_WEIGHT * hog_score
        + _STROKE_WEIGHT * stroke_score
        + _DENSITY_WEIGHT * density_score
        + _ASPECT_WEIGHT * aspect_score
        + _CONTOUR_WEIGHT * contour_score
        + _HU_WEIGHT * hu_score
        + _PROJECTION_WEIGHT * projection_score
    )
    return float(max(0.0, min(1.0, score)))


# ── public API ────────────────────────────────────────────────────────────────


def score_fonts(
    crop: Image.Image,
    text: str,
    fonts: dict[str, str],
    on_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[tuple[str, float, Image.Image]], int]:
    """
    Render *text* in every font from *fonts*, compare each rendering against
    *crop*, and return ``(top_5_results, candidate_count)`` where results are
    ``[(name, score, rendered_image)]`` sorted by descending composite score.

    *on_progress(done, total)* is called after each font if provided.
    *cancel_event*: when set, the loop stops and partial results are returned.
    """
    target = _normalise(crop)
    results: list[tuple[str, float, Image.Image]] = []

    # Strip invisible format characters before rendering so that their
    # presence in the user's text field doesn't affect the visual comparison.
    render_text = "".join(ch for ch in text if ord(ch) not in _INVISIBLE_CODEPOINTS)
    if not render_text.strip():
        render_text = text  # fall back to original if stripping empties it

    # Only score fonts that actually support every visible character in *text*.
    capable = {n: p for n, p in fonts.items() if font_supports_text(p, text)}
    total = len(capable)

    for i, (name, path) in enumerate(capable.items()):
        if cancel_event and cancel_event.is_set():
            break
        rendered = _render(render_text, path)
        if rendered is not None:
            score = _similarity(target, _normalise(rendered))
            results.append((name, score, rendered))
        if on_progress:
            on_progress(i + 1, total)

    results.sort(key=lambda x: -x[1])
    return results[:5], total
