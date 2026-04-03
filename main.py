"""Font Search — Find system fonts matching text in an image.

MIT License
Copyright (c) 2025 font-search contributors
"""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QRect, QPoint, Signal, QObject
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from PIL import Image

from app.font_engine import font_supports_text, get_system_fonts, score_fonts

# ── Constants ──────────────────────────────────────────────────────────────────

# PIL (width, height) used for result card previews
PREVIEW_W, PREVIEW_H = 240, 72

_KHMER_FONT_CANDIDATES = (
    "Noto Sans Khmer UI",
    "Noto Sans Khmer",
    "Khmer UI",
    "Khmer OS System",
    "DaunPenh",
    "Khmer Unicode Serif",
    "Leelawadee UI",
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _pil_to_pixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL Image to a QPixmap."""
    img_rgb = img.convert("RGB")
    data = img_rgb.tobytes("raw", "RGB")
    qimg = QImage(
        data,
        img_rgb.width,
        img_rgb.height,
        img_rgb.width * 3,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimg)


def _normalise_family_name(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", name.casefold())


def _resolve_script_font(
    candidates: tuple[str, ...],
    size: int,
    fonts: dict[str, str] | None = None,
    sample_text: str = "",
) -> QFont:
    """Return the first available font family from *candidates* via QFontDatabase."""
    db_families = {_normalise_family_name(f): f for f in QFontDatabase.families()}
    for name in candidates:
        match = db_families.get(_normalise_family_name(name))
        if match:
            return QFont(match, size)
    if sample_text and fonts:
        for family, path in fonts.items():
            if font_supports_text(path, sample_text):
                match = db_families.get(_normalise_family_name(family))
                if match:
                    return QFont(match, size)
    return QFont()  # system default


# ── Worker signal carriers ─────────────────────────────────────────────────────


class _FontLoadSignals(QObject):
    finished = Signal(dict)


class _SearchSignals(QObject):
    progress = Signal(int, int)      # done, total
    finished = Signal(list, object, int)  # results, crop_img, candidate_count


# ── CropWidget ─────────────────────────────────────────────────────────────────


class CropWidget(QWidget):
    """Displays an image and lets the user drag a selection rectangle."""

    file_dropped = Signal(str)
    zoom_changed = Signal(float)  # emits zoom multiplier relative to fit (1.0 = fit)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setAcceptDrops(True)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._orig: Image.Image | None = None
        self._pixmap: QPixmap | None = None
        self._fit_scale = 1.0   # scale that fits image to widget
        self._zoom = 1.0        # multiplier on top of fit scale
        self._scale = 1.0       # _fit_scale * _zoom  (used by crop property)
        self._off_x = 0
        self._off_y = 0
        self._drag_start: QPoint | None = None
        self._sel: QRect | None = None
        self._pan_start: QPoint | None = None
        self._pan_off_start: tuple[int, int] = (0, 0)
        self._mode = "select"  # "select" or "move"

    # ── public ──

    def load_image(self, img: Image.Image) -> None:
        self._orig = img
        self._sel = None
        self._zoom = 1.0
        self._off_x = 0
        self._off_y = 0
        self._update_pixmap()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def clear_selection(self) -> None:
        self._sel = None
        self.update()

    @property
    def crop(self) -> Image.Image | None:
        """Return the selected region of the original image, or None."""
        if self._orig is None or self._sel is None:
            return None
        sel = self._sel.normalized()
        s = self._scale
        ix0 = int((sel.left() - self._off_x) / s)
        iy0 = int((sel.top() - self._off_y) / s)
        ix1 = int((sel.right() - self._off_x) / s)
        iy1 = int((sel.bottom() - self._off_y) / s)
        ow, oh = self._orig.size
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(ow, ix1), min(oh, iy1)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return self._orig.crop((ix0, iy0, ix1, iy1))

    # ── private ──

    def _update_pixmap(self) -> None:
        if self._orig is None:
            return
        cw = max(self.width(), 1)
        ch = max(self.height(), 1)
        iw, ih = self._orig.size
        self._fit_scale = min(cw / iw, ch / ih, 1.0)
        self._scale = self._fit_scale * self._zoom
        nw = max(1, int(iw * self._scale))
        nh = max(1, int(ih * self._scale))
        # Center when smaller than widget; clamp when larger
        if nw <= cw:
            self._off_x = (cw - nw) // 2
        else:
            self._off_x = max(cw - nw, min(0, self._off_x))
        if nh <= ch:
            self._off_y = (ch - nh) // 2
        else:
            self._off_y = max(ch - nh, min(0, self._off_y))
        resized = self._orig.resize((nw, nh), Image.LANCZOS)
        self._pixmap = _pil_to_pixmap(resized)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        if self._pixmap:
            painter.drawPixmap(self._off_x, self._off_y, self._pixmap)
        if self._sel:
            pen = QPen(QColor("#00AAFF"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._sel.normalized())

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._update_pixmap()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == "move":
                self._pan_start = event.position().toPoint()
                self._pan_off_start = (self._off_x, self._off_y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._drag_start = event.position().toPoint()
                self._sel = None
                self.update()
        elif event.button() in (
            Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton
        ):
            self._pan_start = event.position().toPoint()
            self._pan_off_start = (self._off_x, self._off_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._sel = QRect(self._drag_start, event.position().toPoint())
            self.update()
        elif self._pan_start is not None and self._pixmap is not None:
            dx = int(event.position().x() - self._pan_start.x())
            dy = int(event.position().y() - self._pan_start.y())
            cw, ch = self.width(), self.height()
            nw, nh = self._pixmap.width(), self._pixmap.height()
            new_x = self._pan_off_start[0] + dx
            new_y = self._pan_off_start[1] + dy
            self._off_x = new_x if nw <= cw else max(cw - nw, min(0, new_x))
            self._off_y = new_y if nh <= ch else max(ch - nh, min(0, new_y))
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start is not None:
                self._sel = QRect(self._drag_start, event.position().toPoint())
                self._drag_start = None
                self.update()
            elif self._pan_start is not None:
                self._pan_start = None
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.button() in (
            Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton
        ):
            self._pan_start = None
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if self._mode == "move"
                else Qt.CursorShape.CrossCursor
            )

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._orig is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._apply_zoom(self._zoom * factor, pivot=event.position())

    # ── public zoom API ──

    def zoom_in(self) -> None:
        self._apply_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(self._zoom / 1.25)

    def zoom_reset(self) -> None:
        self._apply_zoom(1.0)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == "move":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _apply_zoom(self, new_zoom: float, pivot=None) -> None:
        if self._orig is None:
            return
        new_zoom = max(0.5, min(10.0, new_zoom))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        cw, ch = self.width(), self.height()
        iw, ih = self._orig.size
        old_scale = self._scale
        self._zoom = new_zoom
        new_scale = self._fit_scale * self._zoom
        self._scale = new_scale
        nw = max(1, int(iw * new_scale))
        nh = max(1, int(ih * new_scale))
        if pivot is not None:
            mx, my = pivot.x(), pivot.y()
            self._off_x = int(mx - (mx - self._off_x) * new_scale / old_scale)
            self._off_y = int(my - (my - self._off_y) * new_scale / old_scale)
        # Clamp / center
        if nw <= cw:
            self._off_x = (cw - nw) // 2
        else:
            self._off_x = max(cw - nw, min(0, self._off_x))
        if nh <= ch:
            self._off_y = (ch - nh) // 2
        else:
            self._off_y = max(ch - nh, min(0, self._off_y))
        resized = self._orig.resize((nw, nh), Image.LANCZOS)
        self._pixmap = _pil_to_pixmap(resized)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())


# ── ResultCard ─────────────────────────────────────────────────────────────────


class ResultCard(QFrame):
    """A single result row: rank, font name, score, and side-by-side previews."""

    def __init__(
        self,
        rank: int,
        name: str,
        score: float,
        crop_img: Image.Image,
        rend_img: Image.Image,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)

        # ── header row ──
        header = QHBoxLayout()
        pct = max(0.0, score) * 100

        name_lbl = QLabel(f"#{rank}  {name}")
        name_font = QFont("Segoe UI", 11)
        name_font.setBold(True)
        name_lbl.setFont(name_font)

        score_lbl = QLabel(f"{pct:.1f} %")
        score_lbl.setFont(QFont("Segoe UI", 11))
        score_lbl.setStyleSheet("color: #007ACC;")
        score_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        header.addWidget(name_lbl)
        header.addStretch()
        header.addWidget(score_lbl)
        layout.addLayout(header)

        # ── preview row ──
        preview = QHBoxLayout()
        for label_text, img in (("Original", crop_img), ("Rendered", rend_img)):
            col = QVBoxLayout()
            col.setSpacing(2)

            caption = QLabel(label_text)
            caption.setFont(QFont("Segoe UI", 9))
            col.addWidget(caption)

            thumb_lbl = QLabel()
            pixmap = _pil_to_pixmap(
                img.convert("RGB").resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
            )
            thumb_lbl.setPixmap(pixmap)
            thumb_lbl.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
            col.addWidget(thumb_lbl)

            preview.addLayout(col)
            preview.addSpacing(12)

        layout.addLayout(preview)


# ── FontSearchApp ──────────────────────────────────────────────────────────────


class FontSearchApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Font Search")
        self.resize(860, 760)
        self.setMinimumSize(720, 600)

        self._image: Image.Image | None = None
        self._fonts: dict[str, str] = {}
        self._cancel_event = threading.Event()
        self._font_load_signals: _FontLoadSignals | None = None
        self._search_signals: _SearchSignals | None = None

        self._build_ui()
        self._load_fonts_async()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(4)

        # ── Image group ──
        img_group = QGroupBox("Image")
        img_layout = QVBoxLayout(img_group)

        btn_bar = QHBoxLayout()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        btn_bar.addWidget(browse_btn)

        clear_btn = QPushButton("Clear selection")
        clear_btn.clicked.connect(self._clear_selection)
        btn_bar.addWidget(clear_btn)

        btn_bar.addSpacing(8)
        self._select_btn = QPushButton("Select")
        self._select_btn.setCheckable(True)
        self._select_btn.setChecked(True)
        self._select_btn.setFixedWidth(54)
        self._select_btn.setToolTip("Selection mode — left-drag to mark crop region")
        btn_bar.addWidget(self._select_btn)

        self._move_btn = QPushButton("Move")
        self._move_btn.setCheckable(True)
        self._move_btn.setFixedWidth(48)
        self._move_btn.setToolTip("Move mode — left-drag to pan the image")
        btn_bar.addWidget(self._move_btn)

        btn_bar.addSpacing(8)
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(28)
        zoom_out_btn.setToolTip("Zoom out  (scroll wheel)")
        btn_bar.addWidget(zoom_out_btn)

        zoom_reset_btn = QPushButton("Fit")
        zoom_reset_btn.setFixedWidth(36)
        zoom_reset_btn.setToolTip("Reset zoom to fit")
        btn_bar.addWidget(zoom_reset_btn)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(28)
        zoom_in_btn.setToolTip("Zoom in  (scroll wheel)")
        btn_bar.addWidget(zoom_in_btn)

        self._zoom_lbl = QLabel("fit")
        self._zoom_lbl.setStyleSheet("color: grey;")
        self._zoom_lbl.setFixedWidth(48)
        btn_bar.addWidget(self._zoom_lbl)

        self._hint_lbl = QLabel("Drop an image here, or click Browse")
        self._hint_lbl.setStyleSheet("color: grey;")
        btn_bar.addWidget(self._hint_lbl)
        btn_bar.addStretch()
        img_layout.addLayout(btn_bar)

        self._canvas = CropWidget()
        self._canvas.file_dropped.connect(self._load_image)
        self._canvas.zoom_changed.connect(self._on_zoom_changed)

        # Wire up mode buttons
        self._select_btn.clicked.connect(lambda: self._set_canvas_mode("select"))
        self._move_btn.clicked.connect(lambda: self._set_canvas_mode("move"))

        # Wire up zoom buttons now that the canvas exists
        zoom_in_btn.clicked.connect(self._canvas.zoom_in)
        zoom_out_btn.clicked.connect(self._canvas.zoom_out)
        zoom_reset_btn.clicked.connect(self._canvas.zoom_reset)

        img_layout.addWidget(self._canvas)
        root_layout.addWidget(img_group, stretch=3)

        # ── Controls row ──
        ctrl_widget = QWidget()
        ctrl = QHBoxLayout(ctrl_widget)
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.addWidget(QLabel("Text in image:"))

        self._text_entry = QLineEdit()
        self._text_entry.setMinimumWidth(280)
        self._text_entry.returnPressed.connect(self._start_search)
        self._text_entry.setFont(_resolve_script_font(_KHMER_FONT_CANDIDATES, 11))
        ctrl.addWidget(self._text_entry)

        self._find_btn = QPushButton("Find Matching Fonts")
        self._find_btn.clicked.connect(self._start_search)
        ctrl.addWidget(self._find_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_search)
        self._cancel_btn.setEnabled(False)
        ctrl.addWidget(self._cancel_btn)
        ctrl.addStretch()
        root_layout.addWidget(ctrl_widget, stretch=0)

        # ── Status / progress row ──
        status_widget = QWidget()
        status_row = QHBoxLayout(status_widget)
        status_row.setContentsMargins(0, 0, 0, 0)

        self._status_lbl = QLabel("Loading fonts…")
        self._status_lbl.setStyleSheet("color: grey;")
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        status_row.addWidget(self._progress)
        root_layout.addWidget(status_widget, stretch=0)

        # ── Results panel ──
        results_group = QGroupBox("Top 5 results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(4, 4, 4, 4)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._res_container = QWidget()
        self._res_layout = QVBoxLayout(self._res_container)
        self._res_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._res_layout.setSpacing(6)
        self._scroll_area.setWidget(self._res_container)

        results_layout.addWidget(self._scroll_area)
        root_layout.addWidget(results_group, stretch=2)

    # ── Font loading ───────────────────────────────────────────────────────────

    def _load_fonts_async(self) -> None:
        signals = _FontLoadSignals()
        signals.finished.connect(self._on_fonts_ready)
        self._font_load_signals = signals  # keep alive

        def _work() -> None:
            fonts = get_system_fonts()
            signals.finished.emit(fonts)

        threading.Thread(target=_work, daemon=True).start()

    def _on_fonts_ready(self, fonts: dict[str, str]) -> None:
        self._fonts = fonts
        self._status_lbl.setText(f"Ready — {len(fonts)} fonts found")
        # Re-resolve Khmer font now that the full font list is available
        khmer_font = _resolve_script_font(
            _KHMER_FONT_CANDIDATES, 11, fonts, "ភាសាខ្មែរ"
        )
        self._text_entry.setFont(khmer_font)

    # ── Image handling ─────────────────────────────────────────────────────────

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp);;All files (*.*)",
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open image", str(exc))
            return
        self._image = img
        self._canvas.load_image(img)
        self._hint_lbl.setText(Path(path).name)

    def _clear_selection(self) -> None:
        if self._image:
            self._canvas.clear_selection()

    def _on_zoom_changed(self, zoom: float) -> None:
        if abs(zoom - 1.0) < 0.02:
            self._zoom_lbl.setText("fit")
        else:
            self._zoom_lbl.setText(f"×{zoom:.1f}")

    def _set_canvas_mode(self, mode: str) -> None:
        self._canvas.set_mode(mode)
        self._select_btn.setChecked(mode == "select")
        self._move_btn.setChecked(mode == "move")

    # ── Search ─────────────────────────────────────────────────────────────────

    def _start_search(self) -> None:
        if self._image is None:
            QMessageBox.warning(self, "No image", "Please load an image first.")
            return
        text = self._text_entry.text().strip()
        if not text:
            QMessageBox.warning(
                self, "No text", "Please type the text shown in the image."
            )
            return
        if not self._fonts:
            QMessageBox.warning(
                self,
                "Fonts not ready",
                "System fonts are still loading, please wait.",
            )
            return

        crop = self._canvas.crop or self._image
        self._cancel_event.clear()
        self._find_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._clear_results()
        self._progress.setValue(0)

        signals = _SearchSignals()
        signals.progress.connect(self._on_progress)
        signals.finished.connect(self._on_search_done)
        self._search_signals = signals  # keep alive

        cancel = self._cancel_event

        def _work() -> None:
            def _tick(done: int, total: int) -> None:
                if not cancel.is_set():
                    signals.progress.emit(done, total)

            results, candidate_count = score_fonts(crop, text, self._fonts, _tick, cancel)
            signals.finished.emit(results, crop, candidate_count)

        threading.Thread(target=_work, daemon=True).start()

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self._progress.setValue(0)
            self._status_lbl.setText("No compatible fonts found for this text.")
            return

        self._progress.setValue(int(done / total * 100))
        self._status_lbl.setText(f"Comparing font {done} / {total}…")

    def _cancel_search(self) -> None:
        self._cancel_event.set()
        self._cancel_btn.setEnabled(False)
        self._status_lbl.setText("Cancelling…")

    def _on_search_done(
        self,
        results: list[tuple[str, float, Image.Image]],
        crop: Image.Image,
        candidate_count: int,
    ) -> None:
        self._find_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

        if not results:
            self._progress.setValue(0)
            if self._cancel_event.is_set():
                self._status_lbl.setText("Search cancelled.")
            elif candidate_count == 0:
                self._status_lbl.setText("No installed fonts can render that text.")
            else:
                self._status_lbl.setText(
                    f"No comparable results — {candidate_count} compatible fonts were checked."
                )
            return

        self._progress.setValue(100)
        self._status_lbl.setText(f"Done — top {len(results)} matches shown")
        for rank, (name, score, rend) in enumerate(results, 1):
            card = ResultCard(rank, name, score, crop, rend)
            self._res_layout.addWidget(card)

    def _clear_results(self) -> None:
        while self._res_layout.count():
            item = self._res_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _icon = QIcon(str(Path(__file__).parent / "assets" / "logo.ico"))
    app.setWindowIcon(_icon)
    window = FontSearchApp()
    window.setWindowIcon(_icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
