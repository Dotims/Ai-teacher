"""
gui.py – Stealth overlay window with control buttons.

Frameless, always-on-top PyQt6 widget with SetWindowDisplayAffinity
so the window is invisible to screen-sharing apps (Zoom, Teams, Discord).
Includes clickable buttons for voice, screenshot, and stealth toggle.
"""

import ctypes
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QApplication,
    QGraphicsDropShadowEffect,
)


# Windows API constants
WDA_EXCLUDEFROMCAPTURE = 0x11
WDA_NONE = 0x00

# Shared button style template
_BTN_STYLE = """
    QPushButton {{
        color: {fg};
        background: {bg};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {hover_bg};
    }}
    QPushButton:pressed {{
        background: {press_bg};
    }}
"""


def _btn_style(fg, bg, border, hover_bg, press_bg) -> str:
    return _BTN_STYLE.format(
        fg=fg, bg=bg, border=border, hover_bg=hover_bg, press_bg=press_bg
    )


class AssistantWindow(QWidget):
    """Stealth overlay that displays AI-generated answers."""

    # Signals emitted by buttons (connected in main.py)
    voice_toggle_clicked = pyqtSignal()
    solve_screen_clicked = pyqtSignal()
    stealth_toggled = pyqtSignal(bool)  # True = stealth ON

    def __init__(self) -> None:
        super().__init__()

        self._stealth_on = True  # stealth enabled by default

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._drag_pos: QPoint | None = None
        self._init_ui()
        self._position_window()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        self._container = QWidget(self)
        self._container.setObjectName("container")
        self._set_container_border("rgba(100, 108, 255, 0.4)")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(100, 108, 255, 80))
        shadow.setOffset(0, 4)
        self._container.setGraphicsEffect(shadow)

        # ----- Title bar -----
        title_bar = QWidget()
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet("background: transparent;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 4, 8, 0)

        self._mode_dot = QLabel("●")
        self._mode_dot.setStyleSheet("color: #646cff; font-size: 10px;")
        title_layout.addWidget(self._mode_dot)

        title_label = QLabel("Assistant")
        title_label.setStyleSheet(
            "color: rgba(255,255,255,0.85); font-size: 13px; font-weight: 600;"
        )
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            "color: rgba(255,255,255,0.45); font-size: 11px;"
        )
        title_layout.addWidget(self._status_label)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            """
            QPushButton {
                color: rgba(255,255,255,0.5);
                background: transparent;
                border: none;
                font-size: 14px;
                border-radius: 12px;
            }
            QPushButton:hover {
                color: #ff4d6a;
                background: rgba(255,77,106,0.15);
            }
            """
        )
        close_btn.clicked.connect(self.hide)
        title_layout.addWidget(close_btn)

        # ----- Button bar -----
        btn_bar = QWidget()
        btn_bar.setStyleSheet("background: transparent;")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(10, 2, 10, 4)
        btn_layout.setSpacing(6)

        # Voice toggle
        self._voice_btn = QPushButton("🎙️ Start nasłuchiwanie")
        self._voice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._voice_btn.setStyleSheet(
            _btn_style(
                "#fff", "rgba(255,77,106,0.15)", "rgba(255,77,106,0.4)",
                "rgba(255,77,106,0.3)", "rgba(255,77,106,0.45)",
            )
        )
        self._voice_btn.clicked.connect(self.voice_toggle_clicked.emit)
        btn_layout.addWidget(self._voice_btn)

        # Solve from screen
        self._solve_btn = QPushButton("📸 Rozwiąż z ekranu")
        self._solve_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._solve_btn.setStyleSheet(
            _btn_style(
                "#fff", "rgba(100,108,255,0.15)", "rgba(100,108,255,0.4)",
                "rgba(100,108,255,0.3)", "rgba(100,108,255,0.45)",
            )
        )
        self._solve_btn.clicked.connect(self.solve_screen_clicked.emit)
        btn_layout.addWidget(self._solve_btn)

        # Stealth toggle
        self._stealth_btn = QPushButton("👁️ Stealth ON")
        self._stealth_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stealth_btn.setStyleSheet(
            _btn_style(
                "#fff", "rgba(46,204,113,0.15)", "rgba(46,204,113,0.4)",
                "rgba(46,204,113,0.3)", "rgba(46,204,113,0.45)",
            )
        )
        self._stealth_btn.clicked.connect(self._on_stealth_toggle)
        btn_layout.addWidget(self._stealth_btn)

        # ----- Response area -----
        self._response_area = QTextEdit()
        self._response_area.setReadOnly(True)
        self._response_area.setFont(QFont("Consolas", 10))
        self._response_area.setStyleSheet(
            """
            QTextEdit {
                background: transparent;
                color: rgba(255,255,255,0.9);
                border: none;
                padding: 8px 12px;
                selection-background-color: rgba(100,108,255,0.35);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(100,108,255,0.4);
                border-radius: 3px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

        # ----- Layout -----
        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(0, 0, 0, 8)
        container_layout.setSpacing(0)
        container_layout.addWidget(title_bar)
        container_layout.addWidget(btn_bar)
        container_layout.addWidget(self._response_area)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(self._container)

        self.resize(520, 460)

    def _set_container_border(self, color: str) -> None:
        self._container.setStyleSheet(
            f"""
            #container {{
                background-color: rgba(18, 18, 24, 235);
                border: 1px solid {color};
                border-radius: 12px;
            }}
            """
        )

    def _position_window(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.right() - self.width() - 20
            y = geo.bottom() - self.height() - 20
            self.move(x, y)

    # ------------------------------------------------------------------
    # Stealth
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._stealth_on:
            self._apply_display_affinity(WDA_EXCLUDEFROMCAPTURE)

    def _apply_display_affinity(self, flag: int) -> None:
        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, flag)
        except Exception:
            pass

    def _on_stealth_toggle(self) -> None:
        self._stealth_on = not self._stealth_on
        if self._stealth_on:
            self._stealth_btn.setText("👁️ Stealth ON")
            self._stealth_btn.setStyleSheet(
                _btn_style(
                    "#fff", "rgba(46,204,113,0.15)", "rgba(46,204,113,0.4)",
                    "rgba(46,204,113,0.3)", "rgba(46,204,113,0.45)",
                )
            )
            self._apply_display_affinity(WDA_EXCLUDEFROMCAPTURE)
        else:
            self._stealth_btn.setText("👁️ Stealth OFF")
            self._stealth_btn.setStyleSheet(
                _btn_style(
                    "#fff", "rgba(255,165,0,0.15)", "rgba(255,165,0,0.4)",
                    "rgba(255,165,0,0.3)", "rgba(255,165,0,0.45)",
                )
            )
            self._apply_display_affinity(WDA_NONE)
        self.stealth_toggled.emit(self._stealth_on)

    # ------------------------------------------------------------------
    # Dragging
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_loading(self, loading: bool) -> None:
        if loading:
            self._mode_dot.setStyleSheet("color: #646cff; font-size: 10px;")
            self._status_label.setText("⏳ Analizuję…")
            self._response_area.setPlainText("")
        else:
            self._status_label.setText("")

    def set_response(self, text: str) -> None:
        self._status_label.setText("")
        self._response_area.setMarkdown(text)

    # ---- Voice mode ----

    def set_voice_active(self, active: bool) -> None:
        if active:
            self._mode_dot.setStyleSheet("color: #ff4d6a; font-size: 10px;")
            self._status_label.setText("🎙️ Nasłuchiwanie…")
            self._voice_btn.setText("⏹️ Stop nasłuchiwanie")
            self._set_container_border("rgba(255, 77, 106, 0.5)")
        else:
            self._mode_dot.setStyleSheet("color: #646cff; font-size: 10px;")
            self._status_label.setText("")
            self._voice_btn.setText("🎙️ Start nasłuchiwanie")
            self._set_container_border("rgba(100, 108, 255, 0.4)")

    def append_voice_text(self, text: str) -> None:
        current = self._response_area.toPlainText()
        self._response_area.setPlainText(current + text)
        scrollbar = self._response_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
