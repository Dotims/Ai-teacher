"""
gui.py – Stealth overlay window with control buttons.

Frameless, always-on-top PyQt6 widget with SetWindowDisplayAffinity
so the window is invisible to screen-sharing apps (Zoom, Teams, Discord).
Includes clickable buttons for voice, screenshot, and stealth toggle.
"""

import ctypes
import markdown
from pygments.formatters import HtmlFormatter
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
    QComboBox,
    QDialog,
    QLineEdit,
    QSizePolicy,
)
import random


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


class StealthConfirmDialog(QDialog):
    """Custom frameless dialog asking for a 3-digit PIN to disable stealth mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedSize(320, 180)

        self._expected_pin = str(random.randint(100, 999))
        self._init_ui()

    def _init_ui(self):
        container = QWidget(self)
        container.setFixedSize(320, 180)
        container.setStyleSheet(
            """
            QWidget {
                background-color: rgba(20, 20, 25, 245);
                border: 1px solid rgba(255, 77, 106, 0.5);
                border-radius: 12px;
                color: rgba(255, 255, 255, 0.95);
                font-family: 'Inter', "Segoe UI", sans-serif;
            }
            """
        )

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("Wyłączenie Stealth Mode")
        title.setStyleSheet("font-size: 15px; font-weight: 600; color: #ff4d6a; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel(f"Przepisz kod <b>{self._expected_pin}</b> aby potwierdzić.")
        desc.setStyleSheet("font-size: 13px; border: none;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        self._input = QLineEdit()
        self._input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._input.setStyleSheet(
            """
            QLineEdit {
                background: rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 6px;
                font-size: 16px;
                letter-spacing: 4px;
            }
            QLineEdit:focus {
                border: 1px solid #ff4d6a;
            }
            """
        )
        self._input.setMaxLength(3)
        self._input.textChanged.connect(self._check_pin)
        layout.addWidget(self._input)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        cancel_btn = QPushButton("Anuluj")
        cancel_btn.setCursor(Qt.CursorShape.ArrowCursor)
        cancel_btn.setStyleSheet(_btn_style("#fff", "rgba(100,100,100,0.2)", "rgba(100,100,100,0.5)", "rgba(100,100,100,0.4)", "rgba(100,100,100,0.6)"))
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)

    def _check_pin(self, text):
        if text == self._expected_pin:
            self.accept()

class AssistantWindow(QWidget):
    """Stealth overlay that displays AI-generated answers."""

    # Signals emitted by buttons (connected in main.py)
    voice_toggle_clicked = pyqtSignal()
    solve_screen_clicked = pyqtSignal()
    stealth_toggled = pyqtSignal(bool)  # True = stealth ON
    language_override_changed = pyqtSignal(str)  # "auto", "pl", "en"
    model_changed = pyqtSignal(str)  # emits model name e.g. "gpt-4o-mini"

    _DEFAULT_FONT_SIZE = 14.5

    def __init__(self) -> None:
        super().__init__()

        self._stealth_on = True  # stealth enabled by default
        self._response_history: list[dict[str, str]] = []
        self._history_index = -1
        self._font_size: float = self._DEFAULT_FONT_SIZE
        self._collapsed = False  # minimize state

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

        # No shadow for stealth
        # shadow = QGraphicsDropShadowEffect(self)
        # shadow.setBlurRadius(30)
        # shadow.setColor(QColor(100, 108, 255, 80))
        # shadow.setOffset(0, 4)
        # self._container.setGraphicsEffect(shadow)

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
        
        # ----- Prompt type selector -----
        self.prompt_combo = QComboBox()
        self.prompt_combo.addItems(["Rozmowa HR (PL/ENG)", "Pytania Techniczne", "Live Coding"])
        self.prompt_combo.setCursor(Qt.CursorShape.ArrowCursor)
        self.prompt_combo.setMinimumWidth(130)
        self.prompt_combo.setMaximumWidth(160)
        self.prompt_combo.setStyleSheet("""
            QComboBox {
                background: rgba(20,20,25,0.95);
                color: rgba(255,255,255,0.9);
                border: 1px solid rgba(80,80,80,0.5);
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: rgba(20,20,25,1);
                color: rgba(255,255,255,0.9);
                selection-background-color: rgba(80,80,80,0.8);
            }
        """)
        title_layout.addWidget(self.prompt_combo)

        # ----- Language override selector -----
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["🌐 Auto", "🇵🇱 Polski", "🇬🇧 English"])
        self.lang_combo.setCursor(Qt.CursorShape.ArrowCursor)
        self.lang_combo.setMinimumWidth(80)
        self.lang_combo.setMaximumWidth(100)
        self.lang_combo.setStyleSheet("""
            QComboBox {
                background: rgba(20,20,25,0.95);
                color: rgba(255,255,255,0.9);
                border: 1px solid rgba(80,80,80,0.5);
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: rgba(20,20,25,1);
                color: rgba(255,255,255,0.9);
                selection-background-color: rgba(80,80,80,0.8);
            }
        """)
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        title_layout.addWidget(self.lang_combo)

        # ----- Model selector -----
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o"])
        self.model_combo.setCursor(Qt.CursorShape.ArrowCursor)
        self.model_combo.setMinimumWidth(80)
        self.model_combo.setMaximumWidth(100)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: rgba(20,20,25,0.95);
                color: rgba(255,255,255,0.9);
                border: 1px solid rgba(80,80,80,0.5);
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: rgba(20,20,25,1);
                color: rgba(255,255,255,0.9);
                selection-background-color: rgba(80,80,80,0.8);
            }
        """)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        title_layout.addWidget(self.model_combo)

        # ----- Font size controls -----
        _fs_btn_css = """
            QPushButton {
                color: rgba(255,255,255,0.45);
                background: transparent;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                padding: 1px 5px;
            }
            QPushButton:hover {
                color: rgba(255,255,255,0.85);
                background: rgba(255,255,255,0.08);
            }
        """

        fs_minus = QPushButton("A−")
        fs_minus.setFixedSize(26, 22)
        fs_minus.setCursor(Qt.CursorShape.ArrowCursor)
        fs_minus.setStyleSheet(_fs_btn_css)
        fs_minus.setToolTip("Zmniejsz czcionkę")
        fs_minus.clicked.connect(lambda: self._change_font_size(-1))
        title_layout.addWidget(fs_minus)

        fs_reset = QPushButton("A")
        fs_reset.setFixedSize(22, 22)
        fs_reset.setCursor(Qt.CursorShape.ArrowCursor)
        fs_reset.setStyleSheet(_fs_btn_css)
        fs_reset.setToolTip("Domyślny rozmiar czcionki")
        fs_reset.clicked.connect(lambda: self._change_font_size(0, reset=True))
        title_layout.addWidget(fs_reset)

        fs_plus = QPushButton("A+")
        fs_plus.setFixedSize(26, 22)
        fs_plus.setCursor(Qt.CursorShape.ArrowCursor)
        fs_plus.setStyleSheet(_fs_btn_css)
        fs_plus.setToolTip("Zwiększ czcionkę")
        fs_plus.clicked.connect(lambda: self._change_font_size(1))
        title_layout.addWidget(fs_plus)

        # ----- Token Info Label -----
        self._token_label = QLabel("")
        self._token_label.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 10px; padding-right: 10px;")
        self._token_label.setFixedWidth(190)
        self._token_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._token_label.setToolTip("")
        title_layout.addWidget(self._token_label)

        title_layout.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            "color: rgba(255,255,255,0.45); font-size: 11px;"
        )
        self._status_label.setMaximumWidth(260)
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        title_layout.addWidget(self._status_label)

        # Minimize button
        minimize_btn = QPushButton("─")
        minimize_btn.setFixedSize(24, 24)
        minimize_btn.setCursor(Qt.CursorShape.ArrowCursor)
        minimize_btn.setStyleSheet(
            """
            QPushButton {
                color: rgba(255,255,255,0.5);
                background: transparent;
                border: none;
                font-size: 14px;
                border-radius: 12px;
            }
            QPushButton:hover {
                color: #646cff;
                background: rgba(100,108,255,0.15);
            }
            """
        )
        minimize_btn.clicked.connect(self._on_minimize_toggle)
        title_layout.addWidget(minimize_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.ArrowCursor)
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
        close_btn.clicked.connect(self._on_close_click)
        title_layout.addWidget(close_btn)

        # ----- Button bar -----
        self._btn_bar = QWidget()
        self._btn_bar.setStyleSheet("background: transparent;")
        btn_layout = QHBoxLayout(self._btn_bar)
        btn_layout.setContentsMargins(10, 2, 10, 4)
        btn_layout.setSpacing(6)

        # Voice toggle
        self._voice_btn = QPushButton("🎙️ Start nasłuchiwanie")
        self._voice_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._voice_btn.setMinimumWidth(120)
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
        self._solve_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._solve_btn.setMinimumWidth(120)
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
        self._stealth_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._stealth_btn.setMinimumWidth(120)
        self._stealth_btn.setStyleSheet(
            _btn_style(
                "#fff", "rgba(46,204,113,0.15)", "rgba(46,204,113,0.4)",
                "rgba(46,204,113,0.3)", "rgba(46,204,113,0.45)",
            )
        )
        self._stealth_btn.clicked.connect(self._on_stealth_toggle)
        btn_layout.addWidget(self._stealth_btn)

        # ----- Response history navigation -----
        self._nav_bar = QWidget()
        self._nav_bar.setStyleSheet("background: transparent;")
        nav_layout = QHBoxLayout(self._nav_bar)
        nav_layout.setContentsMargins(10, 0, 10, 4)
        nav_layout.setSpacing(6)
        nav_layout.addStretch()

        self._prev_btn = QPushButton("◀ Poprzednia")
        self._prev_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._prev_btn.setMinimumWidth(80)
        self._prev_btn.setStyleSheet(
            _btn_style(
                "#fff", "rgba(80,80,80,0.15)", "rgba(120,120,120,0.35)",
                "rgba(100,100,100,0.25)", "rgba(120,120,120,0.35)",
            )
        )
        self._prev_btn.clicked.connect(self._show_previous_response)
        nav_layout.addWidget(self._prev_btn)

        self._history_label = QLabel("0/0")
        self._history_label.setStyleSheet(
            "color: rgba(255,255,255,0.55); font-size: 11px; min-width: 42px;"
        )
        self._history_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._history_label)

        self._next_btn = QPushButton("Następna ▶")
        self._next_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._next_btn.setMinimumWidth(80)
        self._next_btn.setStyleSheet(
            _btn_style(
                "#fff", "rgba(80,80,80,0.15)", "rgba(120,120,120,0.35)",
                "rgba(100,100,100,0.25)", "rgba(120,120,120,0.35)",
            )
        )
        self._next_btn.clicked.connect(self._show_next_response)
        nav_layout.addWidget(self._next_btn)

        # ----- Response area -----
        self._response_area = QTextEdit()
        self._response_area.setReadOnly(True)
        self._response_area.setFont(QFont("Consolas", 10))
        self._response_area.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        # Prevent content from expanding the window width
        self._response_area.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._response_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._response_area.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        # Generate Pygments CSS for dark style (e.g., 'monokai')
        self._pygments_css = HtmlFormatter(style='monokai').get_style_defs('.codehilite')

        self._response_area.setStyleSheet(
            """
            QTextEdit {
                background: transparent;
                color: rgba(255,255,255,0.95);
                border: none;
                padding: 10px 14px;
                selection-background-color: rgba(100,108,255,0.5);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(100,100,100,0.7);
                border-radius: 4px;
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
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(title_bar)
        container_layout.addWidget(self._btn_bar)
        container_layout.addWidget(self._nav_bar)
        container_layout.addWidget(self._response_area)

        # Custom resize handle drawn perfectly at the corner
        self._handle_widget = QWidget()
        handle_layout = QHBoxLayout(self._handle_widget)
        handle_layout.setContentsMargins(0, 0, 4, 4)
        handle_layout.addStretch()
        self._resize_handle = QLabel("↘")
        self._resize_handle.setStyleSheet("color: rgba(255,255,255,0.25); font-size: 14px;")
        self._resize_handle.setCursor(Qt.CursorShape.ArrowCursor)
        handle_layout.addWidget(self._resize_handle)
        container_layout.addWidget(self._handle_widget)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(self._container)

        self.resize(520, 460)
        self._update_history_nav()

    def _set_container_border(self, color: str) -> None:
        self._container.setStyleSheet(
            f"""
            #container {{
                background-color: rgba(15, 15, 15, 245);
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
        if self._stealth_on:
            # We are trying to turn it OFF.
            dialog = StealthConfirmDialog(self)
            # Center dialog over the main window
            dialog.move(
                self.geometry().center() - dialog.rect().center()
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

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

    def _on_close_click(self) -> None:
        """Close with PIN confirmation to prevent accidental clicks."""
        dialog = StealthConfirmDialog(self)
        dialog.move(self.geometry().center() - dialog.rect().center())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.hide()

    def _on_minimize_toggle(self) -> None:
        """Toggle collapse: hide response area, button bar, nav bar, resize handle."""
        self._collapsed = not self._collapsed
        for widget in (self._btn_bar, self._nav_bar, self._response_area, self._handle_widget):
            widget.setVisible(not self._collapsed)

        outer_layout = self.layout()
        if self._collapsed:
            self._saved_size = self.size()
            # Shrink outer margins so title bar fills the window tightly
            outer_layout.setContentsMargins(2, 2, 2, 2)
            self.setFixedHeight(44)
        else:
            outer_layout.setContentsMargins(12, 12, 12, 12)
            # Remove fixed height constraint and restore previous size
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            if hasattr(self, '_saved_size'):
                self.resize(self._saved_size)

    # ------------------------------------------------------------------
    # Dragging & Custom Resizing
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            # If pointer is at the bottom-right corner (within 30x30 px area)
            if pos.x() >= self.width() - 30 and pos.y() >= self.height() - 30:
                self._resizing = True
                self._drag_start_pos = event.globalPosition().toPoint()
                self._drag_start_size = self.size()
            else:
                self._resizing = False
                self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        if getattr(self, '_resizing', False) and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            new_width = max(350, self._drag_start_size.width() + delta.x())
            new_height = max(200, self._drag_start_size.height() + delta.y())
            self.resize(new_width, new_height)
        elif getattr(self, '_drag_pos', None) and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        self._resizing = False

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._change_font_size(1)
            elif delta < 0:
                self._change_font_size(-1)
            event.accept()
        else:
            super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _set_token_info(self, info: str) -> None:
        value = (info or "").strip()
        self._token_label.setToolTip(value)
        # Don't display API info text — only keep as tooltip
        self._token_label.setText("")

    def _build_html_style(self) -> str:
        fs = self._font_size
        code_fs = fs - 1
        return f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            body {{
                font-family: 'Inter', "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
                font-size: {fs}px;
                color: rgba(255, 255, 255, 0.95);
                line-height: 1.65;
                padding: 6px;
                font-weight: 500;
            }}
            code {{
                background-color: rgba(0,0,0,0.4);
                border-radius: 4px;
                padding: 3px 6px;
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: {code_fs}px;
                color: #ff7b72;
                font-weight: 500;
            }}
            pre {{
                background-color: rgba(16,16,20,0.95);
                border-radius: 8px;
                padding: 12px;
                border: 1px solid rgba(255,255,255,0.15);
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
                color: inherit;
                font-weight: normal;
            }}
            {self._pygments_css}
        </style>
        """

    def _change_font_size(self, delta: float, *, reset: bool = False) -> None:
        if reset:
            self._font_size = self._DEFAULT_FONT_SIZE
        else:
            self._font_size = max(10, min(24, self._font_size + delta))
        # Re-render current content with new font size
        if hasattr(self, '_raw_markdown'):
            self._render_markdown(self._raw_markdown)

    def _render_markdown(self, text: str) -> None:
        self._raw_markdown = text
        html_content = markdown.markdown(
            self._raw_markdown, extensions=['fenced_code', 'codehilite', 'tables']
        )
        style = self._build_html_style()
        full_html = f"<html><head>{style}</head><body>{html_content}</body></html>"
        self._response_area.setHtml(full_html)

    def _update_history_nav(self) -> None:
        total = len(self._response_history)
        if total <= 0 or self._history_index < 0:
            self._history_label.setText("0/0")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return

        self._history_label.setText(f"{self._history_index + 1}/{total}")
        self._prev_btn.setEnabled(self._history_index > 0)
        self._next_btn.setEnabled(self._history_index < (total - 1))

    def _show_history_entry(self, index: int) -> None:
        if index < 0 or index >= len(self._response_history):
            return

        self._history_index = index
        entry = self._response_history[index]
        self._set_token_info(entry.get("info", ""))
        self._render_markdown(entry.get("text", ""))
        self._update_history_nav()

        scrollbar = self._response_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.minimum())

    def _store_response(self, text: str, info: str = "") -> None:
        entry = {"text": text, "info": info}
        if self._response_history and self._response_history[-1] == entry:
            self._show_history_entry(len(self._response_history) - 1)
            return

        self._response_history.append(entry)
        if len(self._response_history) > 25:
            self._response_history = self._response_history[-25:]
        self._show_history_entry(len(self._response_history) - 1)

    def _show_previous_response(self) -> None:
        if self._history_index > 0:
            self._show_history_entry(self._history_index - 1)

    def _show_next_response(self) -> None:
        if self._history_index < (len(self._response_history) - 1):
            self._show_history_entry(self._history_index + 1)

    def set_loading(self, loading: bool) -> None:
        if loading:
            self._mode_dot.setStyleSheet("color: #ffaa00; font-size: 10px;")
            self._status_label.setText("⏳ Analizuję… (kliknij 'Anuluj' by przerwać)")
            self._voice_btn.setText("❌ Anuluj")
            self._solve_btn.setEnabled(False)
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
        else:
            self._status_label.setText("")
            self._voice_btn.setText("🎙️ Start nasłuchiwanie")
            self._solve_btn.setEnabled(True)
            self._update_history_nav()

    def set_response(self, text: str, info: str = "") -> None:
        self._status_label.setText("")
        self._store_response(text, info)

    # ---- Language override ----

    def _on_lang_changed(self, index: int) -> None:
        mapping = {0: "auto", 1: "pl", 2: "en"}
        lang = mapping.get(index, "auto")
        print(f"[GUI] Język odpowiedzi: {lang}")
        self.language_override_changed.emit(lang)

    def get_language_override(self) -> str | None:
        """Return forced language code or None for auto-detect."""
        mapping = {0: None, 1: "pl", 2: "en"}
        return mapping.get(self.lang_combo.currentIndex(), None)

    # ---- Model selector ----

    def _on_model_changed(self, model_name: str) -> None:
        print(f"[GUI] Model LLM: {model_name}")
        self.model_changed.emit(model_name)

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
        compact = " ".join(part.strip() for part in text.splitlines() if part.strip())
        if len(compact) > 140:
            compact = compact[:137] + "..."
        self._status_label.setText(compact)
        self._render_markdown(text)

        scrollbar = self._response_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.minimum())
