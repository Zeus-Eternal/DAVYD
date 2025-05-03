#!/usr/bin/env python3
# src/theme_manager.py

from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt

class ThemeManager:
    """
    Manages application theming with dark/light modes and custom palettes.
    Features:
    - Full Fusion style integration
    - Complete palette control
    - Style sheet extensions
    - Runtime theme switching
    """
    
    # Common styles that persist across themes
    COMMON_STYLE = """
        QToolTip {
            background-color: palette(tooltip);
            color: palette(tooltip-text);
            border: 1px solid palette(mid);
            padding: 5px;
            border-radius: 3px;
        }
        QMenuBar::item:selected {
            background-color: palette(highlight);
            color: palette(highlighted-text);
        }
    """
    
    # Dark theme colors
    DARK_PALETTE = {
        "window":           (53, 53, 53),
        "window-text":      (255, 255, 255),
        "base":             (35, 35, 35),
        "alternate-base":   (53, 53, 53),
        "tooltip":          (51, 51, 51),
        "tooltip-text":     (255, 255, 255),
        "text":             (255, 255, 255),
        "button":           (53, 53, 53),
        "button-text":      (255, 255, 255),
        "bright-text":      (255, 0, 0),
        "highlight":        (142, 45, 197),
        "highlight-text":   (0, 0, 0),
        "disabled-text":    (127, 127, 127)
    }
    
    # Light theme colors
    LIGHT_PALETTE = {
        "window":           (240, 240, 240),
        "window-text":      (0, 0, 0),
        "base":             (255, 255, 255),
        "alternate-base":   (233, 233, 235),
        "tooltip":          (255, 255, 220),
        "tooltip-text":     (0, 0, 0),
        "text":             (0, 0, 0),
        "button":           (240, 240, 240),
        "button-text":      (0, 0, 0),
        "bright-text":      (255, 0, 0),
        "highlight":        (100, 149, 237),
        "highlight-text":   (255, 255, 255)
    }

    def __init__(self, app: QtWidgets.QApplication, dark_mode: bool = True):
        """
        Initialize theme manager with application reference and initial mode.
        
        Args:
            app: QApplication instance to style
            dark_mode: Whether to start in dark mode (default: True)
        """
        self.app = app
        self._dark_mode = dark_mode
        self._current_palette = QtGui.QPalette()
        
        # Configure base style
        self.app.setStyle("Fusion")
        self.app.setStyleSheet(self.COMMON_STYLE)
        self.apply_theme()

    @property
    def dark_mode(self) -> bool:
        """Get current theme mode (dark/light)."""
        return self._dark_mode

    @dark_mode.setter
    def dark_mode(self, enabled: bool):
        """Toggle between dark/light themes."""
        self._dark_mode = enabled
        self.apply_theme()

    def apply_theme(self):
        """Apply the current theme palette to the application."""
        palette_config = self.DARK_PALETTE if self.dark_mode else self.LIGHT_PALETTE
        
        # Base colors
        self._current_palette.setColor(QtGui.QPalette.Window, 
            QtGui.QColor(*palette_config["window"]))
        self._current_palette.setColor(QtGui.QPalette.WindowText, 
            QtGui.QColor(*palette_config["window-text"]))
        self._current_palette.setColor(QtGui.QPalette.Base, 
            QtGui.QColor(*palette_config["base"]))
        self._current_palette.setColor(QtGui.QPalette.AlternateBase, 
            QtGui.QColor(*palette_config["alternate-base"]))
            
        # Text colors
        self._current_palette.setColor(QtGui.QPalette.Text, 
            QtGui.QColor(*palette_config["text"]))
        self._current_palette.setColor(QtGui.QPalette.BrightText, 
            QtGui.QColor(*palette_config["bright-text"]))
            
        # Button colors
        self._current_palette.setColor(QtGui.QPalette.Button, 
            QtGui.QColor(*palette_config["button"]))
        self._current_palette.setColor(QtGui.QPalette.ButtonText, 
            QtGui.QColor(*palette_config["button-text"]))
            
        # Interactive elements
        self._current_palette.setColor(QtGui.QPalette.Highlight, 
            QtGui.QColor(*palette_config["highlight"]))
        self._current_palette.setColor(QtGui.QPalette.HighlightedText, 
            QtGui.QColor(*palette_config["highlight-text"]))
            
        # Tooltips
        self._current_palette.setColor(QtGui.QPalette.ToolTipBase, 
            QtGui.QColor(*palette_config["tooltip"]))
        self._current_palette.setColor(QtGui.QPalette.ToolTipText, 
            QtGui.QColor(*palette_config["tooltip-text"]))
            
        # Disabled state handling
        if self.dark_mode:
            self._current_palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, 
                QtGui.QColor(*palette_config["disabled-text"]))
        
        self.app.setPalette(self._current_palette)

    def add_style(self, additional_style: str):
        """
        Add additional style rules while preserving existing styles.
        
        Args:
            additional_style: CSS-style string to append to current styles
        """
        current_style = self.app.styleSheet()
        self.app.setStyleSheet(current_style + additional_style)
