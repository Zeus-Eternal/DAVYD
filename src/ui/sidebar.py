#!/usr/bin/env python3
# src/sidebar.py

import webbrowser
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt

DOCS_URL = "https://github.com/yourusername/davyd/docs"

class Sidebar(QtWidgets.QFrame):
    """
    A self-contained sidebar panel with:
    - Model configuration (provider, API key, model, test/save/reset)
    - Quick actions (switching tabs)
    
    Exposes widgets:
    - provider_combo, api_key_input, model_combo
    - test_btn, save_btn, reset_btn
    - quick_buttons (dict of QPushButton)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.quick_buttons = {}

    def _setup_ui(self):
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFixedWidth(350)
        self.setStyleSheet("""
            QFrame { 
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #2c3e50, stop:1 #34495e);
                border-radius:8px; 
                border:1px solid #444; 
            }
            QGroupBox { 
                color:white; 
                font-weight:bold; 
                border:1px solid #555;
                border-radius:5px; 
                margin-top:10px; 
            }
            QGroupBox::title { 
                subcontrol-origin:margin; 
                left:10px;
                padding:0 3px; 
                color:#ecf0f1; 
            }
            QLabel { 
                color:#ecf0f1; 
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        self._add_header(layout)
        self._add_model_config(layout)
        self._add_quick_actions(layout)
        layout.addStretch()

    def _add_header(self, layout):
        hdr = QtWidgets.QLabel("DAVYD")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("""
            font-size:24px; 
            font-weight:bold; 
            color:#3498db;
            padding:10px; 
            border-bottom:1px solid #555; 
            margin-bottom:15px;
        """)
        layout.addWidget(hdr)

    def _add_model_config(self, layout):
        grp = QtWidgets.QGroupBox("AI Model Configuration")
        gl = QtWidgets.QVBoxLayout(grp)
        gl.setSpacing(8)
        
        gl.addWidget(QtWidgets.QLabel("Provider:"))
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems([
            "Ollama", "DeepSeek", "Gemini", "ChatGPT",
            "Anthropic", "Claude", "Mistral", "Groq", "HuggingFace"
        ])
        self.provider_combo.setStyleSheet("""
            QComboBox { 
                background:#2c3e50; color:white; border:1px solid #555;
                padding:5px; border-radius:4px; 
            }
            QComboBox::drop-down { border:0; }
        """)
        gl.addWidget(self.provider_combo)

        self.api_key_label = QtWidgets.QLabel("API Key:")
        self.api_key_input = QtWidgets.QLineEdit()
        self.api_key_input.setPlaceholderText("Enter API key or URL")
        self.api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_input.setStyleSheet("""
            QLineEdit { 
                background:#2c3e50; color:white; border:1px solid #555;
                padding:5px; border-radius:4px; 
            }
        """)
        gl.addWidget(self.api_key_label)
        gl.addWidget(self.api_key_input)

        gl.addWidget(QtWidgets.QLabel("Model:"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setStyleSheet(self.provider_combo.styleSheet())
        gl.addWidget(self.model_combo)

        test_btn = QtWidgets.QPushButton("Test Connection")
        test_btn.setStyleSheet("""
            QPushButton { 
                background:#3498db; color:white; border:none;
                padding:8px; border-radius:4px; 
            }
            QPushButton:hover { background:#2980b9; }
        """)
        self.test_btn = test_btn

        btns = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setStyleSheet("""
            QPushButton { 
                background:#27ae60; color:white; border:none;
                padding:8px; border-radius:4px; 
            }
            QPushButton:hover { background:#2ecc71; }
        """)
        self.save_btn = save_btn

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setStyleSheet("""
            QPushButton { 
                background:#7f8c8d; color:white; border:none;
                padding:8px; border-radius:4px; 
            }
            QPushButton:hover { background:#95a5a6; }
        """)
        self.reset_btn = reset_btn

        btns.addWidget(test_btn)
        btns.addWidget(save_btn)
        btns.addWidget(reset_btn)
        gl.addLayout(btns)
        layout.addWidget(grp)

    def _add_quick_actions(self, layout):
        qa = QtWidgets.QGroupBox("Quick Actions")
        qal = QtWidgets.QVBoxLayout(qa)
        
        actions = [
            ("Generate Dataset", "#e74c3c", lambda: self.parent().tabs.setCurrentIndex(1)),
            ("View Datasets", "#3498db", lambda: self.parent().tabs.setCurrentIndex(3)),
            ("Edit Fields", "#9b59b6", lambda: self.parent().tabs.setCurrentIndex(0)),
            ("Visualize Data", "#2ecc71", lambda: self.parent().tabs.setCurrentIndex(2)),
            ("Documentation", "#f39c12", lambda: webbrowser.open(DOCS_URL))
        ]
        
        self.quick_buttons = {}
        for txt, col, slot in actions:
            b = QtWidgets.QPushButton(txt)
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color:{col}; 
                    color:white; 
                    border:none;
                    padding:10px; 
                    border-radius:4px; 
                    text-align:left;
                    padding-left:15px;
                }}
                QPushButton:hover {{ background-color:{self._lighten_color(col)}; }}
            """)
            b.setCursor(QtGui.QCursor(Qt.PointingHandCursor))
            b.clicked.connect(slot)
            qal.addWidget(b)
            self.quick_buttons[txt] = b
        
        layout.addWidget(qa)

    def _lighten_color(self, hex_color, amount=20):
        color = QtGui.QColor(hex_color)
        return color.lighter(100 + amount).name()
