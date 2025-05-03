#!/usr/bin/env python3
# src/ui_desktop.py

import sys
import os
import json
import shutil
import logging
import webbrowser
from pathlib import Path

import pandas as pd
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from davyd import Davyd
from utils.manage_dataset import DatasetManager
from utils.main_utils import SETTINGS_FILE, TEMP_DIR, ARCHIVE_DIR, MERGED_DIR, DOCS_URL
from model_providers import (
    OllamaClient, DeepSeekClient, GeminiClient,
    ChatGPTClient, AnthropicClient, ClaudeClient,
    MistralClient, GroqClient, HuggingFaceClient
)
from dataset_generation import DatasetGenerationThread
from dataset_model import DatasetModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThemeManager:
    """
    Fusion style + dark/light palette toggle.
    """
    COMMON_STYLE = """
        QToolTip { background-color:#333; color:white; border:1px solid #555;
                   padding:5px; border-radius:3px; }
        QMenuBar::item:selected { background-color:#555; }
    """

    def __init__(self, app: QtWidgets.QApplication, dark_mode: bool):
        self.app = app
        self.dark_mode = dark_mode

    def apply(self):
        self.app.setStyle("Fusion")
        self.app.setStyleSheet(self.COMMON_STYLE)
        p = QtGui.QPalette()
        if self.dark_mode:
            p.setColor(QtGui.QPalette.Window,           QtGui.QColor(53,53,53))
            p.setColor(QtGui.QPalette.WindowText,       QtCore.Qt.white)
            p.setColor(QtGui.QPalette.Base,             QtGui.QColor(35,35,35))
            p.setColor(QtGui.QPalette.AlternateBase,    QtGui.QColor(53,53,53))
            p.setColor(QtGui.QPalette.ToolTipBase,      QtCore.Qt.white)
            p.setColor(QtGui.QPalette.ToolTipText,      QtCore.Qt.white)
            p.setColor(QtGui.QPalette.Text,             QtCore.Qt.white)
            p.setColor(QtGui.QPalette.Button,           QtGui.QColor(53,53,53))
            p.setColor(QtGui.QPalette.ButtonText,       QtCore.Qt.white)
            p.setColor(QtGui.QPalette.Highlight,        QtGui.QColor(142,45,197).lighter())
            p.setColor(QtGui.QPalette.HighlightedText,  QtCore.Qt.black)

        self.app.setPalette(p)

    def toggle(self, dark_mode: bool):
        self.dark_mode = dark_mode
        self.apply()


class Sidebar(QtWidgets.QFrame):
    """Left‐hand sidebar for provider selection and dataset lists."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFixedWidth(300)
        self._build()

    def _build(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10,10,10,10)
        v.setSpacing(15)

        v.addWidget(QtWidgets.QLabel("Model Provider:"))
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems([
            "", "Ollama","DeepSeek","Gemini","ChatGPT",
            "Anthropic","Claude","Mistral","Groq","HuggingFace"
        ])
        self.provider_combo.currentTextChanged.connect(
            self.main_window.update_provider_ui
        )
        v.addWidget(self.provider_combo)

        v.addWidget(QtWidgets.QLabel("API Key / URL:"))
        self.api_key_input = QtWidgets.QLineEdit()
        v.addWidget(self.api_key_input)

        v.addWidget(QtWidgets.QLabel("Model:"))
        self.model_combo = QtWidgets.QComboBox()
        v.addWidget(self.model_combo)

        btn = QtWidgets.QPushButton("Test Connection")
        btn.clicked.connect(self.main_window.test_connection)
        v.addWidget(btn)

        v.addSpacing(20)

        for title, attr in [
            ("Active Datasets", "active_list"),
            ("Archived Datasets", "archive_list"),
            ("Merged Datasets", "merged_list")
        ]:
            v.addWidget(QtWidgets.QLabel(title))
            lw = QtWidgets.QListWidget()
            setattr(self, attr, lw)
            v.addWidget(lw)

        v.addStretch()


class TabsManager:
    """Four tabs: Structure, Generation, Visualization, Management."""
    def __init__(self, parent, tabs: QtWidgets.QTabWidget):
        self.parent = parent
        self.tabs   = tabs
        self._style()
        self._add_structure_tab()
        self._add_generation_tab()
        self._add_visualization_tab()
        self._add_management_tab()

    def _style(self):
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border:1px solid #444; border-radius:4px; margin-top:-1px; }
            QTabBar::tab { background:#34495e; color:#ecf0f1; padding:8px 16px;
                           border:1px solid #444; margin-right:2px;
                           border-top-left-radius:4px; border-top-right-radius:4px; }
            QTabBar::tab:selected { background:#2c3e50; border-bottom-color:#3498db; }
            QTabBar::tab:hover { background:#3d566e; }
        """)

    def _add_structure_tab(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        l.setContentsMargins(15,15,15,15)
        l.setSpacing(10)

        grp = QtWidgets.QGroupBox("Dataset Fields")
        gl  = QtWidgets.QVBoxLayout(grp)
        self.parent.fields_list = QtWidgets.QListWidget()
        gl.addWidget(self.parent.fields_list)
        hl = QtWidgets.QHBoxLayout()
        for txt, slot in [
            ("Add Field",    self.parent.add_field_dialog),
            ("Remove Field", self.parent.remove_selected_field),
            ("Edit Field",   self.parent.edit_selected_field),
        ]:
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(slot)
            hl.addWidget(b)
        gl.addLayout(hl)
        l.addWidget(grp)

        fmt = QtWidgets.QFormLayout()
        self.parent.separator_combo = QtWidgets.QComboBox()
        self.parent.separator_combo.addItems(["|",":","-","~","•","→"])
        fmt.addRow("Field Separator:", self.parent.separator_combo)

        self.parent.wrapper_combo = QtWidgets.QComboBox()
        self.parent.wrapper_combo.addItems(['"',"'","`","«»","''","``"])
        fmt.addRow("Data Wrapper:", self.parent.wrapper_combo)

        up = QtWidgets.QPushButton("Update Preview")
        up.clicked.connect(self.parent.update_format_preview)
        fmt.addRow(up)

        self.parent.format_preview = QtWidgets.QLabel()
        fmt.addRow("Preview:", self.parent.format_preview)

        l.addLayout(fmt)
        self.tabs.addTab(w, "📝 Structure")

    def _add_generation_tab(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        l.setContentsMargins(15,15,15,15)
        l.setSpacing(10)

        grp = QtWidgets.QGroupBox("Generation Parameters")
        fl  = QtWidgets.QFormLayout(grp)
        self.parent.num_entries = QtWidgets.QSpinBox()
        self.parent.num_entries.setRange(1, 10000)
        fl.addRow("Entries:", self.parent.num_entries)

        self.parent.quality_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.parent.quality_slider.setRange(1,5)
        fl.addRow("Quality:", self.parent.quality_slider)

        self.parent.advanced_options = QtWidgets.QCheckBox("Show Advanced Options")
        self.parent.advanced_options.toggled.connect(self.parent.toggle_advanced_options)
        fl.addRow(self.parent.advanced_options)

        self.parent.advanced_group = QtWidgets.QGroupBox("Advanced")
        self.parent.advanced_group.setVisible(False)
        agl = QtWidgets.QFormLayout(self.parent.advanced_group)
        self.parent.temperature = QtWidgets.QDoubleSpinBox()
        agl.addRow("Temperature:", self.parent.temperature)
        self.parent.max_tokens = QtWidgets.QSpinBox()
        agl.addRow("Max Tokens:", self.parent.max_tokens)
        fl.addRow(self.parent.advanced_group)

        l.addWidget(grp)
        self.parent.progress_bar = QtWidgets.QProgressBar()
        l.addWidget(self.parent.progress_bar)

        gen = QtWidgets.QPushButton("✨ Generate Dataset")
        gen.clicked.connect(self.parent.start_generation)
        l.addWidget(gen)

        self.parent.generation_status = QtWidgets.QLabel()
        l.addWidget(self.parent.generation_status)
        l.addStretch()
        self.tabs.addTab(w, "⚡ Generation")

    def _add_visualization_tab(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        l.setContentsMargins(15,15,15,15)
        l.setSpacing(10)

        hl = QtWidgets.QHBoxLayout()
        self.parent.search_field = QtWidgets.QLineEdit()
        self.parent.search_field.setPlaceholderText("Search…")
        self.parent.search_field.textChanged.connect(self.parent.filter_dataset_view)
        hl.addWidget(self.parent.search_field)
        clr = QtWidgets.QPushButton("Clear")
        clr.clicked.connect(self.parent.search_field.clear)
        hl.addWidget(clr)
        self.parent.row_count_label = QtWidgets.QLabel("0 rows")
        hl.addWidget(self.parent.row_count_label)
        l.addLayout(hl)

        self.parent.dataset_model = DatasetModel(pd.DataFrame())
        tv = QtWidgets.QTableView()
        tv.setModel(self.parent.dataset_model)
        l.addWidget(tv)

        cg = QtWidgets.QGroupBox("Data Visualization")
        vz = QtWidgets.QVBoxLayout(cg)
        self.parent.viz_type = QtWidgets.QComboBox()
        self.parent.viz_type.addItems(["Bar Chart","Pie Chart","Scatter Plot","Histogram"])
        self.parent.viz_type.currentTextChanged.connect(self.parent.update_visualization)
        vz.addWidget(self.parent.viz_type)
        self.parent.figure = Figure()
        self.parent.canvas = FigureCanvas(self.parent.figure)
        vz.addWidget(self.parent.canvas)
        l.addWidget(cg)

        l.addStretch()
        self.tabs.addTab(w, "📊 Visualization")

    def _add_management_tab(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        l.setContentsMargins(15,15,15,15)
        l.setSpacing(10)

        tb = QtWidgets.QTabWidget()
        tb.addTab(self.parent.sidebar.active_list,  "Active")
        tb.addTab(self.parent.sidebar.archive_list, "Archived")
        tb.addTab(self.parent.sidebar.merged_list,  "Merged")
        l.addWidget(tb)

        hl = QtWidgets.QHBoxLayout()
        for txt, slot in [
            ("Download", self.parent.download_dataset),
            ("Archive",  self.parent.archive_dataset),
            ("Restore",  self.parent.restore_dataset),
            ("Delete",   self.parent.delete_dataset),
        ]:
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(slot)
            hl.addWidget(b)
        l.addLayout(hl)

        mg = QtWidgets.QGroupBox("Merge Datasets")
        mgl = QtWidgets.QVBoxLayout(mg)
        self.parent.merge_list = QtWidgets.QListWidget()
        mgl.addWidget(self.parent.merge_list)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Output filename:"))
        self.parent.merge_name = QtWidgets.QLineEdit("merged_dataset.csv")
        row.addWidget(self.parent.merge_name)
        mgl.addLayout(row)
        mb = QtWidgets.QPushButton("Merge Selected")
        mb.clicked.connect(self.parent.merge_datasets)
        mgl.addWidget(mb)
        l.addWidget(mg)

        l.addStretch()
        self.tabs.addTab(w, "🗂 Management")


class MenuBar:
    """File/Edit/View/Help menus."""
    def __init__(self, window: QtWidgets.QMainWindow):
        mb = window.menuBar()

        # File
        fm = mb.addMenu("&File")
        fm.addAction("New Project",    window.new_project,          shortcut="Ctrl+N")
        fm.addAction("Open Dataset…",  window.open_dataset,         shortcut="Ctrl+O")
        fm.addAction("Save Dataset…",  window.save_current_dataset, shortcut="Ctrl+S")
        exp = fm.addMenu("Export…")
        exp.addAction("As CSV",   lambda: window.export_dataset('csv'))
        exp.addAction("As JSON",  lambda: window.export_dataset('json'))
        exp.addAction("As Excel", lambda: window.export_dataset('excel'))
        fm.addSeparator()
        fm.addAction("E&xit", window.close, shortcut="Ctrl+Q")

        # Edit
        em = mb.addMenu("&Edit")
        em.addAction("Undo", window.undo, shortcut="Ctrl+Z")
        em.addAction("Redo", window.redo, shortcut="Ctrl+Y")
        em.addSeparator()
        em.addAction("Preferences…", window.show_preferences)

        # View
        vm = mb.addMenu("&View")
        dm = QtGui.QAction("Dark Mode", window, checkable=True)
        dm.setChecked(window.theme.dark_mode)
        dm.toggled.connect(window.toggle_dark_mode)
        vm.addAction(dm)
        vm.addAction("Zoom In",  window.zoom_in,  shortcut="Ctrl++")
        vm.addAction("Zoom Out", window.zoom_out, shortcut="Ctrl+-")

        # Help
        hm = mb.addMenu("&Help")
        hm.addAction("Documentation", lambda: webbrowser.open(DOCS_URL))
        hm.addAction("About…", window.show_about)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔥 DAVYD - Dataset Generator")
        self.resize(1600, 1000)

        # ── State & Data Managers
        self.state = {
            'fields': ["text","intent","sentiment","sentiment_polarity","tone","category","keywords"],
            'examples': ['"Hi there!"','"greeting"','"positive"','0.9','"friendly"','"interaction"','"hi" "hello" "welcome"'],
            'data_separator': '"',
            'section_separator': '|',
            'model_provider': None,
            'dark_mode': True
        }
        self.dataset_manager = DatasetManager(TEMP_DIR, ARCHIVE_DIR, MERGED_DIR)
        self.current_dataset = pd.DataFrame()
        self.generation_thread = None

        # ── Central & Layout
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        hl = QtWidgets.QHBoxLayout(cw)
        hl.setContentsMargins(10,10,10,10); hl.setSpacing(15)

        # ── Theme
        self.theme = ThemeManager(QtWidgets.QApplication.instance(), self.state['dark_mode'])
        self.theme.apply()

        # ── Sidebar
        self.sidebar = Sidebar(self)
        hl.addWidget(self.sidebar)

        # ── Tabs
        self.tabs = QtWidgets.QTabWidget()
        TabsManager(self, self.tabs)
        hl.addWidget(self.tabs, stretch=1)

        # ── Menu & Status
        MenuBar(self)
        sb = self.statusBar()
        sb.showMessage("Ready")
        self.memory_label = QtWidgets.QLabel()
        sb.addPermanentWidget(self.memory_label)
        QtCore.QTimer(self, timeout=self._update_memory_usage, interval=5000).start()

        # ── Load Settings & Populate
        self.load_settings()
        self.refresh_dataset_lists()


    # ─── Menu Slots ────────────────────────────────────────────────────
    def new_project(self):
        self.current_dataset = pd.DataFrame()
        self.dataset_manager.clear_temp()
        self.refresh_dataset_lists()
        QtWidgets.QMessageBox.information(self,"New Project","New project created.")
        self.statusBar().showMessage("New project created",3000)

    def undo(self):
        QtWidgets.QMessageBox.information(self,"Undo","Undo not implemented yet.")

    def redo(self):
        QtWidgets.QMessageBox.information(self,"Redo","Redo not implemented yet.")

    def show_preferences(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Preferences")
        ly = QtWidgets.QVBoxLayout(dlg)
        cb = QtWidgets.QCheckBox("Dark Mode")
        cb.setChecked(self.theme.dark_mode)
        cb.toggled.connect(self.toggle_dark_mode)
        ly.addWidget(cb)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        ly.addWidget(bb)
        dlg.exec()

    def toggle_dark_mode(self, checked):
        """Switch between dark/light theme and persist."""
        self.state['dark_mode'] = checked
        if checked:
            self.set_dark_theme()
        else:
            self.set_light_theme()
        self.save_settings()

    def show_about(self):
        QtWidgets.QMessageBox.information(self,"About DAVYD",
            "DAVYD Dataset Generator\nVersion 1.0\nPowered by your local LLM setup."
        )


    # ─── Sidebar / Provider Slots ─────────────────────────────────────
    def update_provider_ui(self, provider: str):
        self.state['model_provider'] = provider
        combo = self.sidebar.model_combo
        combo.clear()
        inp   = self.sidebar.api_key_input
        inp.clear()
        inp.setEchoMode(QtWidgets.QLineEdit.Password)
        mapping = {
            "Ollama":      (QtWidgets.QLineEdit.Normal,   "http://localhost:11434", ["llama2","mistral","gemma","phi"]),
            "DeepSeek":    (QtWidgets.QLineEdit.Password, "DeepSeek API key",      ["deepseek-chat"]),
            "Gemini":      (QtWidgets.QLineEdit.Password, "Gemini API key",        ["gemini-pro"]),
            "ChatGPT":     (QtWidgets.QLineEdit.Password, "OpenAI API key",        ["gpt-3.5-turbo","gpt-4"]),
            "Anthropic":   (QtWidgets.QLineEdit.Password, "Anthropic API key",     ["claude-2","claude-instant"]),
            "Claude":      (QtWidgets.QLineEdit.Password, "Claude API key",        ["claude-v1"]),
            "Mistral":     (QtWidgets.QLineEdit.Password, "Mistral API key",       ["mistral-tiny","mistral-small","mistral-medium"]),
            "Groq":        (QtWidgets.QLineEdit.Password, "Groq API key",          ["mixtral-8x7b-32768","llama2-70b-4096"]),
            "HuggingFace": (QtWidgets.QLineEdit.Password, "HF API key",            ["meta-llama/Llama-2-70b-chat-hf"]),
        }
        if provider in mapping:
            mode, ph, models = mapping[provider]
            inp.setEchoMode(mode)
            inp.setPlaceholderText(ph)
            combo.addItems(models)

    def test_connection(self):
        prov = self.sidebar.provider_combo.currentText()
        key  = self.sidebar.api_key_input.text().strip()
        mdl  = self.sidebar.model_combo.currentText()
        if not (prov and key and mdl):
            QtWidgets.QMessageBox.warning(self,"Warning","Select provider, model, and enter key")
            return
        try:
            client = {
                "Ollama":      OllamaClient(host=key),
                "DeepSeek":    DeepSeekClient(api_key=key),
                "Gemini":      GeminiClient(api_key=key),
                "ChatGPT":     ChatGPTClient(api_key=key),
                "Anthropic":   AnthropicClient(api_key=key),
                "Claude":      ClaudeClient(api_key=key),
                "Mistral":     MistralClient(api_key=key),
                "Groq":        GroqClient(api_key=key),
                "HuggingFace": HuggingFaceClient(api_key=key),
            }[prov]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Error",f"Client init failed:\n{e}")
            return
        try:
            ok = client.test_connection() if hasattr(client,"test_connection") else bool(client.list_models())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Error",f"Connection failed:\n{e}")
            return
        QtWidgets.QMessageBox.information(self,"Success",f"Connected to {prov}!" if ok else f"Failed to connect to {prov}")


    # ─── Generation Slots ──────────────────────────────────────────────
    def toggle_advanced_options(self, checked: bool):
        self.advanced_group.setVisible(checked)

    def start_generation(self):
        prov = self.sidebar.provider_combo.currentText()
        key  = self.sidebar.api_key_input.text().strip()
        mdl  = self.sidebar.model_combo.currentText()
        if not (prov and key and mdl):
            QtWidgets.QMessageBox.warning(self,"Warning","Select provider, model, and enter key")
            return
        if self.generation_thread and self.generation_thread.isRunning():
            QtWidgets.QMessageBox.warning(self,"Warning","Generation already running")
            return
        client = {
            "Ollama":      OllamaClient(host=key),
            "DeepSeek":    DeepSeekClient(api_key=key),
            "Gemini":      GeminiClient(api_key=key
),            "ChatGPT":     ChatGPTClient(api_key=key),
            "Anthropic":   AnthropicClient(api_key=key),
            "Claude":      ClaudeClient(api_key=key),
            "Mistral":     MistralClient(api_key=key),
            "Groq":        GroqClient(api_key=key),
            "HuggingFace": HuggingFaceClient(api_key=key),
        }[prov]
        cfg = {
            'num_entries': self.num_entries.value(),
            'model_client': client,
            'model_name': mdl,
            'dataset_manager': self.dataset_manager,
            'section_separator': self.separator_combo.currentText(),
            'data_separator': self.wrapper_combo.currentText(),
            'fields': self.state['fields'],
            'examples': self.state['examples'],
            'temperature': self.temperature.value(),
            'max_tokens': self.max_tokens.value(),
        }
        self.generation_thread = DatasetGenerationThread()
        self.generation_thread.progress_updated.connect(self.update_progress)
        self.generation_thread.generation_complete.connect(self.generation_complete)
        self.generation_thread.error_occurred.connect(self.generation_error)
        self.generation_thread.start()
        self.progress_bar.setValue(0)
        self.generation_status.setText("Generation started…")

    def update_progress(self, v, m):
        self.progress_bar.setValue(v)
        self.generation_status.setText(m)
        
    def generation_complete(self, df: pd.DataFrame):
        self.current_dataset = df
        self.progress_bar.setValue(100)
        self.generation_status.setText("✅ Done")
        self.tabs.setCurrentIndex(2)
        self.dataset_model.update_data(df)
        self.update_row_count()
        self.refresh_dataset_lists()
        QtWidgets.QMessageBox.information(self, "Success", f"Generated {len(df)} entries")
        self.progress_bar.setValue(0)

    def generation_error(self, err: Exception):
        self.progress_bar.setValue(0)
        self.generation_status.setText("❌ Failed")
        QtWidgets.QMessageBox.critical(self,"Error",str(err))


    # ─── Visualization Slots ──────────────────────────────────────────
    def filter_dataset_view(self, text: str):
        self.dataset_model.filter_data(text)

    def update_visualization(self):
        self.figure.clear()
        df = self.current_dataset if not self.current_dataset.empty else self.dataset_model._original_data
        ax = self.figure.add_subplot(111)
        t = self.viz_type.currentText()
        try:
            if t=="Bar Chart":
                nums = df.select_dtypes(include='number')
                if nums.empty: raise ValueError("No numeric columns")
                nums.sum().plot.bar(ax=ax)
            elif t=="Pie Chart":
                if df.empty: raise ValueError("No data")
                df.iloc[:,0].value_counts().plot.pie(ax=ax,autopct="%1.1f%%")
            elif t=="Scatter Plot":
                nums = df.select_dtypes(include='number')
                if nums.shape[1]<2: raise ValueError("Need ≥2 numeric cols")
                ax.scatter(nums.iloc[:,0], nums.iloc[:,1])
                ax.set_xlabel(nums.columns[0]); ax.set_ylabel(nums.columns[1])
            elif t=="Histogram":
                nums = df.select_dtypes(include='number')
                if nums.empty: raise ValueError("No numeric columns")
                nums.plot.hist(ax=ax, bins=20)
            else:
                ax.text(0.5,0.5,"Unknown chart",ha='center')
        except Exception as e:
            ax.clear()
            ax.text(0.5,0.5,f"Error:\n{e}",ha='center',color='red')
        finally:
            self.canvas.draw()

    def update_row_count(self):
        cnt = self.dataset_model.rowCount()
        self.row_count_label.setText(f"{cnt} rows")


    # ─── Management Slots ─────────────────────────────────────────────
    def download_dataset(self):
        for it in self.sidebar.active_list.selectedItems():
            src = os.path.join(TEMP_DIR, it.text())
            dst, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save As",it.text())
            if dst:
                shutil.copy(src, dst)
        self.statusBar().showMessage("Download complete",3000)

    def archive_dataset(self):
        for it in self.sidebar.active_list.selectedItems():
            self.dataset_manager.archive_dataset(it.text())
        self.refresh_dataset_lists()
        QtWidgets.QMessageBox.information(self,"Archived","Selected datasets archived.")

    def restore_dataset(self):
        for it in self.sidebar.archive_list.selectedItems():
            self.dataset_manager.restore_dataset(it.text())
        self.refresh_dataset_lists()
        QtWidgets.QMessageBox.information(self,"Restored","Selected datasets restored.")

    def delete_dataset(self):
        ans = QtWidgets.QMessageBox.question(self,"Confirm Delete",
            "Delete selected datasets? This cannot be undone.",
            QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No
        )
        if ans!=QtWidgets.QMessageBox.Yes:
            return
        for lw,path in [
            (self.sidebar.active_list, TEMP_DIR),
            (self.sidebar.archive_list, ARCHIVE_DIR),
            (self.sidebar.merged_list, MERGED_DIR),
        ]:
            for it in lw.selectedItems():
                try: os.remove(os.path.join(path,it.text()))
                except: pass
        self.refresh_dataset_lists()
        QtWidgets.QMessageBox.information(self,"Deleted","Selected datasets deleted.")

    def merge_datasets(self):
        names = [it.text() for it in self.merge_list.selectedItems()]
        if len(names)<2:
            QtWidgets.QMessageBox.warning(self,"Warning","Select ≥2 to merge")
            return
        out = self.merge_name.text().strip()
        if not (out.lower().endswith(".csv") or out.lower().endswith(".json")):
            QtWidgets.QMessageBox.warning(self,"Warning","Filename must end in .csv or .json")
            return
        ok = self.dataset_manager.merge_datasets(names, out)
        self.refresh_dataset_lists()
        QtWidgets.QMessageBox.information(self,"Merged",f"Merged into {out}" if ok else "Merge failed")

    def refresh_dataset_lists(self):
        for lw,dpath in [
            (self.sidebar.active_list, TEMP_DIR),
            (self.sidebar.archive_list, ARCHIVE_DIR),
            (self.sidebar.merged_list, MERGED_DIR),
        ]:
            lw.clear()
            if os.path.isdir(dpath):
                for fn in sorted(os.listdir(dpath)):
                    if fn.lower().endswith((".csv",".json",".xlsx")):
                        lw.addItem(fn)
        # merge list
        self.merge_list.clear()
        for lst in (self.sidebar.active_list, self.sidebar.archive_list, self.sidebar.merged_list):
            for i in range(lst.count()):
                self.merge_list.addItem(lst.item(i).text())


    # ─── File I/O Slots ───────────────────────────────────────────────
    def open_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Dataset", "", "Data (*.csv *.json *.xlsx)"
        )
        if not path: return
        try:
            if path.endswith(".json"):
                df = pd.read_json(path)
            elif path.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            self.current_dataset = df
            self.dataset_model.update_data(df)
            self.tabs.setCurrentIndex(2)
            self.update_row_count()
            QtWidgets.QMessageBox.information(self,"Loaded",f"Loaded {len(df)} rows")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Error",f"Failed to load:\n{e}")

    def save_current_dataset(self):
        if self.current_dataset.empty:
            QtWidgets.QMessageBox.warning(self,"Warning","No dataset to save")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,"Save Dataset","","CSV (*.csv);;JSON (*.json);;Excel (*.xlsx)"
        )
        if not path: return
        try:
            if path.endswith(".json"):
                self.current_dataset.to_json(path, orient='records', indent=2)
            elif path.endswith(".xlsx"):
                self.current_dataset.to_excel(path, index=False)
            else:
                self.current_dataset.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self,"Saved",f"Saved to {path}")
            self.refresh_dataset_lists()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Error",f"Save failed:\n{e}")

    def export_dataset(self, fmt):
        ext = {"csv":"CSV (*.csv)","json":"JSON (*.json)","excel":"Excel (*.xlsx)"}[fmt]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Export Dataset","",ext)
        if not path: return
        try:
            if fmt=="json":
                self.current_dataset.to_json(path, orient='records', indent=2)
            elif fmt=="excel":
                self.current_dataset.to_excel(path, index=False)
            else:
                self.current_dataset.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self,"Exported",f"Exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Error",f"Export failed:\n{e}")


    # ─── Format Slots ─────────────────────────────────────────────────
    def add_field_dialog(self):
        f, ok = QtWidgets.QInputDialog.getText(self,"Add Field","Field name:")
        if ok and f:
            self.state['fields'].append(f)
            self.dataset_model.update_format(self.state['fields'])
            self.update_format_preview()

    def remove_selected_field(self):
        idx = self.fields_list.currentRow()
        if idx>=0:
            self.state['fields'].pop(idx)
            self.dataset_model.update_format(self.state['fields'])
            self.update_format_preview()

    def edit_selected_field(self):
        idx = self.fields_list.currentRow()
        if idx<0: return
        name, ok = QtWidgets.QInputDialog.getText(self,"Edit Field","Field name:",
                                                  text=self.state['fields'][idx])
        if ok and name:
            self.state['fields'][idx] = name
            self.dataset_model.update_format(self.state['fields'])
            self.update_format_preview()

    def update_format_preview(self):
        sep  = self.separator_combo.currentText()
        wrap = self.wrapper_combo.currentText()
        parts = [f"{wrap}{f}{wrap}" for f in self.state['fields']]
        self.format_preview.setText(sep.join(parts))


    # ─── Settings I/O ─────────────────────────────────────────────────
    def load_settings(self):
        if not Path(SETTINGS_FILE).exists():
            return
        try:
            s = json.loads(Path(SETTINGS_FILE).read_text())
            self.state.update(s)
            prov = s.get('model_provider')
            if prov:
                self.sidebar.provider_combo.setCurrentText(prov)
                self.sidebar.api_key_input.setText(s.get('api_key',''))
                self.sidebar.model_combo.setCurrentText(s.get('model_name',''))
            self.wrapper_combo.setCurrentText(s.get('data_separator', self.state['data_separator']))
            self.separator_combo.setCurrentText(s.get('section_separator', self.state['section_separator']))
            self.theme.dark_mode = s.get('dark_mode', self.theme.dark_mode)
            self.theme.apply()
        except Exception:
            logger.exception("Failed to load settings")

    def save_settings(self):
        s = {
            'fields': self.state['fields'],
            'examples': self.state['examples'],
            'data_separator': self.wrapper_combo.currentText(),
            'section_separator': self.separator_combo.currentText(),
            'model_provider': self.sidebar.provider_combo.currentText(),
            'model_name': self.sidebar.model_combo.currentText(),
            'api_key': self.sidebar.api_key_input.text(),
            'dark_mode': self.theme.dark_mode,
        }
        try:
            with open(SETTINGS_FILE,'w') as f:
                json.dump(s,f,indent=2)
            QtWidgets.QMessageBox.information(self,"Success","Settings saved")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Error",f"Save failed:\n{e}")

    def reset_settings(self):
        if QtWidgets.QMessageBox.question(self,"Reset Settings",
                "Reset to defaults?",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No
           ) != QtWidgets.QMessageBox.Yes:
            return
        try: Path(SETTINGS_FILE).unlink()
        except: pass
        QtWidgets.QMessageBox.information(self,"Reset","Settings reset. Restart to apply.")


    # ─── Utility ──────────────────────────────────────────────────────
    def zoom_in(self):
        f = QtWidgets.QApplication.instance().font()
        f.setPointSize(f.pointSize()+1)
        QtWidgets.QApplication.instance().setFont(f)

    def zoom_out(self):
        f = QtWidgets.QApplication.instance().font()
        f.setPointSize(max(8, f.pointSize()-1))
        QtWidgets.QApplication.instance().setFont(f)

    def _update_memory_usage(self):
        try:
            import psutil
            m = psutil.Process(os.getpid()).memory_info().rss/1024**2
            self.memory_label.setText(f"{m:.1f} MB")
        except ImportError:
            self.memory_label.setText("psutil missing")


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont("Segoe UI" if sys.platform=="win32" else "Arial", 10)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
