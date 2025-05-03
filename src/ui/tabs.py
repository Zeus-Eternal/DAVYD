#!/usr/bin/env python3
# src/ui/tabs.py

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import pandas as pd
from dataset_model import DatasetModel

class TabsManager:
    """
    Enhanced TabsManager with complete styling and functionality from original implementation
    """
    def __init__(self, parent, tabs_widget: QtWidgets.QTabWidget, layout: QtWidgets.QLayout):
        self.parent = parent
        self.tabs = tabs_widget
        self.layout = layout

        self._style_tabs()
        self._add_tabs()
        self.layout.addWidget(self.tabs)

    def _style_tabs(self):
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border:1px solid #444; 
                border-radius:4px; 
                margin-top:-1px; 
            }
            QTabBar::tab { 
                background:#34495e; 
                color:#ecf0f1; 
                padding:8px 16px;
                border:1px solid #444; 
                margin-right:2px;
                border-top-left-radius:4px; 
                border-top-right-radius:4px; 
            }
            QTabBar::tab:selected { 
                background:#2c3e50; 
                border-bottom-color:#3498db; 
            }
            QTabBar::tab:hover { 
                background:#3d566e; 
            }
        """)

    def _add_tabs(self):
        self._add_structure_tab()
        self._add_generation_tab()
        self._add_visualization_tab()
        self._add_management_tab()

    # ─── Dataset Structure Tab ──────────────────────────────────────────────
    def _add_structure_tab(self):
        tab = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(tab)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(15)

        # Fields List
        grp = QtWidgets.QGroupBox("Dataset Fields")
        fl = QtWidgets.QVBoxLayout(grp)
        self.parent.fields_list = QtWidgets.QListWidget()
        self.parent.fields_list.setStyleSheet("""
            QListWidget { 
                background:#2c3e50; 
                color:white; 
                border:1px solid #444; 
                border-radius:4px; 
            }
            QListWidget::item { 
                padding:5px; 
                border-bottom:1px solid #444; 
            }
            QListWidget::item:selected { 
                background:#3498db; 
                color:white; 
            }
        """)
        fl.addWidget(self.parent.fields_list)

        # Field Controls
        btns = QtWidgets.QHBoxLayout()
        for txt, ico, slot in [
            ("Add Field", QtWidgets.QStyle.SP_FileDialogNewFolder, self.parent.add_field_dialog),
            ("Remove", QtWidgets.QStyle.SP_TrashIcon, self.parent.remove_selected_field),
            ("Edit", QtWidgets.QStyle.SP_FileDialogDetailedView, self.parent.edit_selected_field)
        ]:
            btn = QtWidgets.QPushButton(txt)
            btn.setIcon(self.parent.style().standardIcon(ico))
            btn.setStyleSheet("""
                QPushButton { 
                    padding:5px 10px; 
                    border-radius:4px;
                    background:#34495e;
                    color:white;
                }
                QPushButton:hover {
                    background:#3d566e;
                }
            """)
            btn.clicked.connect(slot)
            btns.addWidget(btn)
        fl.addLayout(btns)
        l.addWidget(grp)

        # Formatting Options
        fmt_grp = QtWidgets.QGroupBox("Formatting Options")
        fmt = QtWidgets.QFormLayout(fmt_grp)
        
        self.parent.separator_combo = QtWidgets.QComboBox()
        self.parent.separator_combo.addItems(["|", ":", "-", "~", "•", "→"])
        fmt.addRow("Field Separator:", self.parent.separator_combo)
        
        self.parent.wrapper_combo = QtWidgets.QComboBox()
        self.parent.wrapper_combo.addItems(['"', "'", "`", "«»", "''", "``"])
        fmt.addRow("Data Wrapper:", self.parent.wrapper_combo)

        preview_btn = QtWidgets.QPushButton("Update Preview")
        preview_btn.clicked.connect(self.update_format_preview)
        fmt.addRow(preview_btn)
        
        self.parent.format_preview = QtWidgets.QLabel()
        self.parent.format_preview.setStyleSheet("""
            QLabel { 
                background:#2c3e50; 
                color:#ecf0f1; 
                padding:10px;
                border-radius:4px; 
                border:1px solid #444; 
            }
        """)
        fmt.addRow("Preview:", self.parent.format_preview)
        
        l.addWidget(fmt_grp)
        self.tabs.addTab(tab, "📝 Dataset Structure")

    # ─── Generation Tab ──────────────────────────────────────────────────────
    def _add_generation_tab(self):
        tab = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(tab)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(15)

        # Parameters
        grp = QtWidgets.QGroupBox("Generation Parameters")
        fl = QtWidgets.QFormLayout(grp)
        
        self.parent.num_entries = QtWidgets.QSpinBox()
        self.parent.num_entries.setRange(1, 10000)
        self.parent.num_entries.setValue(500)
        fl.addRow("Number of Entries:", self.parent.num_entries)

        # Advanced Options
        self.parent.advanced_group = QtWidgets.QGroupBox("Advanced Options")
        self.parent.advanced_group.setVisible(False)
        adv = QtWidgets.QFormLayout(self.parent.advanced_group)
        
        self.parent.temperature = QtWidgets.QDoubleSpinBox()
        self.parent.temperature.setRange(0.1, 2.0)
        self.parent.temperature.setValue(0.7)
        adv.addRow("Temperature:", self.parent.temperature)
        
        self.parent.max_tokens = QtWidgets.QSpinBox()
        self.parent.max_tokens.setRange(100, 10000)
        self.parent.max_tokens.setValue(2000)
        adv.addRow("Max Tokens:", self.parent.max_tokens)
        
        fl.addRow(self.parent.advanced_group)

        # Toggle
        self.parent.advanced_options = QtWidgets.QCheckBox("Show Advanced Options")
        self.parent.advanced_options.toggled.connect(self.parent.toggle_advanced_options)
        fl.addRow(self.parent.advanced_options)

        l.addWidget(grp)

        # Progress
        self.parent.progress_bar = QtWidgets.QProgressBar()
        self.parent.progress_bar.setStyleSheet("""
            QProgressBar { 
                border:1px solid #444; 
                border-radius:4px; 
                height:20px; 
            }
            QProgressBar::chunk { 
                background-color:#3498db; 
            }
        """)
        l.addWidget(self.parent.progress_bar)

        # Generate Button
        gen_btn = QtWidgets.QPushButton("✨ Generate Dataset")
        gen_btn.setStyleSheet("""
            QPushButton { 
                font-weight:bold; 
                font-size:16px; 
                padding:12px;
                background-color:#e74c3c; 
                color:white; 
                border-radius:6px; 
            }
            QPushButton:hover { 
                background-color:#c0392b; 
            }
        """)
        gen_btn.clicked.connect(self.parent.start_generation)
        l.addWidget(gen_btn)

        # Status
        self.parent.generation_status = QtWidgets.QLabel()
        self.parent.generation_status.setAlignment(Qt.AlignCenter)
        l.addWidget(self.parent.generation_status)
        
        l.addStretch()
        self.tabs.addTab(tab, "⚡ Generation")

    # ─── Visualization Tab ───────────────────────────────────────────────────
    def _add_visualization_tab(self):
        tab = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(tab)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(15)

        # Search
        hl = QtWidgets.QHBoxLayout()
        self.parent.search_field = QtWidgets.QLineEdit(placeholderText="Search dataset...")
        self.parent.search_field.textChanged.connect(self.parent.filter_dataset_view)
        hl.addWidget(self.parent.search_field)
        
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.parent.search_field.clear())
        hl.addWidget(clear_btn)
        
        self.parent.row_count_label = QtWidgets.QLabel("0 rows")
        hl.addWidget(self.parent.row_count_label)
        l.addLayout(hl)

        # Table
        self.parent.dataset_model = DatasetModel(pd.DataFrame())
        self.parent.dataset_table = QtWidgets.QTableView()
        self.parent.dataset_table.setStyleSheet("""
            QTableView { 
                background:#2c3e50; 
                color:white; 
                border:1px solid #444;
                gridline-color:#444; 
                alternate-background-color:#34495e;
            }
            QTableView::item:selected { 
                background:#3498db; 
                color:white; 
            }
            QHeaderView::section { 
                background-color:#34495e; 
                color:white;
                padding:5px; 
                border:1px solid #444; 
            }
        """)
        self.parent.dataset_table.setModel(self.parent.dataset_model)
        l.addWidget(self.parent.dataset_table)

        # Visualization
        viz_grp = QtWidgets.QGroupBox("Data Visualization")
        vz = QtWidgets.QVBoxLayout(viz_grp)
        
        self.parent.viz_type = QtWidgets.QComboBox()
        self.parent.viz_type.addItems(["Bar Chart", "Pie Chart", "Scatter Plot", "Histogram"])
        vz.addWidget(self.parent.viz_type)
        
        self.parent.figure = Figure()
        self.parent.canvas = FigureCanvas(self.parent.figure)
        vz.addWidget(self.parent.canvas)
        
        l.addWidget(viz_grp)
        self.tabs.addTab(tab, "📊 Visualization")

    # ─── Management Tab ──────────────────────────────────────────────────────
    def _add_management_tab(self):
        tab = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(tab)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(15)

        # Dataset Lists
        self.parent.dataset_lists = QtWidgets.QTabWidget()
        for which, title in [
            ("active_list", "Active Datasets"),
            ("archive_list", "Archived Datasets"),
            ("merged_list", "Merged Datasets")
        ]:
            container = QtWidgets.QWidget()
            cl = QtWidgets.QVBoxLayout(container)
            
            lw = QtWidgets.QListWidget()
            lw.setStyleSheet("""
                QListWidget { 
                    background:#2c3e50; 
                    color:white; 
                    border:1px solid #444;
                    border-radius:4px; 
                }
            """)
            setattr(self.parent, which, lw)
            
            search = QtWidgets.QLineEdit(placeholderText="Search...")
            search.textChanged.connect(lambda text, lw=lw: self.parent.filter_datasets(text, lw))
            cl.addWidget(search)
            cl.addWidget(lw)
            
            self.parent.dataset_lists.addTab(container, title)
        l.addWidget(self.parent.dataset_lists)

        # Action Buttons
        btns = QtWidgets.QHBoxLayout()
        for name, slot, color in [
            ("Download", self.parent.download_dataset, "#3498db"),
            ("Archive", self.parent.archive_dataset, "#9b59b6"),
            ("Restore", self.parent.restore_dataset, "#2ecc71"),
            ("Delete", self.parent.delete_dataset, "#e74c3c")
        ]:
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet(f"""
                QPushButton {{ 
                    background-color:{color}; 
                    color:white; 
                    padding:8px; 
                    border-radius:4px;
                }}
                QPushButton:hover {{ 
                    background-color:{self._darken_color(color)}; 
                }}
            """)
            btn.clicked.connect(slot)
            btns.addWidget(btn)
        l.addLayout(btns)

        # Merge Section
        merge_grp = QtWidgets.QGroupBox("Merge Datasets")
        mg = QtWidgets.QVBoxLayout(merge_grp)
        
        self.parent.merge_list = QtWidgets.QListWidget()
        self.parent.merge_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        mg.addWidget(self.parent.merge_list)
        
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Output filename:"))
        self.parent.merge_name = QtWidgets.QLineEdit("merged_dataset.csv")
        row.addWidget(self.parent.merge_name)
        mg.addLayout(row)
        
        merge_btn = QtWidgets.QPushButton("Merge Selected")
        merge_btn.setStyleSheet("""
            QPushButton { 
                background-color:#9b59b6; 
                color:white; 
                padding:8px; 
                border-radius:4px;
            }
            QPushButton:hover { 
                background-color:#8e44ad; 
            }
        """)
        merge_btn.clicked.connect(self.parent.merge_datasets)
        mg.addWidget(merge_btn)
        
        l.addWidget(merge_grp)
        self.tabs.addTab(tab, "🗂 Management")

    # ─── Helper Methods ──────────────────────────────────────────────────────
    def update_format_preview(self):
        """Update the format preview label"""
        sep = self.parent.separator_combo.currentText()
        wrap = self.parent.wrapper_combo.currentText()
        parts = [f"{wrap}{f}{wrap}" for f in self.parent.state['fields']]
        self.parent.format_preview.setText(sep.join(parts))

    def _darken_color(self, hex_color, amount=20):
        color = QtGui.QColor(hex_color)
        return color.darker(100 + amount).name()
