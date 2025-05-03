#!/usr/bin/env python3
# src/menus.py

import webbrowser
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt

DOCS_URL = "https://github.com/yourusername/davyd/docs"

class MenuBar:
    """
    Encapsulates the File, Edit, View, and Help menus.
    Attach to a QMainWindow via `MenuBar(window)`.
    """

    def __init__(self, window: QtWidgets.QMainWindow):
        self.window = window
        self._build()

    def _build(self):
        mb = self.window.menuBar()

        # ─ File ────────────────────────────────────────────────────────────
        fm = mb.addMenu("&File")
        fm.addAction(QtGui.QAction("&New Project",  self.window))
        fm.addAction(QtGui.QAction("&Open Dataset", self.window, shortcut="Ctrl+O",
                                  triggered=self.window.open_dataset))
        fm.addAction(QtGui.QAction("&Save Dataset", self.window, shortcut="Ctrl+S",
                                  triggered=self.window.save_current_dataset))
        em = fm.addMenu("&Export")
        em.addAction(QtGui.QAction("As CSV",   self.window, triggered=lambda: self.window.export_dataset('csv')))
        em.addAction(QtGui.QAction("As JSON",  self.window, triggered=lambda: self.window.export_dataset('json')))
        em.addAction(QtGui.QAction("As Excel", self.window, triggered=lambda: self.window.export_dataset('excel')))
        fm.addSeparator()
        fm.addAction(QtGui.QAction("&Exit", self.window, shortcut="Ctrl+Q", triggered=self.window.close))

        # ─ Edit ────────────────────────────────────────────────────────────
        ed = mb.addMenu("&Edit")
        ed.addAction(QtGui.QAction("&Undo", self.window, shortcut="Ctrl+Z", triggered=self.window.undo))
        ed.addAction(QtGui.QAction("&Redo", self.window, shortcut="Ctrl+Y", triggered=self.window.redo))
        ed.addSeparator()
        ed.addAction(QtGui.QAction("&Preferences", self.window, triggered=self.window.show_preferences))

        # ─ View ────────────────────────────────────────────────────────────
        vm = mb.addMenu("&View")
        self.dark_mode_action = QtGui.QAction("&Dark Mode", self.window, checkable=True,
                                              triggered=self.window.toggle_dark_mode)
        self.dark_mode_action.setChecked(self.window.state['dark_mode'])
        vm.addAction(self.dark_mode_action)
        vm.addAction(QtGui.QAction("Zoom &In",  self.window, shortcut="Ctrl++", triggered=self.window.zoom_in))
        vm.addAction(QtGui.QAction("Zoom &Out", self.window, shortcut="Ctrl+-", triggered=self.window.zoom_out))

        # ─ Help ────────────────────────────────────────────────────────────
        hm = mb.addMenu("&Help")
        hm.addAction(QtGui.QAction("Documentation", self.window, triggered=lambda: webbrowser.open(DOCS_URL)))
        hm.addAction(QtGui.QAction("&About", self.window, triggered=self.window.show_about))
