# src/dataset_model.py

import pandas as pd
from PySide6 import QtCore
from PySide6.QtCore import Qt


class DatasetModel(QtCore.QAbstractTableModel):
    """
    A Qt table model wrapping a pandas DataFrame,
    with support for editable cells and live text filtering.
    """
    def __init__(self, data: pd.DataFrame = None, parent=None):
        """
        Args:
            data (pd.DataFrame, optional): initial data. Defaults to empty DataFrame.
            parent: optional Qt parent.
        """
        super().__init__(parent)
        self._original_data: pd.DataFrame = data.copy() if data is not None else pd.DataFrame()
        self._filtered_data: pd.DataFrame = self._original_data.copy()
        self._filter_text: str = ""

    def rowCount(self, parent=None) -> int:
        return len(self._filtered_data)

    def columnCount(self, parent=None) -> int:
        return len(self._filtered_data.columns)

    def data(self, index: QtCore.QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        value = self._filtered_data.iat[index.row(), index.column()]
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            return str(value)
        if role == Qt.TextAlignmentRole:
            return Qt.AlignLeft | Qt.AlignVCenter
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._filtered_data.columns[section])
            else:
                # 1-based row numbers
                return str(section + 1)
        if role == Qt.TextAlignmentRole:
            return Qt.AlignLeft | Qt.AlignVCenter
        return None

    def flags(self, index: QtCore.QModelIndex):
        # Allow editing
        return super().flags(index) | Qt.ItemIsEditable

    def setData(self, index: QtCore.QModelIndex, value, role=Qt.EditRole) -> bool:
        if role == Qt.EditRole and index.isValid():
            # Update filtered data; original remains unchanged
            self._filtered_data.iat[index.row(), index.column()] = value
            self.dataChanged.emit(index, index)
            return True
        return False

    def update_data(self, new_data: pd.DataFrame):
        """
        Replace the model's data with a new DataFrame, clearing any existing filter.
        Emits modelReset.
        """
        self.beginResetModel()
        self._original_data = new_data.copy()
        self._filtered_data = new_data.copy()
        self._filter_text = ""
        self.endResetModel()

    def filter_data(self, text: str):
        """
        Apply a case-insensitive substring filter across all columns.
        Rows that do not contain `text` in any cell are hidden.
        Emits modelReset.
        """
        self.beginResetModel()
        self._filter_text = text or ""
        if not self._filter_text:
            self._filtered_data = self._original_data.copy()
        else:
            mask = self._original_data.apply(
                lambda row: row.astype(str)
                                .str.contains(self._filter_text, case=False)
                                .any(),
                axis=1
            )
            self._filtered_data = self._original_data.loc[mask].copy()
        self.endResetModel()

    def clear_filter(self):
        """Remove any active filter, restoring full dataset."""
        if self._filter_text:
            self.filter_data("")

