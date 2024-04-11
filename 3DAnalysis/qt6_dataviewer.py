import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._data.columns[section]
            else:
                return self._data.index[section]
        return None


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.table = QTableView(self)
        self.model = None
        self.initUI()

    def initUI(self):
        # Sample data
        data = pd.DataFrame({
          'Column1': [1, 2, 3, 4],
          'Column2': [5, 6, 7, 8],
          'Column3': ['A', 'B', 'C', 'D']
        })
        
        self.model = PandasModel(data)
        self.table.setModel(self.model)

        self.setCentralWidget(self.table)
        self.setWindowTitle('DataFrame Example')
        self.resize(400, 250)


def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()