import sys
from PyQt6 import QtGui, QtWidgets
import convert2sfrm
from convert2sfrm.classes import ConversionWindow
from convert2sfrm.functions import prepare_BioMAX, convert_to_sfrm

def BioMAX2sfrm():
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    app.setFont(font)
    app.setStyle('fusion')
    win = ConversionWindow(title=f'Convert MicroMAX / BioMAX to .sfrm, version {convert2sfrm.__version__} (released {convert2sfrm.__date__})',
                           fn_suffix='_master.h5',
                           header_ext=['Run', 'Kappa'],
                           # 'add': add to counter
                           # 'mul': multiply with counter
                           # None : insert value
                           header_val=[('add', 1), ('mul', 50)],
                           fc_prepare=prepare_BioMAX,
                           fc_process=convert_to_sfrm)
    win.show()
    sys.exit(app.exec())
