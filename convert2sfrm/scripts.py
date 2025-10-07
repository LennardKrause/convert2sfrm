import sys
from PyQt6 import QtGui, QtWidgets
import convert2sfrm
from convert2sfrm.classes import ConversionWindow

def BioMAX2sfrm():
    from convert2sfrm.functions import prepare_BioMAX, convert_h5_to_sfrm
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
                           fc_process=convert_h5_to_sfrm)
    win.show()
    sys.exit(app.exec())

def BioMAX2cbf():
    from convert2sfrm.functions import prepare_BioMAX, convert_h5_to_cbf
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    app.setFont(font)
    app.setStyle('fusion')
    win = ConversionWindow(title=f'Convert MicroMAX / BioMAX to .cbf, version {convert2sfrm.__version__} (released {convert2sfrm.__date__})',
                           fn_suffix='_master.h5',
                           header_ext=['Run', 'Kappa'],
                           # 'add': add to counter
                           # 'mul': multiply with counter
                           # None : insert value
                           header_val=[('add', 1), ('mul', 50)],
                           fc_prepare=prepare_BioMAX,
                           fc_process=convert_h5_to_cbf)
    win.show()
    sys.exit(app.exec())

def ESRF2sfrm():
    from convert2sfrm.functions import prepare_ESRF, convert_img_to_sfrm
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    app.setFont(font)
    app.setStyle('fusion')
    win = ConversionWindow(title=f'Convert ESRF cbf.gz to .sfrm, version {convert2sfrm.__version__} (released {convert2sfrm.__date__})',
                           fn_suffix='/',
                           header_ext=['Run'],
                           # 'add': add to counter
                           # 'mul': multiply with counter
                           # None : insert value
                           header_val=[('add', 1)],
                           fc_prepare=prepare_ESRF,
                           fc_process=convert_img_to_sfrm)
    win.show()
    sys.exit(app.exec())
