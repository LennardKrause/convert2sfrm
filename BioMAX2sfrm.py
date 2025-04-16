import os
import sys
import collections
import h5py, hdf5plugin
import numpy as np
import multiprocessing as mp
from time import sleep
from PyQt6 import QtCore, QtGui, QtWidgets

# *** FUTURE ***
# figure out a way to calculate the sensor efficiency fast (without imports)
# we know energy, material and thickness
#
# Bruker/APEX doesn't support the JUNGFRAU detector
# so we tell it to use the EIGER settings e.g. HPAD, 10/37 px gaps
# by setting the name to: 'JUNGFRAU(EIGER2)'
# APEX only parses for 'EIGER' in the name

__version__ = '1.0.0'
__author__ = 'Lennard Krause'
__email__ = 'lkrause@chem.au.dk'
__date__ = '15.04.2025'
__year__ = '2025'

class ConversionWindow(QtWidgets.QMainWindow):
    process_finished = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # variables
        self.title = kwargs.get('title', f'Convert MicroMAX / BioMAX to .sfrm, version {__version__} (released {__date__})')
        self.kappa_inc = kwargs.get('kappa_inc', 50)
        self.file_suffix = kwargs.get('file_suffix', '_master.h5')
        # internals
        self.button_group_rem = QtWidgets.QButtonGroup()
        self.threads = QtCore.QThreadPool()
        self.stopped = False
        self.progress = 0

        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800, 400)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # add frame and layout
        layout = QtWidgets.QVBoxLayout()
        conv = QtWidgets.QFrame()
        conv.setLayout(layout)

        # add table widget
        self.table = QtWidgets.QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(5)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.table.setHorizontalHeaderLabels(['Path', 'Run', 'Kappa', 'Progress', 'Delete'])
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        progress_bar = ProgressDelegate(self.table)
        self.table.setItemDelegateForColumn(3, progress_bar)
        layout.addWidget(self.table)

        # add start and stop buttons
        self.btn_box = QtWidgets.QGroupBox()
        self.btn_box_layout = QtWidgets.QHBoxLayout()
        self.btn_box.setLayout(self.btn_box_layout)
        self.btn_conv = QtWidgets.QPushButton('Convert')
        self.btn_conv.clicked.connect(self.on_run_clicked)
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet('QPushButton:enabled{background-color: red;}')
        self.btn_stop.clicked.connect(self.thread_abort)
        self.rdo_overwrite = QtWidgets.QRadioButton('Overwrite')
        self.rdo_overwrite.setChecked(True)
        self.btn_box_layout.addWidget(self.btn_stop, 1)
        self.btn_box_layout.addWidget(self.btn_conv, 10)
        self.btn_box_layout.addWidget(self.rdo_overwrite, 1)
        layout.addWidget(self.btn_box)

        # add frame and layout
        help = QtWidgets.QScrollArea()
        help_text = QtWidgets.QLabel()
        help_text.setText('<b><h1>How to use</h1></b>'
                          '<ul><h2>General</h2>'
                          '<li> Drag and drop the h5 master files (<i>*_master.h5</i>) onto the window </li>'
                          '<li> You can drop all files, only <i>*_master.h5</i> files will be added </li>'
                          '<li> You can change the run number and the kappa angle for each run </li>'
                          '<li> The rows (runs) will be converted successively using all the cores you have </li>'
                          '</ul>'
                          '<ul><h2>Buttons</h2>'
                          '<li> Click <b>Convert</b> to start the conversion </li>'
                          '<li> Using <b>Stop</b> might save storage space by aborting the process </li>'
                          '<li> The <b>X</b> button will remove the row from the table </li>'
                          '<li> Untick <b>Overwrite</b> if you want to save time finishing previously aborted conversions </li>'
                          '</ul>'
                          '<ul><h2>Features</h2>'
                          '<li> If multiple <i>*_master.h5</i> files are dropped together,<br> they have their run number and kappa angle incemented automatically </li>'
                          '</ul>'
                          )
        help.setWidget(help_text)

        tabs.addTab(conv, 'Convert')
        tabs.addTab(help, 'Help')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            counter = 0
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path and file_path.endswith(self.file_suffix):
                    self.table.setRowCount(self.table.rowCount() + 1)
                    self.table.setItem(self.table.rowCount()-1, 0, QtWidgets.QTableWidgetItem(file_path))
                    self.table.setItem(self.table.rowCount()-1, 1, QtWidgets.QTableWidgetItem(str(counter+1)))
                    self.table.setItem(self.table.rowCount()-1, 2, QtWidgets.QTableWidgetItem(str(counter*self.kappa_inc)))
                    # progress bar
                    bar = QtWidgets.QTableWidgetItem()
                    bar.setData(QtCore.Qt.ItemDataRole.UserRole+1000, 0)
                    item = QtWidgets.QTableWidgetItem(bar)
                    item.setFlags(~QtCore.Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(self.table.rowCount()-1, 3, item)
                    # remove button
                    button_rem = QtWidgets.QToolButton()
                    button_rem.setText('X')
                    button_rem.clicked.connect(self.table_delete_row)        
                    self.button_group_rem.addButton(button_rem)
                    self.table.setIndexWidget(self.table.model().index(self.table.rowCount()-1, 4), button_rem)
                    counter += 1
            event.accept()
        else:
            event.ignore()
    
    def table_delete_row(self):
        button = self.sender()
        if button:
            row = self.table.indexAt(button.pos()).row()
            self.table.removeRow(row)
            self.button_group_rem.removeButton(button)

    def make_runlist(self):
        self.runlist = []
        for row in range(self.table.rowCount()):
            entries = [row]
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                if col in [0, 1, 2]:
                    text = item.text()
                    entries.append(text)
                if col == 3:
                    if item.data(QtCore.Qt.ItemDataRole.UserRole+1000) < 100:
                        self.runlist.append(tuple(entries))
                    break

    def on_run_clicked(self):
        # construct runlist from table
        self.make_runlist()
        # iteerate over runs
        if self.runlist:
            # disable buttons
            self.buttons_enable(False)
            for (idx, h5_master, run_number, gon_kappa) in self.runlist:
                self.worker = Worker(self.prepare_BioMAX_processing, [str(h5_master), int(run_number), int(gon_kappa)], {'index':int(idx)})
                self.worker.signals.finished.connect(self.thread_finished)
                self.worker.signals.aborted.connect(self.thread_aborted)
                self.worker.signals.update.connect(self.thread_update)
                self.threads.start(self.worker)
                # wait until done
                loop = QtCore.QEventLoop()
                self.process_finished.connect(loop.quit)
                loop.exec()
                if self.stopped:
                    break
        # enable buttons
        self.buttons_enable(True)
        # reset stop button
        self.stopped = False

    def buttons_enable(self, toggle=False):
        self.btn_stop.setEnabled(not toggle)
        self.btn_conv.setEnabled(toggle)
        for btn in self.button_group_rem.buttons():
            btn.setEnabled(toggle)

    def thread_update(self, index, value):
        self.processed.append(value)
        self.progress = min(max(int(len(self.processed) / self.progress_total * 100), 1), 99)
        self.table.item(index, 3).setData(QtCore.Qt.ItemDataRole.UserRole+1000, self.progress)
    
    def thread_aborted(self, index):
        # negative progress triggers red coloring of progress bar 
        self.table.item(index, 3).setData(QtCore.Qt.ItemDataRole.UserRole+1000, -self.progress)
        self.thread_cleanup()

    def thread_finished(self, index):
        # any progress value above max (100) triggers green coloring of progress bar 
        self.table.item(index, 3).setData(QtCore.Qt.ItemDataRole.UserRole+1000, 999)
        self.thread_cleanup()

    def thread_cleanup(self):
        # wait for pool to finish
        self.threads.clear()
        self.threads.waitForDone()
        self.process_finished.emit()

    def thread_abort(self):
        self.stopped = True

    def prepare_BioMAX_processing(self, h5_master, run_number, gon_kappa, index=None):
        # BioMAX Dectris EIGER2 CdTe 16M D023193
        h5_paths = {'dat_images':'entry/data/',
                    'src_beamline_name':'entry/instrument/name',
                    'src_facility_name':'entry/source/name',
                    'src_facility_type':'entry/source/type',
                    'src_wavelength':'entry/instrument/beam/incident_wavelength',
                    'det_bc_x':'entry/instrument/detector/beam_center_x',
                    'det_bc_y':'entry/instrument/detector/beam_center_y',
                    'det_distance':'entry/instrument/detector/detector_distance',
                    'det_frame_time':'entry/instrument/detector/frame_time',
                    'det_bit_depth':'entry/instrument/detector/bit_depth_image',
                    'det_id':'entry/instrument/detector/detector_number',
                    'det_desc':'entry/instrument/detector/description',
                    'det_pxs_x':'entry/instrument/detector/x_pixel_size',
                    'det_pxs_y':'entry/instrument/detector/y_pixel_size',
                    'det_px_x':'entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
                    'det_px_y':'entry/instrument/detector/detectorSpecific/y_pixels_in_detector',
                    'sensor_material':'entry/instrument/detector/sensor_material',
                    'sensor_thickness':'entry/instrument/detector/sensor_thickness',
                    'gon_scan_width':'entry/sample/goniometer/omega_range_average',
                    'gon_omega_start':'entry/sample/goniometer/omega',
                    'gon_omega_end':'entry/sample/goniometer/omega_end',
                    'gon_omega':'entry/sample/transformations/Omega',
                    }

        par = {}
        par['h5_file'] = h5_master
        par['overwrite'] = self.rdo_overwrite.isChecked()

        with h5py.File(par['h5_file'], 'r') as h5f:
            for key, path in h5_paths.items():
                if key == 'dat_images':
                    par[key] = list()
                    for end in list(h5f[path].keys()):
                        link = os.path.join(path, end)
                        # check if the files are available or None
                        if not h5f.get(link, default=None) is None:
                            par[key].append({'link':link, 'num':len(h5f[link])})
                elif path in h5f:
                    value = h5f[path][()]
                    if isinstance(value, np.bytes_):
                        par[key] = value.decode()
                    elif isinstance(value, float) or isinstance(value, np.floating):
                        par[key] = np.round(float(value), 6)
                    elif isinstance(value, int) or isinstance(value, np.integer):
                        par[key] = int(value)
                    else:
                        par[key] = value

        assert par['det_pxs_x'] == par['det_pxs_y'], 'Pixel sizes are not equal!'
        par['det_pxs'] = par['det_pxs_x']

        if 'JUNGFRAU' in par['det_desc']:
            par['gon_scan_width'] = np.round(np.mean(par['gon_omega'][1:]-par['gon_omega'][:-1]), 6)
            par['gon_omega_start'] = np.round(par['gon_omega'], 6)
            par['gon_omega_end'] = np.round(np.append(par['gon_omega'][1:], np.array([par['gon_omega'][-1]+par['gon_scan_width']])), 6)
            # Bruker/APEX doesn't support the JUNGFRAU detector
            # so we tell it to use the EIGER settings
            # e.g. HPAD, 10/37 px gaps
            # APEX only parses for EIGER in the name
            par['det_name'] = 'JUNGFRAU(EIGER2)'
            # *** FUTURE ***
            # figure out a way to calculate the efficiency fast (without imports)
            # we know energy, material and thickness
            par['sensor_efficiency'] = 1.0
        elif 'EIGER2' in par['det_desc']:
            par['det_name'] = 'EIGER2'
            par['sensor_efficiency'] = 1.0
        else:
            print('Detector not supported!')
            return

        par['user_name'] = '?'
        par['sample_name'] = '?'
        par['sample_number'] = 0
        
        temp_path, temp_name = os.path.split(par['h5_file'])
        # get run number from h5-file name
        # e.g. sample1-crystal2_3_master.h5
        #  - remove the extension
        #  - remove '_master'
        #  - reverse the string
        #  - split once at '_'
        par['frm_run'], par['frm_name'] = (os.path.splitext(temp_name)[0]).removesuffix('_master')[::-1].split('_', 1)
        par['frm_name'] = par['frm_name'][::-1]
        par['frm_run'] = run_number
        # sum images to 0.5 degree scan frames
        #  - save storage space
        #  - frame queue in SAINT is memory limited (> EIGER2 16M)
        par['frm_sum'] = int(round(0.5 / par['gon_scan_width'], 0))
        # mini-kappa angle in degrees
        par['kappa'] = gon_kappa
        # mini-kappa alpha angle in degrees
        par['alpha'] = 24.0
        # target sfrm directory
        par['frm_path'] = os.path.join(temp_path, f'{par["frm_name"]}_sfrm')
        # get max vlaue (16 or 32 bit images) to flag dead pixels
        par['det_max_counts'] = 2**par['det_bit_depth']-1
        # total number of images in h5-file
        par['dat_images_num'] = sum([i['num'] for i in par['dat_images']])
        # Bruker convention
        # detector [0,0] is lower left
        # numpy [0,0] is upper left
        par['det_bc_y'] = par['det_px_y'] - par['det_bc_y']
        
        # create the output directory
        if not os.path.exists(par['frm_path']):
            os.mkdir(par['frm_path'])
        
        # prepare a SAINT integration mask (X-ray Aperture, xa) file
        prepare_saint_mask(par['h5_file'], par['dat_images'][0]['link'], par)

        # todo: total number of images to be converted
        assert par['dat_images_num'] % par['frm_sum'] == 0, 'Number of images must be an integer multiple of image summation!'
        self.progress_total = par['dat_images_num'] // par['frm_sum']
        self.processed = []

        with mp.Pool() as self.pool:
            # the number of the resulting sfrm file
            # it is assumed that one h5master file contains one run of data
            # so inum is running continuously over one h5master file
            frm_num = 0
            # iterate over the h5files, the enumeration (idx) is needed to get the
            # number of images stored in this h5file (h5inum[idx]) to end the while loop
            for h5_num in par['dat_images']:
                # index for the current slice, slice from idx to idx + par['sfrm_sum']
                # the slice is summed and converted to sfrm
                h5_start = 0
                h5_end = h5_num['num'] - par['frm_sum']
                h5_step = h5_num['num'] // par['frm_sum']
                assert h5_num['num'] % par['frm_sum'] == 0, 'Number of images must be an integer multiple of image summation!'
                for h5_idx in np.linspace(h5_start, h5_end, h5_step, dtype=int):
                    frm_num += 1
                    # run the conversion of the slice in parallel
                    self.pool.apply_async(convert_to_sfrm, args=(h5_num['link'], h5_idx, frm_num, par), callback=lambda x: self.worker.signals.update.emit(index, x))
            # we're done filling the pool
            self.pool.close()
            
            while len(self.processed) < self.progress_total:
                if self.stopped:
                    self.pool.terminate()
                    self.pool.join()
                    return index, False
                sleep(1)
            
            # wait for the pool to finish
            self.pool.join()
            return index, True

class Worker(QtCore.QRunnable):
    def __init__(self, fn_funct, fn_args, fn_kwargs):
        '''
         code from:
         https://www.mfitzp.com/tutorials/multithreading-pyqt-applications-qthreadpool/
         
         fn_funct:  Conversion function
         fn_args:   Arguments to pass to the function
         fn_kwargs: Keywords to pass to the function
        '''
        QtCore.QRunnable.__init__(self)
        self.funct = fn_funct
        self.args = fn_args
        #if not 'parent' in fn_kwargs:
        #    fn_kwargs.update({'parent':self})
        self.kwargs = fn_kwargs
        self.signals = Signals()
    
    @QtCore.pyqtSlot()
    def run(self):
        # funct: returns int (index), bool (success)
        index, success = self.funct(*self.args, **self.kwargs)
        if success:
            self.signals.finished.emit(index, success)
        else:
            self.signals.aborted.emit(index, success)
    
    @QtCore.pyqtSlot()
    def update(self, index, value):
        self.signals.update.emit(index, value)

class Signals(QtCore.QObject):
    '''
     code from:
     https://www.mfitzp.com/tutorials/multithreading-pyqt-applications-qthreadpool/
     
     Custom signals can only be defined on objects derived from QObject
    '''
    finished = QtCore.pyqtSignal(int, bool)
    aborted = QtCore.pyqtSignal(int, bool)
    update = QtCore.pyqtSignal(int, int)

class ProgressDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter, option, index):
        progress = index.data(QtCore.Qt.ItemDataRole.UserRole+1000)
        opt = QtWidgets.QStyleOptionProgressBar()
        opt.rect = option.rect
        opt.minimum = 0
        opt.maximum = 100
        opt.progress = min(abs(progress), 100)
        opt.text = f'{opt.progress}%'
        opt.textVisible = True
        if progress > opt.maximum:
            opt.palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(93, 156, 89))
        elif progress < opt.minimum:
            opt.palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(223, 46, 56))
        QtWidgets.QApplication.style().drawControl(QtWidgets.QStyle.ControlElement.CE_ProgressBar, opt, painter)

def prepare_saint_mask(h5file, h5path, par):
    # open the h5 file and read the first image
    with h5py.File(h5file, 'r') as h5:
        data = h5[h5path][0,:,:]
    
    # make up a name for the mask
    # the masks we use are called X-ray Aperture (xa) files in saint and
    # have the naming convention: some_data_xa_01_0001.sfrm 
    # they can be provided per run and must be stored together with the frames
    mask_name = '{}_xa_{:02}_0001.sfrm'.format(os.path.join(par['frm_path'], par['frm_name']), par['frm_run'])
    
    # the dead areas are flagged as saturated, so their value is 65535 for
    # 16 bit unsigned integers. Eiger2 images can be stored at 16 or 32 bit
    # par['det_max_counts'] knows that value.
    # Everything saturated can't be trusted, everything below (even zero)
    # is data and is set to 1 (active)
    data[data < par['det_max_counts']] = 1
    # the dead areas are set to 0 (inactive)
    data[data == par['det_max_counts']] = 0
    data = data.astype(np.uint8)
    
    # calculate detector pixel per cm
    # this is normalized to a 512x512 detector format
    # Eiger2-1M pixel size is 0.075 mm 
    pix_per_512 = round((10.0 / 0.075) * (512.0 / data.shape[0]), 6)
    
    # default Bruker header
    header = init_bruker_header()
    
    # fill known header items
    header['NCOLS']      = [data.shape[1], 4]
    header['NROWS']      = [data.shape[0], 8]
    header['CCDPARM']    = [0.0, 1.00, 1.00, 0.0, (2**20)*par['frm_sum']]
    header['DETPAR']     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    header['DETTYPE']    = ['Eiger2', pix_per_512, 0.0, 0, 0.0, 0.0012, 1]
    header['SITE']       = par['src_facility_name']
    header['MODEL']      = par['src_beamline_name']
    header['TARGET']     = par['src_facility_type']
    header['WAVELEN']    = [par['src_wavelength'],par['src_wavelength'],par['src_wavelength']]
    header['TYPE']       = 'ACTIVE MASK'
    header['NPIXELB']    = [1, 1]
    header['MAXIMUM']    = 1
    header['MINIMUM']    = 0
    header['NCOUNTS']    = [data.sum(), 0]
    header['PHD']        = [1.00, 0.1]
    
    # write the frame
    write_bruker_frame(mask_name, header, data)

def init_bruker_header():
    header = collections.OrderedDict()
    header['FORMAT']  = 100                                                     # Frame Format -- 86=SAXI, 100=Bruker
    header['VERSION'] = 18                                                      # Header version number
    header['HDRBLKS'] = 15                                                      # Header size in 512-byte blocks
    header['TYPE']    = '?'                                                     # String indicating kind of data in the frame
    header['SITE']    = '?'                                                     # Site name
    header['MODEL']   = '?'                                                     # Diffractometer model
    header['USER']    = '?'                                                     # Username
    header['SAMPLE']  = '?'                                                     # Sample ID
    header['SETNAME'] = '?'                                                     # Basic data set name
    header['RUN']     = 1                                                       # Run number within the data set
    header['SAMPNUM'] = 1                                                       # Specimen number within the data set
    header['TITLE']   = ['', '', '', '', '', '', '', '']                        # User comments (8 lines)
    header['NCOUNTS'] = np.array([-9999, 0])                                    # Total frame counts, Reference detector counts
    header['NOVERFL'] = np.array([-1, 0, 0])                                    # SAXI Format: Number of overflows
                                                                                # Bruker Format: #Underflows; #16-bit overfl; #32-bit overfl
    header['MINIMUM'] = 0                                                       # Minimum pixel value
    header['MAXIMUM'] = 1                                                       # Maximum pixel value
    header['NONTIME'] = 0                                                       # Number of on-time events
    header['NLATE']   = 0                                                       # Number of late events for multiwire data
    header['FILENAM'] = '?'                                                     # (Original) frame filename
    header['CREATED'] = '?'                                                     # Date and time of creation
    header['CUMULAT'] = 0.0                                                     # Accumulated exposure time in real hours
    header['ELAPSDR'] = 0.0                                                     # Requested time for this frame in seconds
    header['ELAPSDA'] = 0.0                                                     # Actual time for this frame in seconds
    header['OSCILLA'] = 0                                                       # Nonzero if acquired by oscillation
    header['NSTEPS']  = 0                                                       # steps or oscillations in this frame
    header['RANGE']   = 0.0                                                     # Magnitude of scan range in decimal degrees
    header['START']   = 0.0                                                     # Starting scan angle value, decimal deg
    header['INCREME'] = 0.0                                                     # Signed scan angle increment between frames
    header['NUMBER']  = 1                                                       # Number of this frame in series (zero-based)
    header['NFRAMES'] = 1                                                       # Number of frames in the series
    header['ANGLES']  = np.array([0.0, 0.0, 0.0, 0.0])                          # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
    header['NOVER64'] = 0                                                       # Number of pixels > 64K
    header['NPIXELB'] = np.array([1, 1])                                        # Number of bytes/pixel; Number of bytes per underflow entry
    header['NROWS']   = np.array([512, 1])                                      # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
                                                                                # for each mosaic tile, X varying fastest
    header['NCOLS']   = np.array([512, 1])                                      # Number of pixels per row; number of mosaic tiles in X; dZ/dX
                                                                                # value for each mosaic tile, X varying fastest
    header['WORDORD'] = 0                                                       # Order of bytes in word; always zero (0=LSB first)
    header['LONGORD'] = 0                                                       # Order of words in a longword; always zero (0=LSW first
    header['TARGET']  = '?'                                                     # X-ray target material)
    header['SOURCEK'] = 0.0                                                     # X-ray source kV
    header['SOURCEM'] = 0.0                                                     # Source milliamps
    header['FILTER']  = '?'                                                     # Text describing filter/monochromator setting
    header['CELL']    = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])             # Cell constants, 2 lines  (A,B,C,Alpha,Beta,Gamma)
    header['MATRIX']  = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]) # Orientation matrix, 3 lines
    header['LOWTEMP'] = np.array([1, -17300, -6000])                            # Low temp flag; experiment temperature*100; detector temp*100
    header['ZOOM']    = np.array([0.0, 0.0, 1.0])                               # Image zoom Xc, Yc, Mag
    header['CENTER']  = np.array([256.0, 256.0, 256.0, 256.0])                  # X, Y of direct beam at 2-theta = 0
    header['DISTANC'] = 5.0                                                     # Sample-detector distance, cm
    header['TRAILER'] = -1                                                      # Byte pointer to trailer info (unused; obsolete)
    header['COMPRES'] = 0                                                       # Text describing compression method if any
    header['LINEAR']  = np.array([1.0, 0.0])                                    # Linear scale, offset for pixel values
    header['PHD']     = np.array([1.0, 0.1])                                    # Discriminator settings
    header['PREAMP']  = 0                                                       # Preamp gain setting
    header['CORRECT'] = 'INTERNAL'                                              # Flood correction filename
    header['WARPFIL'] = 'LINEAR'                                                # Spatial correction filename
    header['WAVELEN'] = np.array([0.0, 0.0, 0.0])                               # Wavelengths (average, a1, a2)
    header['MAXXY']   = np.array([0, 0])                                        # X,Y pixel # of maximum counts
    header['AXIS']    = 2                                                       # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    header['ENDING']  = np.array([0.0, 0.0, 0.0, 0.0])                          # Setting angles read at end of scan
    header['DETPAR']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])                # Detector position corrections (Xc,Yc,Dist,Pitch,Roll,Yaw)
    header['LUT']     = 'lut'                                                   # Recommended display lookup table
    header['DISPLIM'] = np.array([0.0, 63.0])                                   # Recommended display contrast window settings
    header['PROGRAM'] = 'Python Image Conversion'                               # Name and version of program writing frame
    header['ROTATE']  = 0                                                       # Nonzero if acquired by rotation (GADDS)
    header['BITMASK'] = '$NULL'                                                 # File name of active pixel mask (GADDS)
    header['OCTMASK'] = np.array([0, 0, 0, 0, 0, 0, 0, 0])                      # Octagon mask parameters (GADDS) #min x, min x+y, min y, max x-y, max x, max x+y, max y, max y-x
    header['ESDCELL'] = np.array([0.001, 0.001, 0.001, 0.02, 0.02, 0.02])       # Cell ESD's, 2 lines (A,B,C,Alpha,Beta,Gamma)
    header['DETTYPE'] = 'UNKNOWN'                                               # Detector type
    header['NEXP']    = np.array([1, 0, 0, 0, 0])                               # Number exposures in this frame; CCD bias level*100,;
                                                                                # Baseline offset (usually 32); CCD orientation; Overscan Flag
    header['CCDPARM'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])                     # CCD parameters for computing pixel ESDs; readnoise, e/ADU, e/photon, bias, full scale
    header['CHEM']    = '?'                                                     # Chemical formula
    header['MORPH']   = '?'                                                     # CIFTAB string for crystal morphology
    header['CCOLOR']  = '?'                                                     # CIFTAB string for crystal color
    header['CSIZE']   = '?'                                                     # String w/ 3 CIFTAB sizes, density, temp
    header['DNSMET']  = '?'                                                     # CIFTAB string for density method
    header['DARK']    = 'INTERNAL'                                              # Dark current frame name
    header['AUTORNG'] = np.array([0.0, 0.0, 0.0, 0.0, 1.0])                     # Autorange gain, time, scale, offset, full scale
    header['ZEROADJ'] = np.array([0.0, 0.0, 0.0, 0.0])                          # Adjustments to goniometer angle zeros (tth, omg, phi, chi)
    header['XTRANS']  = np.array([0.0, 0.0, 0.0])                               # Crystal XYZ translations
    header['HKL&XY']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0])                     # HKL and pixel XY for reciprocal space (GADDS)
    header['AXES2']   = np.array([0.0, 0.0, 0.0, 0.0])                          # Diffractometer setting linear axes (4 ea) (GADDS)
    header['ENDING2'] = np.array([0.0, 0.0, 0.0, 0.0])                          # Actual goniometer axes @ end of frame (GADDS)
    header['FILTER2'] = np.array([0.0, 0.0, 0.0, 1.0])                          # Monochromator 2-theta (deg), roll (deg), beam tilt, attenuation factor
    header['LEPTOS']  = ''
    #header['CFR']     = ['']
    return header

def write_bruker_frame(fname, fheader, fdata):
    ########################
    ## write_bruker_frame ##
    ##     FUNCTIONS      ##
    ########################
    def pad_table(table, bpp):
        '''
        pads a table with zeros to a multiple of 16 bytes
        '''
        padded = np.zeros(int(np.ceil(table.size * abs(bpp) / 16)) * 16 // abs(bpp)).astype(_BPP_TO_DT[bpp])
        padded[:table.size] = table
        return padded

    def format_bruker_header(fheader):
        '''
        
        '''
        fmt = {1:'{:<71} ',
               2:'{:<35} {:<35} ',
               3:'{:<23} {:<23} {:<23} ',
               4:'{:<17} {:<17} {:<17} {:<17} ',
               5:'{:<13} {:<13} {:<13} {:<13} {:<15} '}
    
        hdr_lines = []
        for name, entry in fheader.items():
            # cast entry to array
            entry = np.atleast_1d(entry)
            
            # round if (all) float
            if np.issubdtype(entry.dtype, np.floating):
                entry = np.round(entry, 6)

            # store header block index
            # calculate / write number
            # of header blocks at the end
            if name == 'HDRBLKS':
                hdr_hdrblks_idx = len(hdr_lines)

            # TITLE has 8 lines
            if name == 'TITLE':
                num = len(entry)
                for idx in range(8):
                    if num < idx:
                        line = f'{name:<7}:{entry[idx]:<72}'
                    else:
                        line = f'{name:<7}:{" ":<72}'
                    hdr_lines.append(line)
                    assert len(line) == 80, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                continue
            
            # OCTMASK: 6+2 entries in line
            if name == 'OCTMASK':
                part = '{:<11} {:<11} {:<11} {:<11} {:<11} {:<11} '.format(*entry[:6])
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
                part = fmt[len(entry[6:])].format(*entry[6:])
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
                continue
            
            # DETTYPE: Mixes Entry Types
            if name == 'DETTYPE':
                part = '{:<20} {:<11} {:<10} {:<2} {:<10} {:<10} {:<2} '.format(*entry)
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
                continue
            
            # write entries, up to 5 per line
            part = fmt[len(entry[:5])].format(*entry[:5])
            assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
            hdr_lines.append(f'{name:<7}:{part}')
            
            # add new line for remainder if more than 5
            if len(entry) > 5:
                part = fmt[len(entry[5:])].format(*entry[5:])
                assert len(part) == 72, f'Error writing Bruker header, line exceeds 80 characters! Key: {name}'
                hdr_lines.append(f'{name:<7}:{part}')
            continue
    
        # add header ending
        # header must be multiple of 512 bytes
        # 1 header block (HDRBLKS) is 512 bytes
        hdr_pad = 512 - (len(hdr_lines) * 80 % 512)
        hdr_end = '\x1a\x04'          # 2 bytes
        hdr_last = 'CFR: HDR: IMG: ' # 15 bytes
        
        # fill to next 512 bytes
        while hdr_pad > 80:
            hdr_lines.append(hdr_end + ''.join(['.'] * 78))
            hdr_pad -= 80
        
        # close header
        hdr_fill = str('.' * (hdr_pad - 17))
        hdr_lines.append(hdr_last + hdr_fill + hdr_end)
        
        # calculate new header size
        # and update header blocks
        name = 'HDRBLKS'
        entry = ((len(hdr_lines)-1) * 80 + hdr_pad) // 512
        hdr_lines[hdr_hdrblks_idx] = f'{name:<7}:{entry:<72}'

        return ''.join(hdr_lines)
    ########################
    ## write_bruker_frame ##
    ##   FUNCTIONS END    ##
    ########################
    
    # assign bytes per pixel to numpy integers
    # int8   Byte (-128 to 127)
    # int16  Integer (-32768 to 32767)
    # int32  Integer (-2147483648 to 2147483647)
    # uint8  Unsigned integer (0 to 255)
    # uint16 Unsigned integer (0 to 65535)
    # uint32 Unsigned integer (0 to 4294967295)
    _BPP_TO_DT = {1: np.uint8,
                  2: np.uint16,
                  4: np.uint32,
                 -1: np.int8,
                 -2: np.int16,
                 -4: np.int32}
    
    # read the bytes per pixel
    # frame data (bpp), underflow table (bpp_u)
    bpp, bpp_u = fheader['NPIXELB']

    # expand data to int64
    fdata = fdata.astype(np.int64)
    
    # generate underflow table
    # does not work as APEXII reads the data as uint8/16/32!
    if fheader['NOVERFL'][0] >= 0:
        data_underflow = fdata[fdata <= 0]
        fheader['NOVERFL'][0] = data_underflow.shape[0]
        table_underflow = pad_table(data_underflow, -1 * bpp_u)
        fdata[fdata < 0] = 0

    # generate 32 bit overflow table
    if bpp < 4:
        data_over_uint16 = fdata[fdata >= 65535]
        table_data_uint32 = pad_table(data_over_uint16, 4)
        fheader['NOVERFL'][2] = data_over_uint16.shape[0]
        fdata[fdata >= 65535] = 65535

    # generate 16 bit overflow table
    if bpp < 2:
        data_over_uint8 = fdata[fdata >= 255]
        table_data_uint16 = pad_table(data_over_uint8, 2)
        fheader['NOVERFL'][1] = data_over_uint8.shape[0]
        fdata[fdata >= 255] = 255

    # shrink data to desired bpp
    fdata = fdata.astype(_BPP_TO_DT[bpp])
    
    # write frame
    with open(fname, 'wb') as brukerFrame:
        brukerFrame.write(format_bruker_header(fheader).encode('ASCII'))
        brukerFrame.write(fdata.tobytes())
        if fheader['NOVERFL'][0] >= 0:
            brukerFrame.write(table_underflow.tobytes())
        if bpp < 2 and fheader['NOVERFL'][1] > 0:
            brukerFrame.write(table_data_uint16.tobytes())
        if bpp < 4 and fheader['NOVERFL'][2] > 0:
            brukerFrame.write(table_data_uint32.tobytes())

def kappa_to_euler(k_omg, kappa, alpha, k_phi):
    '''
     converts kappa to eulerian geometry
     needed for .sfrm header
     r_: in radians
     k_: kappa geometry
     e_: euler geometry
     alternative delta:
     r_delta_ = np.arcsin(np.cos(np.deg2rad(alpha)) * np.sin(np.deg2rad(kappa) / 2.0) / np.cos(r_e_chi / 2.0))
    '''
    import numpy as np
    r_k_omg = np.deg2rad(k_omg)
    r_k_phi = np.deg2rad(k_phi)
    r_e_chi = 2.0 * np.arcsin(np.sin(np.deg2rad(kappa) / 2.0) * np.sin(np.deg2rad(alpha)))
    r_delta = np.arccos(np.cos(np.deg2rad(kappa) / 2.0) / np.cos(r_e_chi / 2.0))
    e_omg = np.round(np.rad2deg(r_k_omg + r_delta), 5)
    e_phi = np.round(np.rad2deg(r_k_phi + r_delta), 5)
    e_chi = np.round(np.rad2deg(r_e_chi), 5)
    return e_omg, e_phi, e_chi

def convert_to_sfrm(h5_path, h5_idx, frm_num, par):
    frm_name = '{}_{:02}_{:04}.sfrm'.format(os.path.join(par['frm_path'], par['frm_name']), par['frm_run'], frm_num)
    # Overwrite existing files if overwrite is flagged True
    if not par['overwrite'] and os.path.isfile(frm_name):
        return False
        
    # open the h5 file and sum the images in the given range
    with h5py.File(par['h5_file'], 'r') as h5f:
        frm_data = np.sum(h5f[h5_path][h5_idx:h5_idx+par['frm_sum'],:,:], axis=0)

    # set bad pixels to zero
    # this is only cosmetic but makes the indexing / visual analysis easier
    # as long as the xa mask files are used during integration
    # these pixels are masked and properly handled with
    frm_data[frm_data == par['det_max_counts'] * par['frm_sum']] = 0
    frm_data[frm_data < 0] = 0

    # scan parameters
    # increment, exposure time, start and end angle of the omega scan
    frm_idx = (frm_num - 1) * par['frm_sum']
    scn_inc = par['gon_scan_width'] * par['frm_sum']
    scn_exp = par['det_frame_time'] * par['frm_sum']
    scn_sta = par['gon_omega_start'][frm_idx]
    scn_end = par['gon_omega_end'][frm_idx + par['frm_sum'] - 1]

    # convert kappa to euler geometry
    omg_0, phi_0, chi_0 = kappa_to_euler(scn_sta, par['kappa'], par['alpha'], 0)
    omg_1, phi_1, chi_1 = kappa_to_euler(scn_end, par['kappa'], par['alpha'], 0)

    # angle correction
    phi_0 = 0.0 - phi_0
    phi_1 = 0.0 - phi_1
    omg_0 = omg_0 - 40.8
    omg_1 = omg_1 - 40.8

    # calculate detector pixel per cm
    # this is normalized to a 512x512 detector format
    # Eiger2 pixel size is 0.075 mm
    pix_per_512 = round((10.0 / (par['det_pxs']*1e3)) * (512.0 / frm_data.shape[0]), 6)
    
    # default bruker header
    header = init_bruker_header()
    
    # fill known header items
    header['NCOLS']      = [frm_data.shape[1], 4]                               # Number of pixels per row; number of mosaic tiles in X; dZ/dX
    header['NROWS']      = [frm_data.shape[0], 8]                               # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
    header['CENTER']     = [par['det_bc_x'], par['det_bc_y'],
                            par['det_bc_x'], par['det_bc_y']]                   # adjust the beam center for the filling/cutting of the frame
    header['DETTYPE']    = [par['det_name'], pix_per_512, 0.001, 0, 0.001, 0.001, 1]   # Detector type, pix512percm, cmtogrid, circular, brassspacing, windowthickness, accuratetiming
    header['DETPAR']     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                       # Detector position corrections (Xc,Yc,Dist,Pitch,Roll,Yaw)
    header['CCDPARM']    = [0.0, 1.0, 1.0, 0.0, (2**20)*par['frm_sum']]         # CCD parameters for computing pixel ESDs; readnoise, e/ADU, e/photon, bias, full scale
    header['SITE']       = par['src_facility_name']                             # Site name
    header['MODEL']      = par['src_beamline_name']                             # Diffractometer model
    header['TARGET']     = par['src_facility_type']                             # X-ray target material)
    header['USER']       = par['user_name']                                     # Username
    header['FILENAM']    = par['frm_name']                                      # (Original) frame filename
    header['RUN']        = par['frm_run']                                       # Run number within the data set
    header['SAMPLE']     = par['sample_name']                                   # Samplename
    header['SAMPNUM']    = 0                                                    # Specimen number within the data set
    header['SETNAME']    = '?'                                                  # Setname
    header['FILTER']     = 'Si(111) DCM (hor.), KB mirror pair (VFM, HFM)'      # Text describing filter/monochromator setting
    header['SOURCEK']    = '?'                                                  # X-ray source kV
    header['SOURCEM']    = '?'                                                  # Source milliamps
    header['WAVELEN']    = [par['src_wavelength'],par['src_wavelength'],par['src_wavelength']] # Wavelengths (average, a1, a2)
    header['CUMULAT']    = scn_exp                                              # Accumulated exposure time in real hours
    header['ELAPSDR']    = scn_exp                                              # Requested time for this frame in seconds
    header['ELAPSDA']    = scn_exp                                              # Actual time for this frame in seconds
    header['START']      = omg_0                                                # Starting scan angle value, decimal deg
    header['ANGLES']     = [0.0, omg_0, phi_0, chi_0]                           # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
    header['ENDING']     = [0.0, omg_1, phi_1, chi_1]                           # Setting angles read at end of scan
    header['TYPE']       = 'Generic Omega Scan'                                 # String indicating kind of data in the frame
    header['DISTANC']    = par['det_distance'] * 100                            # Sample-detector distance, cm
    header['RANGE']      = abs(scn_inc)                                         # Magnitude of scan range in decimal degrees
    header['INCREME']    = scn_inc                                              # Signed scan angle increment between frames
    header['NUMBER']     = frm_num                                              # Number of this frame in series (zero-based)
    header['NFRAMES']    = len(par['gon_omega_start'])                          # Number of frames in the series
    header['AXIS']       = 2                                                    # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    header['LOWTEMP']    = [1, int((-273.15 + 100.0) * 100.0), -6000]           # Low temp flag; experiment temperature*100; detector temp*100
    header['NEXP']       = [1, 0, 0, 0, 0]                                      # Number exposures in this frame; CCD bias level*100,;
                                                                                # Baseline offset (usually 32); CCD orientation; Overscan Flag
    header['MAXXY']      = np.vstack((frm_data == frm_data.max()).nonzero())[:,0] # X,Y pixel # of maximum counts
    header['MAXIMUM']    = np.max(frm_data)
    header['MINIMUM']    = np.min(frm_data)
    header['NCOUNTS']    = [frm_data.sum(), 0]
    header['NOVER64']    = frm_data[frm_data > 64000].shape[0]                  # Number of pixels > 64K
    header['PHD']        = [par['sensor_efficiency'], par['sensor_thickness']*1e2] # Phosphor efficiency, phosphor thickness [cm]
    ox = frm_data.shape[1]
    oy = frm_data.shape[0]
    header['OCTMASK']    = [0, 0, 0, ox-1, ox-1, ox+oy-1, oy-1, oy-1]           # Octagon mask parameters (GADDS) #min x, min x+y, min y, max x-y, max x, max x+y, max y, max y-x
    header['FILTER2']    = [90.0, 90.0, 0.0, 1.0]                               # Monochromator 2-theta (deg), roll (deg), beam tilt, attenuation factor
    # Monochromator 2-theta (deg) : Degree of polarisation, 0.0 unpolarised, 90.0 fully polarised
    # Monochromator roll (deg)    : Direction of the plane of polarization
    # Synchrotron, goniometer flat: 90.0 90
    # Synchrotron, goniometer side: 90.0  0
    # Lab-Source , mirror         :  0.0  0
    
    # write the frame
    write_bruker_frame(frm_name, header, frm_data)
    return True

def create_application():
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace) 
    app.setFont(font)
    app.setStyle("fusion")
    return app

if __name__ == '__main__':
    app = create_application()
    win = ConversionWindow()
    win.show()
    sys.exit(app.exec())