import multiprocessing as mp
from time import sleep
from PyQt6 import QtCore, QtGui, QtWidgets

class ConversionWindow(QtWidgets.QMainWindow):
    process_finished = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__()
        # variables
        self.title = kwargs.get('title', 'None')
        self.header_ext = kwargs.get('header_ext', [])
        self.header_val = kwargs.get('header_val', [])
        self.fn_suffix = kwargs.get('fn_suffix', '_master.h5')
        self.fc_prepare = kwargs.get('fc_prepare', None)
        self.fc_process = kwargs.get('fc_process', None)
        # check prepare function
        if self.fc_prepare is None:
            print('Prepare function is None')
            raise SystemExit
        # check processing function
        if self.fc_process is None:
            print('Processing function is None')
            raise SystemExit
        # internals
        self.button_group_rem = QtWidgets.QButtonGroup()
        self.thread_pool = QtCore.QThreadPool()
        self.flag_stop = False
        self.progress = 0
        self.mp_timeout = 1
        self.header = ['Path', 'Progress', 'Delete']
        for item in self.header_ext[::-1]:
            self.header.insert(1, item)
        self.header_len = len(self.header)
        self.col_num_progress = self.header_len-2
        self.col_num_delete = self.header_len-1

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
        self.table.setColumnCount(self.header_len)
        self.table.setHorizontalHeaderLabels(self.header)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for i in range(1, self.header_len):
            self.table.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        progress_bar = ProgressDelegate(self.table)
        self.table.setItemDelegateForColumn(self.col_num_progress, progress_bar)
        layout.addWidget(self.table)

        # add start and stop buttons
        self.btn_box = QtWidgets.QGroupBox()
        self.btn_box_layout = QtWidgets.QHBoxLayout()
        self.btn_box.setLayout(self.btn_box_layout)
        self.btn_conv = QtWidgets.QPushButton('Start')
        self.btn_conv.clicked.connect(self.on_run_clicked)
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet('QPushButton:enabled{background-color: red;}')
        self.btn_stop.clicked.connect(self.thread_stop)
        self.rdo_overwrite = QtWidgets.QRadioButton('Overwrite')
        self.rdo_overwrite.setChecked(False)
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
                if file_path and file_path.endswith(self.fn_suffix):
                    self.table.setRowCount(self.table.rowCount() + 1)
                    self.table.setItem(self.table.rowCount()-1, 0, QtWidgets.QTableWidgetItem(file_path))
                    if self.header_len > 3:
                        for i, (op, val) in enumerate(self.header_val):
                            if op == 'add':
                                entry = counter+val
                            elif op == 'mul':
                                entry = counter*val
                            else:
                                entry = val
                            self.table.setItem(self.table.rowCount()-1, i+1, QtWidgets.QTableWidgetItem(str(entry)))
                    # progress bar
                    bar = QtWidgets.QTableWidgetItem()
                    bar.setData(QtCore.Qt.ItemDataRole.UserRole+1000, 0)
                    item = QtWidgets.QTableWidgetItem(bar)
                    item.setFlags(~QtCore.Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(self.table.rowCount()-1, self.col_num_progress, item)
                    # remove button
                    button_rem = QtWidgets.QToolButton()
                    button_rem.setText('X')
                    button_rem.clicked.connect(self.table_delete_row)        
                    self.button_group_rem.addButton(button_rem)
                    self.table.setIndexWidget(self.table.model().index(self.table.rowCount()-1, self.col_num_delete), button_rem)
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
        runlist = []
        for row in range(self.table.rowCount()):
            entries = []
            for col in range(self.col_num_progress):
                item = self.table.item(row, col)
                entries.append(item.text())
            runlist.append(tuple(entries))
        return runlist

    def on_run_clicked(self):
        # construct runlist from table
        # iteerate over runs
        # disable buttons
        self.buttons_enable(False)
        for idx, run in enumerate(self.make_runlist()):
            fn_iter, fn_args, total = self.fc_prepare(self, *run)
            self.worker = Worker(self.thread_run, [fn_iter, fn_args, total, idx], {})
            self.worker.signals.finished.connect(self.thread_finished)
            self.worker.signals.update.connect(self.thread_update)
            self.thread_pool.start(self.worker)
            # wait until done
            loop = QtCore.QEventLoop()
            self.process_finished.connect(loop.quit)
            loop.exec()
            if self.flag_stop:
                break
        # enable buttons
        self.buttons_enable(True)
        # reset stop button
        self.flag_stop = False

    def buttons_enable(self, toggle=False):
        self.btn_stop.setEnabled(not toggle)
        self.btn_conv.setEnabled(toggle)
        for btn in self.button_group_rem.buttons():
            btn.setEnabled(toggle)

    def thread_update(self, row, value):
        self.processed.append(value)
        self.progress = min(max(int(len(self.processed) / self.progress_total * 100), 1), 99)
        self.table.item(row, self.col_num_progress).setData(QtCore.Qt.ItemDataRole.UserRole+1000, self.progress)

    def thread_finished(self, row, success):
        if success:
            # any progress value above max (100) triggers green coloring of progress bar 
            self.table.item(row, self.col_num_progress).setData(QtCore.Qt.ItemDataRole.UserRole+1000, 999)
        else:
            # negative progress triggers red coloring of progress bar 
            self.table.item(row, self.col_num_progress).setData(QtCore.Qt.ItemDataRole.UserRole+1000, -self.progress)
        self.thread_cleanup()

    def thread_cleanup(self):
        # wait for pool to finish
        self.thread_pool.clear()
        self.thread_pool.waitForDone()
        self.process_finished.emit()

    def thread_stop(self):
        self.flag_stop = True

    def thread_run(self, fn_iter, fn_args, total, row):
        self.progress_total = total
        self.processed = []
        with mp.Pool() as self.pool:
            idx_write = 0
            for h5_link, h5_iter in fn_iter:
                for idx_read in h5_iter:
                    idx_write += 1
                    self.pool.apply_async(self.fc_process,
                                          args=(h5_link, idx_read, idx_write, fn_args),
                                          callback=lambda x: self.worker.signals.update.emit(row, x))
            self.pool.close()
            # check for stop
            while len(self.processed) < self.progress_total:
                if self.flag_stop:
                    self.pool.terminate()
                    self.pool.join()
                    return row, False
                sleep(self.mp_timeout)
            # wait for the pool to finish
            self.pool.join()
            return row, True

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
        self.signals.finished.emit(*self.funct(*self.args, **self.kwargs))
    
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
        opt.progress = min(abs(progress), opt.maximum)
        opt.text = f'{opt.progress}%'
        opt.textVisible = True
        if progress > opt.maximum:
            opt.palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(93, 156, 89))
        elif progress < opt.minimum:
            opt.palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(223, 46, 56))
        QtWidgets.QApplication.style().drawControl(QtWidgets.QStyle.ControlElement.CE_ProgressBar, opt, painter)
