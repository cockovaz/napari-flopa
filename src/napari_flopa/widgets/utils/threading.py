import traceback

from qtpy.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(tuple)  # (exctype, value, traceback_str)
    finished = Signal()
    progress = Signal(str)


class Worker(QRunnable):
    """
    QRunnable-based background worker for use with QThreadPool.

    Usage:
        worker = Worker(fn, arg1, kwarg=val)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(on_error)
        worker.signals.finished.connect(on_finished)
        threadpool.start(worker)
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit((type(e), e, tb))
        finally:
            self.signals.finished.emit()
