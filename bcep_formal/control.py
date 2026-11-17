"""Request-only signals: checkpoint at an update/validation item boundary."""
import signal
_PAUSE=False

def request_pause(*_):
    global _PAUSE
    _PAUSE=True

def install():
    signal.signal(signal.SIGTERM,request_pause)
    signal.signal(signal.SIGINT,request_pause)

def pause_requested():return _PAUSE
