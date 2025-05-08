import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    original_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = original_stderr
        devnull.close()