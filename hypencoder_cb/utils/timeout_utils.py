# timeout utility used for making the trainer shutdown while it is 
# training, to test configs that might cause a data hang issue

# This timeout utility will shut down the trainer after a particular
# amount of time so it will shutdown the training regardless of 
# whether any problems happened or not.

# The timeout period is defined inside the train.py file


import signal

class TimeoutException(Exception):
    """Custom exception to be raised on a timeout."""
    pass

class timeout:
    def __init__(self, seconds=60, error_message="Timeout after {} seconds"):
        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("The timeout context manager is not supported on this OS.")
        self.seconds = seconds
        self.error_message = error_message.format(seconds)

    def _handle_timeout(self, signum, frame):
        raise TimeoutException(self.error_message)

    def __enter__(self):
        # Set the signal handler and the alarm
        self.old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        # Disable the alarm
        signal.alarm(0)
        # Restore the original signal handler
        signal.signal(signal.SIGALRM, self.old_handler)