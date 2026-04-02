class ProgressLogger:
    """Simple logger that either prints or collects messages."""

    def __init__(self, mode: str = "print"):
        """
        Args:
            mode: 'print' to print to stdout, 'collect' to store in self.messages.
        """
        self.mode = mode
        self.messages = []

    def log(self, message: str):
        self.messages.append(message)
        if self.mode == "print":
            print(message)
