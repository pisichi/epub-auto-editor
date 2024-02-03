from config import Config


class CustomLogger:
    def __init__(self, config):
        self.config = config

    def print(self, message):
        if self.config.verbose_logging:
            print(message)
