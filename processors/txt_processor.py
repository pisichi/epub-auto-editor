import asyncio
from utils.http import post_request
from utils.text_processing import filter_text, visualize_differences

class TxtProcessor:
    def __init__(self, config):
        self.config = config