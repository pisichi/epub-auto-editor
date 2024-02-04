from dotenv import load_dotenv
import os


class Config:
    def __init__(self):
        load_dotenv()
        self.input_folder = os.getenv("INPUT_FOLDER", "input")
        self.output_folder = os.getenv("OUTPUT_FOLDER", "output")
        self.llama_url = os.getenv("LLAMA_URL", "")
        self.use_cache = not os.getenv("NO_CACHE", False)
        self.verbose_logging = os.getenv("VERBOSE", "False").lower() == "true"
        self.model_path = os.getenv("MODEL_PATH")
        # self.min_paragraph_characters = 300
        self.max_paragraph_characters = 2000
        self.rules = [
            # example for filtering out text using regex
            # (r'SCENE CHANGE', '\n-]|[-\n'),
            # (r'\b(?i)(?:an:|author[\'’]s? note:)\b.*', ''),
            # (r'Author Note.*', ''),
        ]
        self.CACHE_FOLDER = "cache"
