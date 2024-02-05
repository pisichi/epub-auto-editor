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
        self.merge_paragraphs = os.getenv("MERGE_PARAGRAPHS", False)
        self.min_paragraph_characters = int(
            os.getenv("MIN_PARAGRAPH_CHARACTERS", 50))
        self.max_paragraph_characters = int(
            os.getenv("MAX_PARAGRAPH_CHARACTERS", 150))
        self.rules = [
            # example for filtering out text using regex
            # (r'SCENE CHANGE', '\n-]|[-\n'),
            # (r'\b(?i)(?:an:|author[\'’]s? note:)\b.*', ''),
            # (r'Author Note.*', ''),
        ]
        self.cache_folder = os.getenv("CACHE_FOLDER", "cache")

    def update_from_args(self, args):
        self.input_folder = args.input or self.input_folder
        self.output_folder = args.output or self.output_folder
        self.llama_url = args.url or self.llama_url
        self.use_cache = not args.no_cache if args.no_cache is not None else self.use_cache
        self.verbose_logging = args.verbose or self.verbose_logging
        self.model_path = args.model_path or self.model_path
        self.merge_paragraphs = args.merge_paragraphs or self.merge_paragraphs
        self.min_paragraph_characters = args.min_char or self.min_paragraph_characters
        self.max_paragraph_characters = args.max_char or self.max_paragraph_characters
