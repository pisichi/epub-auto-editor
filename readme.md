# EpubAutoEditor

EpubAutoEditor is a tool designed for automatically processing EPUB files, providing grammar correction, and enabling bulk editing of entire EPUB collections. <br>
It is designed to integrate with my custom gguf agent [gguf-py-backend](https://github.com/pisichi/gguf-py-backend).

## Requirements

- Python 3.x
- Dependencies (Install using `pip install -r requirements.txt`)

## Usage

1. Clone or download this repository.
3. Run the script: `python epub.py`


## Env

   ```env
   INPUT_FOLDER="input"
   OUTPUT_FOLDER="output"
   LLAMA_URL=""
   NO_CACHE=False
   VERBOSE=False
   ```


### Command-line Arguments
All args is optional, it'll has higher priority than env if provided.

- `-i, --input`: Input folder containing EPUB files (default: `input`).
- `-o, --output`: Output folder for processed EPUB files (default: `output`).
- `--url`: URL of the Llama agent (default: `http://localhost:8083/generate`).
- `--no-cache`: Disable caching (default is false).
- `--verbose`: Enable verbose logging.


# Example 1: Provide custom input and output folders
python epub.py -i custom_input_folder -o custom_output_folder

# Example 2: Use a custom Llama agent URL
python epub.py --url http://custom.llama-agent-url:8083/generate

# Example 3: Disable caching explicitly
python epub.py --no-cache

# Example 4: Enable verbose logging
python epub.py --verbose

