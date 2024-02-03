import argparse
import os
from config import Config
from parsers.epub_parser import parse_epub
from parsers.txt_parser import parse_txt
from processors.epub_processor import EpubProcessor
from processors.txt_processor import TxtProcessor
import asyncio
from model import Model


def parse_arguments() -> Config:
    parser = argparse.ArgumentParser(description="EPUB Processing Script")
    parser.add_argument("-i", "--input", type=str,
                        help="Input folder containing EPUB files")
    parser.add_argument("-o", "--output", type=str,
                        help="Output folder for processed EPUB files")
    parser.add_argument("--url", type=str,
                        help="URL of the Llama agent", default="")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching (default is false)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--model-path", type=str, help="Path to the model")
    args = parser.parse_args()

    config = Config()
    config.input_folder = args.input or config.input_folder
    config.output_folder = args.output or config.output_folder
    config.llama_url = args.url or config.llama_url
    config.use_cache = not args.no_cache if args.no_cache is not None else config.use_cache
    config.verbose_logging = args.verbose or config.verbose_logging
    config.model_path = args.model_path or config.model_path

    return config


async def main():
    config = parse_arguments()
    if config.model_path:
        model = Model(config.model_path)
    epub_processor = EpubProcessor(config, model)
    # txt_processor = TxtProcessor(config)

    # Process EPUB files
    for filename in os.listdir(config.input_folder):
        if filename.endswith(".epub"):
            await epub_processor.process_epub_file(os.path.join(
                config.input_folder, filename), config.output_folder)

    # # Process TXT files
    # for filename in os.listdir(config.input_folder):
    #     if filename.endswith(".txt"):
    #         txt_processor.process_txt_file(os.path.join(
    #             config.input_folder, filename), config.output_folder)


if __name__ == "__main__":
    asyncio.run(main())
