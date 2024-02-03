from ebooklib import epub
from bs4 import BeautifulSoup

def parse_epub(input_file: str) -> epub.EpubBook:
    return epub.read_epub(input_file)