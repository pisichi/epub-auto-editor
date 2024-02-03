import os
import json

def get_cache_file_name(book_title: str, chapter_num: int) -> str:
    return f"{book_title}_cache_chapter_{chapter_num}.json"


def save_cache_to_file(cache_folder, book_title: str, chapter_num: int, paragraph_cache: dict) -> None:
    cache_file = os.path.join(
        cache_folder, book_title, get_cache_file_name(book_title, chapter_num))
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as cache_file:
        json.dump(paragraph_cache, cache_file, indent=2)


def load_cache_from_file(cache_folder, book_title: str, chapter_num: int) -> dict:
    cache_file = os.path.join(
        cache_folder, book_title, get_cache_file_name(book_title, chapter_num))
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as cache_file:
            return json.load(cache_file)
    return {}
