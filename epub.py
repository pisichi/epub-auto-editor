import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from datetime import datetime
import aiohttp
import asyncio
import json
import difflib
import chardet
import argparse
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from model import Model

# Load environment variables from the .env file
load_dotenv()

# Constants for default values
DEFAULT_INPUT_FOLDER = "input"
DEFAULT_OUTPUT_FOLDER = "output"
CACHE_FOLDER = "cache"  # cache folder
DEFAULT_LLAMA_URL = ""

# Parse command-line arguments
parser = argparse.ArgumentParser(description="EPUB Processing Script")
parser.add_argument("-i", "--input", type=str,
                    help="Input folder containing EPUB files")
parser.add_argument("-o", "--output", type=str,
                    help="Output folder for processed EPUB files")
parser.add_argument("--url", type=str,
                    help="URL of the Llama agent", default=DEFAULT_LLAMA_URL)
parser.add_argument("--no-cache", action="store_true",
                    help="Disable caching (default is false)")
parser.add_argument("--verbose", action="store_true",
                    help="Enable verbose logging")
parser.add_argument("--model-path", type=str,
                    help="Path to the model", default=os.getenv("MODEL_PATH"))
args = parser.parse_args()

# Read parameters from environment variables if not provided via command line
input_folder = args.input or os.getenv("INPUT_FOLDER", DEFAULT_INPUT_FOLDER)
output_folder = args.output or os.getenv(
    "OUTPUT_FOLDER", DEFAULT_OUTPUT_FOLDER)
llama_url = args.url or os.getenv("LLAMA_URL", DEFAULT_LLAMA_URL)
use_cache = not args.no_cache if args.no_cache is not None else not os.getenv(
    "NO_CACHE", True)
verbose_logging = args.verbose or (
    os.getenv("VERBOSE", "False").lower() == "true")
model_path = args.model_path or os.getenv("MODEL_PATH")

book_title = "ebook"

# Cache dictionary to store processed paragraphs
paragraph_cache = {}

total_chapters = 0
total_books = 0

if os.getenv("MODEL_PATH"):
    model = Model(model_path)

# Function to generate a unique cache file name for each book


def get_cache_file_name(chapter_num):
    return f"{book_title}_cache_chapter_{chapter_num}.json"

# Function to save cache to file for a specific book


def save_cache_to_file(chapter_num):
    global paragraph_cache
    cache_file = os.path.join(CACHE_FOLDER, book_title,
                              get_cache_file_name(chapter_num))
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as cache_file:
        json.dump(paragraph_cache, cache_file, indent=2)


# Function to load cache from file for a specific book
def load_cache_from_file(chapter_num):
    global paragraph_cache
    cache_file = os.path.join(CACHE_FOLDER, book_title,
                              get_cache_file_name(chapter_num))
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as cache_file:
            paragraph_cache = json.load(cache_file)


async def filter_text(sentence):
    # TODO
    return sentence


def visualize_differences(original, modified):
    d = difflib.Differ()
    diff = list(d.compare(original.split(), modified.split()))

    for token in diff:
        if token.startswith(' '):
            print(token, end='')
        elif token.startswith('- '):
            # Red color for deleted words
            print(f" \033[91m{token[2:]}\033[0m", end='')
        elif token.startswith('+ '):
            # Green color for added words
            print(f" \033[92m{token[2:]}\033[0m", end='')


async def send_to_llama_agent(session, input_text, max_tokens, retry_count=3, timeout=5000):
    # Define the endpoint
    endpoint = llama_url

    for _ in range(retry_count + 1):
        try:
            # Use the local function or make an HTTP request
            llama_response = await model.generate(input_text, max_tokens) if os.getenv("MODEL_PATH") else await get_llama_response(session, endpoint, input_text, max_tokens, timeout)

            # if llama_response is an object with output, else just use llama_response as is
            llama_response_output = llama_response.get(
                "output", llama_response) if isinstance(llama_response, dict) else llama_response

            # Early return for invalid cases
            if not llama_response_output or len(llama_response_output) < 1:
                custom_print(f"Retrying with original sentence: {input_text}")
                continue

            # Check the difference between input_text and llama_response using difflib
            diff = list(difflib.Differ().compare(
                input_text.split(), llama_response_output.split()))

            # Calculate the number of added and removed tokens
            added_tokens = sum(1 for d in diff if d.startswith('+ '))
            removed_tokens = sum(1 for d in diff if d.startswith('- '))

            # Check if the token difference is too much
            if abs(added_tokens - removed_tokens) > 30:
                modified_input_text = input_text + "."
                if verbose_logging:
                    visualize_differences(input_text, llama_response_output)
                custom_print(
                    f"Token difference too much. using original input_text: {input_text}")
                return input_text

            custom_print(f"Generated sentence:")
            return llama_response_output

        except aiohttp.ClientError as client_error:
            # Handle specific client errors
            custom_print(f"Aiohttp client error occurred: {client_error}")
        except asyncio.TimeoutError:
            custom_print(f"Asyncio timeout error occurred")
        except Exception as e:
            # Catch other exceptions
            custom_print(f"An error occurred: {e}")

        # await asyncio.sleep(0.3)  # Wait for 1 second before retrying

    # Return the original sentence if retry count is exhausted
    logging.warning(
        f"Retry count exhausted. Using original sentence: {input_text}")
    return input_text


async def get_llama_response(session, endpoint, input_text, max_tokens, timeout):
    async with session.post(endpoint, json={"input_text": input_text, "max_tokens": max_tokens}, timeout=timeout) as response:
        return await response.json()


async def process_sentence(sentence, session):
    # Filter and send the sentence asynchronously
    filtered_sentence = await filter_text(sentence)

    if len(filtered_sentence.split()) <= 3:
        custom_print(
            f"Paragraph is too short. Using original paragraph: {filtered_sentence}")
        return filtered_sentence

    endpoint = llama_url

    llama_response = await model.generate(filtered_sentence, -1) if os.getenv("MODEL_PATH") else await get_llama_response(session, endpoint, filtered_sentence, -1, 5000)

    llama_response = llama_response.get("output", llama_response) if isinstance(
        llama_response, dict) else llama_response

    visualize_differences(filtered_sentence, llama_response)
    print(f"\n")

    return llama_response


def split_paragraph_into_sentences(paragraph):
    sentences = []
    current_sentence = ""
    for char in paragraph:
        current_sentence += char
        if char in ".!?":
            sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence:
        sentences.append(current_sentence.strip())
    return sentences


async def process_paragraph(paragraph, session, chap_num):
    global paragraph_cache  # Use the global cache variable
    # Filter and send the paragraph asynchronously
    paragraph_text = paragraph.get_text() if isinstance(paragraph, dict) else paragraph

    filtered_paragraph = await filter_text(paragraph_text)

    # Check if the filtered paragraph is already in the cache
    # if filtered_paragraph in paragraph_cache and paragraph_text != paragraph_cache[filtered_paragraph]:

    if filtered_paragraph in paragraph_cache:
        custom_print(f"Using cached paragraph.")
        # if verbose_logging:
        #     visualize_differences(paragraph_text, paragraph_cache[filtered_paragraph])
        return paragraph_cache[filtered_paragraph]

    # Check if the filtered paragraph is too short
    if len(filtered_paragraph.split()) <= 8:
        custom_print(
            f"Paragraph is too short. Using original paragraph: {paragraph_text}")
        return paragraph_text

    # Call the Llama agent and store the result in the cache
    llama_response = await send_to_llama_agent(session, filtered_paragraph, max_tokens=-1)
    paragraph_cache[filtered_paragraph] = llama_response

    # Save the updated cache to file
    if use_cache:
        save_cache_to_file(chap_num)

    if verbose_logging:
        visualize_differences(paragraph_text, llama_response)

    return llama_response


async def process_chapter(chapter, session, progress_data, book_title, output_folder, chap_num):
    global paragraph_cache  # Use the global cache variable
    paragraph_cache = {}

    # Load the cache for the current chapter
    if use_cache:
        load_cache_from_file(chap_num)

    try:
        # Try decoding using utf-8
        chapter_content = chapter.get_content().decode('utf-8')
    except UnicodeDecodeError:
        # If decoding as utf-8 fails, try to detect the encoding
        encoding_detection_result = chardet.detect(chapter.get_content())
        detected_encoding = encoding_detection_result.get('encoding')

        if detected_encoding:
            chapter_content = chapter.get_content().decode(detected_encoding)
        else:
            # If detection fails, use a default encoding (e.g., 'latin-1')
            chapter_content = chapter.get_content().decode('latin-1')

    # Use BeautifulSoup to parse HTML content
    soup = BeautifulSoup(chapter_content, 'html.parser')

    # Check if the chapter has the title "Information"
    title_tag = soup.find('title')
    if title_tag and title_tag.text.strip().lower() == 'information':
        custom_print("Skipping chapter with title 'Information'")
        return

    # Get total paragraphs in the chapter
    paragraphs = soup.find_all(['p'])
    total_paragraphs = len(paragraphs)

    # Get the saved progress for the current chapter
    start_paragraph_index = progress_data.get(str(chapter), 0)

    if not verbose_logging:
        progress_bar_chapter = tqdm(
            total=total_paragraphs, desc=f"Book progress: {chap_num + 1}/{total_chapters} chapters processed.", dynamic_ncols=True)

    # Process each paragraph in the chapter sequentially
    for i, paragraph in enumerate(paragraphs[start_paragraph_index:], start=start_paragraph_index):
        # Process one paragraph at a time
        if len(paragraph.get_text().split()) > 300:
            sentences = split_paragraph_into_sentences(paragraph.get_text())

            # Group sentences into new paragraphs, each containing approximately 500 words
            words_count = 0
            new_paragraphs = []
            current_paragraph = ""
            for sentence in sentences:
                if words_count + len(sentence.split()) <= 300:
                    current_paragraph += sentence + " "
                    words_count += len(sentence.split())
                else:
                    new_paragraphs.append(current_paragraph.strip())
                    current_paragraph = sentence + " "
                    words_count = len(sentence.split())

            if current_paragraph:
                new_paragraphs.append(current_paragraph.strip())

            # Process each new paragraph separately
            for new_paragraph in new_paragraphs:
                modified_paragraph = await process_paragraph(new_paragraph, session, chap_num)
                chapter_content = chapter_content.replace(
                    new_paragraph, modified_paragraph)
        else:
            # Process normal paragraphs
            modified_paragraph = await process_paragraph(paragraph.get_text(), session, chap_num)
            chapter_content = chapter_content.replace(
                paragraph.get_text(), modified_paragraph)

        # Save the updated cache to file for each paragraph
        if use_cache:
            save_cache_to_file(chap_num)

        # Print progress
        custom_print(
            f"\nBook progress: {chap_num + 1}/{total_chapters} chapters processed.")
        custom_print(
            f"Chapter progress: {i + 1}/{total_paragraphs} paragraphs processed.\n")
        if not verbose_logging:
            progress_bar_chapter.update(1)

    # Update the chapter content
    chapter.set_content(chapter_content.encode('utf-8'))


async def process_all_epubs(input_folder, output_folder):

    # Ensure the output folder and cache folder exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(CACHE_FOLDER, exist_ok=True)

    # Process all EPUB files in the input folder sequentially
    for filename in os.listdir(input_folder):
        if filename.endswith(".epub"):
            input_file = os.path.join(input_folder, filename)

            # Create an aiohttp session
            async with aiohttp.ClientSession() as session:
                # Read the EPUB file
                book = epub.read_epub(input_file)

                # Extract the book title from the EPUB metadata or use the file name
                global book_title
                book_title = book.get_metadata("DC", "title")[0][0] if book.get_metadata(
                    "DC", "title") else os.path.splitext(os.path.basename(input_file))[0]

                # Update the output folder for the specific book title
                book_output_folder = os.path.join(output_folder, book_title)
                os.makedirs(book_output_folder, exist_ok=True)

                global total_chapters
                total_chapters = total_chapters = sum(
                    1 for item in book.items
                    if isinstance(item, ebooklib.epub.EpubItem) and "<p>" in item.content.decode('utf-8', errors='ignore')
                )

                # Process each chapter in the book sequentially
                for i, item in enumerate(book.items):
                    if isinstance(item, ebooklib.epub.EpubItem):
                        # Read the chapter content
                        chapter_content = item.content.decode('utf-8', errors='ignore')  # Decode the content


                        # Check if the detected language is English and confidence is high

                        if "<p>" in chapter_content:
                            custom_print(f"Chapter name '{item.file_name}'")
                            await process_chapter(item, session, {}, book_title, output_folder, i)

                # Save the modified book to a new EPUB file
                output_epub_filename = f"{book_title}_{datetime.now().strftime('%Y%m%d')}.epub"
                output_epub_file = get_unique_filename(
                    output_folder, book_title, output_epub_filename)
                epub.write_epub(output_epub_file, book)
                custom_print(
                    f"EPUB processing complete. Output saved to {output_epub_file}")

                # Save the modified content to a text file
                output_text_filename = f"{book_title}_{datetime.now().strftime('%Y%m%d')}.txt"
                output_text_file = get_unique_filename(
                    output_folder, book_title, output_text_filename)
                await save_to_text(book, output_text_file)
                custom_print(f"Text content saved to {output_text_file}")


def get_unique_filename(folder, title, base_filename):
    index = 1
    while os.path.exists(os.path.join(folder, title, base_filename)):
        base_filename = f"{title}_{datetime.now().strftime('%Y%m%d')}_{index}.{'epub' if 'epub' in base_filename else 'txt'}"
        index += 1
    return os.path.join(folder, title, base_filename)


async def process_raw_text(input_text, session, output_text_file):
    # Split raw text into sentences based on spaces between paragraphs
    sentences = re.split(r'\s\s+', input_text)

    # Process each sentence asynchronously
    os.makedirs(os.path.dirname(output_text_file), exist_ok=True)

    processed_sentences = []
    for sentence in sentences:
        modified_sentence = await process_sentence(sentence, session)
        processed_sentences.append(modified_sentence)
        with open(output_text_file, 'w', encoding='utf-8') as output_file:
            modified_text = '\n\n\n'.join(processed_sentences)
            output_file.write(modified_text)

    return "complete"


async def save_to_text(book, output_text_file):
    text_content = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text_content += item.get_content().decode('utf-8')

    os.makedirs(os.path.dirname(output_text_file), exist_ok=True)

    with open(output_text_file, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)


async def process_all_txt(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all TXT files in the input folder sequentially
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)

            # Create an aiohttp session
            async with aiohttp.ClientSession() as session:
                # Read the raw text file
                with open(input_file, 'r', encoding='utf-8') as txt_file:
                    raw_text = txt_file.read()

                # Process the raw text and save modified text to a new file
                output_txt_filename = f"{os.path.splitext(filename)[0]}_modified_{datetime.now().strftime('%Y%m%d')}.txt"
                output_txt_file = os.path.join(
                    output_folder, output_txt_filename)
                modified_text = await process_raw_text(raw_text, session, output_txt_file)

                custom_print(
                    f"TXT processing complete. Output saved to {output_txt_file}")


def custom_print(message):
    if verbose_logging:
        print(message)


def main():
    asyncio.run(process_all_epubs(input_folder, output_folder))
    # asyncio.run(process_all_txt(input_folder, output_folder))


if __name__ == "__main__":
    main()
