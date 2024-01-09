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

book_title = "ebook"

CACHE_FOLDER = "cache"  # New cache folder

# Cache dictionary to store processed paragraphs
paragraph_cache = {}

total_chapters = 0
total_books = 0

# Function to generate a unique cache file name for each book
def get_cache_file_name(chapter_num):
    return f"{book_title}_cache_chapter_{chapter_num}.json"

# Function to save cache to file for a specific book
def save_cache_to_file(chapter_num):
    global paragraph_cache
    cache_file = os.path.join(CACHE_FOLDER, book_title, get_cache_file_name(chapter_num))
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as cache_file:
        json.dump(paragraph_cache, cache_file, indent=2)


# Function to load cache from file for a specific book
def load_cache_from_file(chapter_num):
    global paragraph_cache
    cache_file = os.path.join(CACHE_FOLDER, book_title, get_cache_file_name(chapter_num))
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as cache_file:
            paragraph_cache = json.load(cache_file)


async def filter_text(sentence):
    # Implement your text filtering logic here
    # For example, let's remove all vowels from the sentence
    # filtered_sentence = re.sub('[aeiouAEIOU]', '', sentence)
    return sentence


def visualize_differences(original, modified):
    d = difflib.Differ()
    diff = list(d.compare(original.split(), modified.split()))

    for token in diff:
        if token.startswith(' '):
            print(token, end='')
        elif token.startswith('- '):
            print(f" \033[91m{token[2:]}\033[0m", end='')  # Red color for deleted words
        elif token.startswith('+ '):
            print(f" \033[92m{token[2:]}\033[0m", end='')  # Green color for added words


async def send_to_llama_agent(session, input_text, max_tokens, retry_count=3, timeout=5000):
    # Define the endpoint
    endpoint = 'http://localhost:8083/generate'

    # Retry logic
    for _ in range(retry_count + 1):
        try:
            # Send asynchronous HTTP POST request with a timeout
            async with session.post(endpoint, json={"input_text": input_text, "max_tokens": max_tokens}, timeout=timeout) as response:
                llama_response = await response.json()

                # Check for conditions to use the original sentence
                if response.status == 404 or len(llama_response.get("output", "")) > timeout:
                    print(f"Original sentence used: {input_text}")
                    return input_text
                elif not llama_response.get("output") or len(llama_response.get("output")) < 5:
                    await asyncio.sleep(1)  # Wait for 1 second before retrying
                    print(f"Retrying with original sentence: {input_text}")
                else:
                    print(
                        f"Generated sentence:")
                    return llama_response.get("output")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Catch various errors, including network-related issues
            print(f"An error occurred: {e}")
            await asyncio.sleep(1)  # Wait for 1 second before retrying

    # Return the original sentence if retry count is exhausted
    print(f"Retry count exhausted. Using original sentence: {input_text}")
    return input_text


async def process_sentence(sentence, session):
    # Filter and send the sentence asynchronously
    filtered_sentence = await filter_text(sentence)
    llama_response = await send_to_llama_agent(session, filtered_sentence, max_tokens=32)

    return llama_response


async def process_paragraph(paragraph, session, chap_num):
    global paragraph_cache  # Use the global cache variable

    # Filter and send the paragraph asynchronously
    filtered_paragraph = await filter_text(paragraph.get_text())

    # Check if the filtered paragraph is already in the cache
    if filtered_paragraph in paragraph_cache:
        print(f"Using cached paragraph.")
        visualize_differences(paragraph.get_text(), paragraph_cache[filtered_paragraph])
        return paragraph_cache[filtered_paragraph]

    # Check if the filtered paragraph is too short
    if len(filtered_paragraph.split()) <= 5:
        print(
            f"Paragraph is too short. Using original paragraph: {paragraph.get_text()}")
        return paragraph.get_text()

    # Call the Llama agent and store the result in the cache
    llama_response = await send_to_llama_agent(session, filtered_paragraph, max_tokens=32)
    paragraph_cache[filtered_paragraph] = llama_response

    # Save the updated cache to file
    save_cache_to_file(chap_num)

    visualize_differences(paragraph.get_text(), llama_response)

    return llama_response


async def process_chapter(chapter, session, progress_data, book_title, output_folder, chap_num):
    global paragraph_cache  # Use the global cache variable
    paragraph_cache = {}

    # Load the cache for the current chapter
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
        print("Skipping chapter with title 'Information'")
        return

    # Get total paragraphs in the chapter
    paragraphs = soup.find_all(['p'])
    total_paragraphs = len(paragraphs)

    # Get the saved progress for the current chapter
    start_paragraph_index = progress_data.get(str(chapter), 0)

    # Process each paragraph in the chapter sequentially
    for i, paragraph in enumerate(paragraphs):
        if i < start_paragraph_index:
            continue

        # Process one paragraph at a time
        modified_paragraph = await process_paragraph(paragraph, session, chap_num)
        chapter_content = chapter_content.replace(
            paragraph.get_text(), modified_paragraph)
        
        # Save the updated cache to file for each paragraph
        save_cache_to_file(chap_num)

        # Print progress
        # print(
        #     f"\nJob progress: {i + 1}/{total_books} books processed.\n\n")
        print(
            f"\nBook progress: {chap_num + 1}/{total_chapters} chapters processed.")
        print(
            f"Chapter progress: {i + 1}/{total_paragraphs} paragraphs processed.\n")

    # Update the chapter content
    chapter.set_content(chapter_content.encode('utf-8'))


async def process_all_epubs(input_folder, output_folder):

    global paragraph_cache

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

                # Extract the book title from the EPUB metadata
                global book_title
                book_title = book.get_metadata("DC", "title")[0][0] if book.get_metadata(
                    "DC", "title") else "Untitled"
                
                # Update the output folder for the specific book title
                book_output_folder = os.path.join(output_folder, book_title)

                # Additional code to create a folder for each book title
                os.makedirs(book_output_folder, exist_ok=True)

                global total_chapters
                total_chapters = sum(1 for item in book.items if isinstance(item, ebooklib.epub.EpubItem) and re.match(r'^Text/.*', item.file_name))

                # Process each chapter in the book sequentially
                for i, item in enumerate(book.items):
                    if isinstance(item, ebooklib.epub.EpubItem):
                        # Check if the chapter file name is "Cover.xhtml" and skip if true
                        print(f"Chapter name'{item.file_name}'")
                        if re.match(r'^Text/.*', item.file_name):
                            await process_chapter(item, session, {}, book_title, output_folder, i)

                # Save the modified book to a new EPUB file
                output_epub_file = os.path.join(
                    output_folder, book_title, f"{book_title}_{datetime.now().strftime('%Y%m%d')}.epub")
                epub.write_epub(output_epub_file, book)
                print("EPUB processing complete. Output saved to", output_epub_file)

                # Save the modified content to a text file
                output_text_file = os.path.join(
                    output_folder, book_title, f"{book_title}_{datetime.now().strftime('%Y%m%d')}.txt")
                await save_to_text(book, output_text_file)
                print("Text content saved to", output_text_file)


async def save_to_text(book, output_text_file):
    text_content = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text_content += item.get_content().decode('utf-8')

    os.makedirs(os.path.dirname(output_text_file), exist_ok=True)

    with open(output_text_file, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"

    asyncio.run(process_all_epubs(input_folder, output_folder))