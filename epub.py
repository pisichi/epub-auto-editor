import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from datetime import datetime
import aiohttp
import asyncio
import json

PROGRESS_FILE = "progress.json"

async def save_progress(progress_data):
    with open(PROGRESS_FILE, 'w') as progress_file:
        json.dump(progress_data, progress_file)

async def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as progress_file:
            return json.load(progress_file)
    return {}

async def filter_text(sentence):
    # Implement your text filtering logic here
    # For example, let's remove all vowels from the sentence
    # filtered_sentence = re.sub('[aeiouAEIOU]', '', sentence)
    return sentence

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
                    print(f"Generated sentence: {llama_response.get('output')}")
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

async def process_paragraph(paragraph, session):
    # Filter and send the paragraph asynchronously
    filtered_paragraph = await filter_text(paragraph.text)

    # Check if the filtered paragraph is too short
    if len(filtered_paragraph.split()) <= 2:
        print(f"Paragraph is too short. Using original paragraph: {paragraph.text}")
        return paragraph.text

    llama_response = await send_to_llama_agent(session, filtered_paragraph, max_tokens=32)
    return llama_response

async def process_chapter(chapter, session, progress_data):
    # Extract text from the chapter
    chapter_content = chapter.get_content().decode('utf-8')

    # Use BeautifulSoup to parse HTML content
    soup = BeautifulSoup(chapter_content, 'html.parser')

    # Check if the chapter has the title "Information"
    title_tag = soup.find('title')
    if title_tag and title_tag.text.strip().lower() == 'information':
        print("Skipping chapter with title 'Information'")
        return

    # Get total paragraphs in the chapter
    total_paragraphs = len(soup.find_all('p'))

    # Get the saved progress for the current chapter
    start_paragraph_index = progress_data.get(str(chapter), 0)

    # Process each paragraph in the chapter sequentially
    for i, paragraph in enumerate(soup.find_all('p')):
        if i < start_paragraph_index:
            continue

        # Process one paragraph at a time
        modified_paragraph = await process_paragraph(paragraph, session)
        chapter_content = chapter_content.replace(paragraph.text, modified_paragraph)

        # Print progress
        print(f"Chapter progress: {i + 1}/{total_paragraphs} paragraphs processed.")

        # Save progress after processing each paragraph
        progress_data[str(chapter)] = i + 1
        await save_progress(progress_data)

    # Update the chapter content
    chapter.set_content(chapter_content.encode('utf-8'))

async def save_to_text(book, output_text_file):
    text_content = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text_content += item.get_content().decode('utf-8')

    with open(output_text_file, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)

async def process_epub(input_file, output_folder, progress_data):
    # Read the EPUB file
    book = epub.read_epub(input_file)

    # Create an aiohttp session
    async with aiohttp.ClientSession() as session:
        # Process each chapter in the book sequentially
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            await process_chapter(item, session, progress_data)

        # Generate the output file path for EPUB
        output_epub_file = os.path.join(output_folder, os.path.basename(input_file).replace('.epub', f'_{datetime.now().strftime("%Y%m%d")}.epub'))

        # Save the modified book to a new EPUB file
        epub.write_epub(output_epub_file, book)
        print("EPUB processing complete. Output saved to", output_epub_file)

        # Generate the output file path for text
        output_text_file = os.path.join(output_folder, os.path.basename(input_file).replace('.epub', f'_{datetime.now().strftime("%Y%m%d")}.txt'))

        # Save the modified content to a text file
        await save_to_text(book, output_text_file)
        print("Text content saved to", output_text_file)

async def process_all_epubs(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load progress data
    progress_data = await load_progress()

    # Process all EPUB files in the input folder sequentially
    for filename in os.listdir(input_folder):
        if filename.endswith(".epub"):
            input_file = os.path.join(input_folder, filename)
            await process_epub(input_file, output_folder, progress_data)

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"

    asyncio.run(process_all_epubs(input_folder, output_folder))
