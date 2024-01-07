import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from datetime import datetime
import aiohttp
import asyncio

async def filter_text(sentence):
    # Implement your text filtering logic here
    # For example, let's remove all vowels from the sentence
    # filtered_sentence = re.sub('[aeiouAEIOU]', '', sentence)
    return sentence

async def send_to_llama_agent(session, input_text, max_tokens, retry_count=3, timeout=10):
    # Define the endpoint
    endpoint = 'http://localhost:8083/generate'

    # Retry logic
    for _ in range(retry_count + 1):
        try:
            # Send asynchronous HTTP POST request with a timeout
            async with session.post(endpoint, json={"input_text": input_text, "max_tokens": max_tokens}, timeout=timeout) as response:
                llama_response = await response.json()

                # Check for conditions to use the original sentence
                if response.status == 404 or len(llama_response.get("generated_text", "")) > timeout:
                    return input_text
                elif not llama_response.get("generated_text") or len(llama_response.get("generated_text")) < 5:
                    await asyncio.sleep(1)  # Wait for 1 second before retrying
                else:
                    return llama_response.get("generated_text")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Catch various errors, including network-related issues
            print(f"An error occurred: {e}")
            await asyncio.sleep(1)  # Wait for 1 second before retrying

    # Return the original sentence if retry count is exhausted
    return input_text

async def process_sentence(sentence, session):
    # Filter and send the sentence asynchronously
    filtered_sentence = await filter_text(sentence)
    llama_response = await send_to_llama_agent(session, filtered_sentence, max_tokens=32)

    return llama_response

async def process_chapter(chapter, session):
    # Extract text from the chapter
    chapter_content = chapter.get_content().decode('utf-8')

    # Use BeautifulSoup to parse HTML content
    soup = BeautifulSoup(chapter_content, 'html.parser')

    # Process each sentence in the chapter sequentially
    for sentence in soup.find_all('p'):
        # Process one sentence at a time
        modified_sentence = await process_sentence(sentence.text, session)
        chapter_content = chapter_content.replace(sentence.text, modified_sentence)

    # Update the chapter content
    chapter.set_content(chapter_content.encode('utf-8'))

async def save_to_text(book, output_text_file):
    text_content = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text_content += item.get_content().decode('utf-8')

    with open(output_text_file, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)

async def process_epub(input_file, output_folder):
    # Read the EPUB file
    book = epub.read_epub(input_file)

    # Create an aiohttp session
    async with aiohttp.ClientSession() as session:
        # Process each chapter in the book sequentially
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            await process_chapter(item, session)

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

    # Process all EPUB files in the input folder sequentially
    for filename in os.listdir(input_folder):
        if filename.endswith(".epub"):
            input_file = os.path.join(input_folder, filename)
            await process_epub(input_file, output_folder)

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"

    asyncio.run(process_all_epubs(input_folder, output_folder))
