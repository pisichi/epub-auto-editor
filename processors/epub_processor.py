import asyncio
import os
from ebooklib import epub
from bs4 import BeautifulSoup
from datetime import datetime
import aiohttp
import asyncio
import difflib
import logging
from tqdm import tqdm
import aiohttp
from utils.cache import save_cache_to_file, load_cache_from_file
from utils.http import post_request
from utils.text_processing import filter_text, visualize_differences
from parsers.epub_parser import parse_epub
from utils.logger import CustomLogger


class EpubProcessor:
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.paragraph_cache = {}
        self.logger = CustomLogger(config)

    def decode_content(self, item):
        try:
            return item.content.decode('utf-8', errors='ignore')
        except UnicodeDecodeError as e:
            # Log the error for debugging
            self.logger.print(f"Error decoding content: {e}")
            return ""

    def is_content_chapter(self, item):
        return isinstance(item, epub.EpubItem) and "<p>" in self.decode_content(item)

    async def process_epub_file(self, input_file: str, output_folder: str):
        global book_title, total_chapters

        # Read the EPUB file
        async with aiohttp.ClientSession() as session:
            book = parse_epub(input_file)
            # Extract the book title from the EPUB metadata or use the file name
            book_title = book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else os.path.splitext(
                os.path.basename(input_file))[0]
            # Update the output folder for the specific book title
            book_output_folder = os.path.join(output_folder, book_title)
            os.makedirs(book_output_folder, exist_ok=True)

            total_chapters = sum(
                1 for item in book.items if self.is_content_chapter(item))

            chapter_items = [
                (i, item) for i, item in enumerate(book.items)
                if self.is_content_chapter(item)
            ]

            # Process each chapter in the book sequentially
            for i, item in chapter_items:
                await self.process_chapter(item, session, {}, book_title, output_folder, i)

            # Save the modified book to a new EPUB file
            output_epub_filename = f"{book_title}_{datetime.now().strftime('%Y%m%d')}.epub"
            output_epub_file = self.get_unique_filename(
                output_folder, book_title, output_epub_filename)

            epub.write_epub(output_epub_file, book)

            self.logger.print(
                f"EPUB processing complete. Output saved to {output_epub_file}")

    def get_unique_filename(self, folder, title, base_filename):
        index = 1
        while os.path.exists(os.path.join(folder, title, base_filename)):
            base_filename = f"{title}_{datetime.now().strftime('%Y%m%d')}_{index}.{'epub' if 'epub' in base_filename else 'txt'}"
            index += 1
        return os.path.join(folder, title, base_filename)

    async def process_chapter(self, chapter, session, progress_data, book_title, output_folder, chap_num):
        # Load the cache for the current chapter
        if self.config.use_cache:
            self.paragraph_cache = load_cache_from_file(
                self.config.cache_folder, book_title, chap_num)

        # Try decoding using utf-8
        chapter_content = self.decode_content(chapter)

        # Use BeautifulSoup to parse HTML content
        soup = BeautifulSoup(chapter_content, 'html.parser')

        # Get total paragraphs in the chapter
        paragraphs = soup.find_all(['p'])
        total_paragraphs = len(paragraphs)

        # Get the saved progress for the current chapter
        start_paragraph_index = progress_data.get(str(chapter), 0)

        if not self.config.verbose_logging:
            progress_bar_chapter = tqdm(
                total=total_paragraphs, desc=f"{book_title}: {chap_num}/{total_chapters} chapters processed.", dynamic_ncols=True)

        # Merge paragraphs if enabled
        if self.config.merge_paragraphs:
            # Code for merging paragraphs
            paragraphs, soup = self.merge_paragraphs(paragraphs, soup)

        # Process each paragraph in the chapter sequentially
        for i, paragraph in enumerate(paragraphs[start_paragraph_index:], start=start_paragraph_index):
            # Process one paragraph at a time
            self.logger.print(
                f"\n{book_title}: {chap_num}/{total_chapters} chapters processed.")
            self.logger.print(
                f"Chapter progress: {i + 1}/{total_paragraphs} paragraphs processed.\n")

            if not self.config.verbose_logging:
                progress_bar_chapter.update(1)

            processed_paragraph = await self.process_individual_paragraph(paragraph, session, chap_num)
            if self.config.merge_paragraphs:
                new_paragraph_tag = soup.new_tag('p')
                new_paragraph_tag.string = processed_paragraph
                soup.append(new_paragraph_tag)
            else:
                chapter_content = chapter_content.replace(
                    paragraph.get_text(), processed_paragraph)
            if self.config.use_cache:
                save_cache_to_file(self.config.cache_folder,
                                   book_title, chap_num, self.paragraph_cache)

        # Update the chapter content
        if self.config.merge_paragraphs:
            chapter.set_content(
                (u'<html><body><div>' + str(soup) + '</div></body></html>').encode('utf-8'))
        else:
            chapter.set_content(chapter_content.encode('utf-8'))

    def merge_paragraphs(self, paragraphs, soup):
        total_paragraphs = len(paragraphs)
        merged_paragraphs = []

        i = 0
        while i < total_paragraphs:
            current_paragraph = paragraphs[i].get_text().strip()

            while i < total_paragraphs - 1 and len(current_paragraph + " " + paragraphs[i + 1].get_text().strip()) <= self.config.min_paragraph_characters:
                current_paragraph += " " + paragraphs[i + 1].get_text().strip()
                i += 1

            merged_paragraphs.append(current_paragraph)
            i += 1

        for paragraph in paragraphs:
            paragraph.decompose()

        for merged_paragraph in merged_paragraphs:
            new_paragraph_tag = soup.new_tag('p')
            new_paragraph_tag.string = merged_paragraph
            soup.append(new_paragraph_tag)

        paragraphs = soup.find_all(['p'])
        soup.clear()
        return paragraphs, soup

    def merge_sentences(self, sentences):
        merged_paragraphs = []

        i = 0
        total_sentences = len(sentences)

        while i < total_sentences:
            current_sentence = sentences[i].strip()

            # Merge adjacent sentences until each paragraph has more than min_paragraph_characters
            while i < total_sentences - 1 and len(current_sentence + " " + sentences[i + 1].strip()) <= self.config.min_paragraph_characters:
                current_sentence += " " + sentences[i + 1].strip()
                i += 1

            # Append the merged paragraph to the new list
            merged_paragraphs.append(current_sentence)

            i += 1

        return merged_paragraphs

    async def process_paragraph(self, paragraph, session, chap_num):
        # Filter and send the paragraph asynchronously
        paragraph_text = paragraph.get_text() if isinstance(paragraph, dict) else paragraph

        filtered_paragraph = filter_text(paragraph_text, self.config.rules)

        if filtered_paragraph is None or not filtered_paragraph.strip():
            # Save the updated cache to file
            if self.config.use_cache:
                save_cache_to_file(self.config.cache_folder,
                                   book_title, chap_num, self.paragraph_cache)

            if self.config.verbose_logging:
                visualize_differences(paragraph_text, filtered_paragraph)

            return ""

        # Check if the filtered paragraph is already in the cache
        # if filtered_paragraph in paragraph_cache and paragraph_text != paragraph_cache[filtered_paragraph]:

        if filtered_paragraph in self.paragraph_cache:
            self.logger.print(f"Using cached paragraph.")
            # if verbose_logging:
            #     visualize_differences(paragraph_text, paragraph_cache[filtered_paragraph])
            return self.paragraph_cache[filtered_paragraph]

        # Check if the filtered paragraph is too short
        if len(filtered_paragraph.split()) <= 0:
            self.logger.print(
                f"Paragraph is too short. Using original paragraph: {paragraph_text}")
            return paragraph_text

        # Call the Llama agent and store the result in the cache
        llama_response = await self.send_to_llama_agent(session, filtered_paragraph, max_tokens=-1)
        self.paragraph_cache[filtered_paragraph] = llama_response

        # Save the updated cache to file
        if self.config.use_cache:
            save_cache_to_file(self.config.cache_folder,
                               book_title, chap_num, self.paragraph_cache)

        if self.config.verbose_logging:
            visualize_differences(paragraph_text, llama_response)

        # await asyncio.sleep(1.0)
        return llama_response

    async def process_individual_paragraph(self, paragraph, session, chap_num):
        if len(paragraph.get_text().split()) > self.config.max_paragraph_characters:
            return await self.process_large_paragraph(paragraph, session, chap_num)
        else:
            return await self.process_paragraph(paragraph.get_text(), session, chap_num)

    def split_paragraph_into_sentences(self, paragraph):
        sentences = []
        current_sentence = ""
        in_dialogue = False
        quote_char = None  # To keep track of the type of quotation marks

        for char in paragraph:
            current_sentence += char

            if char in ".!?":
                if not in_dialogue:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            elif char in ['"', '“', '”']:
                if not in_dialogue:
                    in_dialogue = True
                    quote_char = char
                elif in_dialogue and char == quote_char:
                    in_dialogue = False

        if current_sentence:
            sentences.append(current_sentence.strip())

        return sentences

    async def send_to_llama_agent(self, session, input_text, max_tokens, retry_count=3, timeout=5000):
        # Define the endpoint
        endpoint = self.config.llama_url

        for _ in range(retry_count + 1):
            try:
                # Use the local function or make an HTTP request
                llama_response = await self.model.generate(input_text, max_tokens) if self.model is not None else await self.get_llama_response(session, endpoint, input_text, max_tokens, timeout)

                # if llama_response is an object with output, else just use llama_response as is
                llama_response_output = llama_response.get(
                    "output", llama_response) if isinstance(llama_response, dict) else llama_response

                # Early return for invalid cases
                if not llama_response_output or len(llama_response_output) < 1:
                    self.logger.print(
                        f"Retrying with original sentence: {input_text}")
                    continue

                # Check the difference between input_text and llama_response using difflib
                diff = list(difflib.Differ().compare(
                    input_text.split(), llama_response_output.split()))

                # Calculate the number of added and removed tokens
                added_tokens = sum(1 for d in diff if d.startswith('+ '))
                removed_tokens = sum(1 for d in diff if d.startswith('- '))

                # Check if the token difference is too much
                if abs(added_tokens - removed_tokens) > 30:
                    if self.verbose_logging:
                        visualize_differences(
                            input_text, llama_response_output)
                    self.logger.print(
                        f"Token difference too much. using original input_text: {input_text}")
                    return input_text

                await asyncio.sleep(1.0)
                self.logger.print(f"\nGenerated sentence:")
                return llama_response_output

            except aiohttp.ClientError as client_error:
                # Handle specific client errors
                logging.warning(
                    f"Aiohttp client error occurred: {client_error}")
            except asyncio.TimeoutError:
                logging.warning(f"Asyncio timeout error occurred")
            except Exception as e:
                # Catch other exceptions
                logging.warning(f"An error occurred: {e}")

            await asyncio.sleep(0.5)

        # Return the original sentence if retry count is exhausted
        logging.warning(
            f"Retry count exhausted. Using original sentence: {input_text}")

        return input_text

    async def get_llama_response(session, endpoint, input_text, max_tokens, timeout):
        async with post_request(endpoint, json={"input_text": input_text, "max_tokens": max_tokens}, timeout=timeout) as response:
            return await response.json()

    async def process_large_paragraph(self, paragraph, session, chap_num):
        sentences = self.split_paragraph_into_sentences(paragraph.get_text())

        sentences = self.merge_sentences(sentences)

        new_paragraphs = []

        # Process each sentence separately asynchronously
        for sentence in sentences:
            processed_sentence = await self.process_paragraph(sentence, session, chap_num)
            self.logger.print(f"\n")
            new_paragraphs.append(processed_sentence)

        # Merge the processed sentences into a single result
        final_result = " ".join(new_paragraphs)

        # Return the final result
        return final_result

    async def process_all_epubs(self, input_folder: str, output_folder: str):
        # Ensure the output folder and cache folder exist
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.config.cache_folder, exist_ok=True)

        # Process all EPUB files in the input folder sequentially
        for filename in os.listdir(input_folder):
            if filename.endswith(".epub"):
                await self.process_epub_file(os.path.join(input_folder, filename), output_folder)

    def run(self):
        asyncio.run(self.process_all_epubs(
            self.config.input_folder, self.config.output_folder))
