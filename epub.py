import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from datetime import datetime

def filter_text(sentence):
    # Implement your text filtering logic here
    # For example, let's remove all vowels from the sentence
    filtered_sentence = re.sub('[aeiouAEIOU]', '', sentence)
    print(sentence)
    return filtered_sentence

def process_chapter(chapter):
    # Extract text from the chapter
    chapter_content = chapter.get_content().decode('utf-8')
    
    # Use BeautifulSoup to parse HTML content
    soup = BeautifulSoup(chapter_content, 'html.parser')

    # Get all sentences in the chapter
    sentences = [sentence.text for sentence in soup.find_all('p')]

    # Filter each sentence
    filtered_sentences = [filter_text(sentence) for sentence in sentences]

    # Replace the original sentences with the filtered ones
    for orig_sentence, filtered_sentence in zip(sentences, filtered_sentences):
        chapter_content = chapter_content.replace(orig_sentence, filtered_sentence)

    # Update the chapter content
    chapter.set_content(chapter_content.encode('utf-8'))

def save_to_text(book, output_text_file):
    text_content = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text_content += item.get_content().decode('utf-8')

    with open(output_text_file, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)

def process_epub(input_file, output_folder):
    # Read the EPUB file
    book = epub.read_epub(input_file)

    # Process each chapter in the book
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        process_chapter(item)

    # Generate the output file path for EPUB
    output_epub_file = os.path.join(output_folder, os.path.basename(input_file).replace('.epub', f'_{datetime.now().strftime("%Y%m%d")}.epub'))

    # Save the modified book to a new EPUB file
    epub.write_epub(output_epub_file, book)
    print("EPUB processing complete. Output saved to", output_epub_file)

    # Generate the output file path for text
    output_text_file = os.path.join(output_folder, os.path.basename(input_file).replace('.epub', f'_{datetime.now().strftime("%Y%m%d")}.txt'))

    # Save the modified content to a text file
    save_to_text(book, output_text_file)
    print("Text content saved to", output_text_file)

def process_all_epubs(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all EPUB files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".epub"):
            input_file = os.path.join(input_folder, filename)
            process_epub(input_file, output_folder)

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"

    process_all_epubs(input_folder, output_folder)
