def parse_txt(input_file: str) -> str:
    with open(input_file, 'r', encoding='utf-8') as txt_file:
        return txt_file.read()