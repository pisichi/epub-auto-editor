from ast import List, Tuple
import re
import difflib


def filter_text(sentence, rules):
    for pattern, replacement in rules:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence


def visualize_differences(original, modified) -> None:
    d = difflib.Differ()
    diff = list(d.compare(original.split(), modified.split()))
    for token in diff:
        if token.startswith(' '):
            print(token, end='')
        elif token.startswith('- '):
            print(f" \033[91m{token[2:]}\033[0m", end='')
        elif token.startswith('+ '):
            print(f" \033[92m{token[2:]}\033[0m", end='')
