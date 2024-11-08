"""This script reads documentation from /docs and puts it into zoo python files."""

import os
import re
from collections import defaultdict


def _get_python_file_name(env_type, env_name):
    dir_path = os.path.join("..", "..", "free_range_zoo", 'envs')
    for env_file in os.listdir(os.path.join(dir_path, env_type)):
        if env_file == env_name:
            with open(
                    os.path.join(dir_path, env_type, env_file, env_file + ".py"),
                    encoding="utf-8",
            ) as file:
                if env_name in file.name:
                    return file.name


def _insert_docstring_into_python_file(file_path, doc):
    new_docstring = f'"""\n{doc.strip()}\n"""'
    leading_docstring_pattern = re.compile(r'^\s*("""|\'\'\').*?\1', re.DOTALL)

    with open(file_path, "r+", encoding="utf-8") as file:
        file_text = file.read()

        match = leading_docstring_pattern.search(file_text)
        if match:
            start, end = match.span()
            file_text = file_text[:start] + new_docstring + file_text[end:]
        else:
            file_text = new_docstring + file_text

        file.seek(0)
        file.write(file_text)


def _insert_docstring_into_markdown_file(file_path, doc):
    with open(file_path, "r+", encoding="utf-8") as file:
        file.write(doc)


def _remove_front_matter(string):
    regex = re.compile(r"---\s*(\n|.)*?(---\s*\n)")
    match = regex.match(string)
    if match:
        g = match.group(0)
        return string[len(g):]
    else:
        return string


def _parse_markdown(string):
    parsed_content = defaultdict(str)

    # Regex to find headers and code blocks
    header_pattern = re.compile(r'^(#+) (.+)', re.MULTILINE)
    start_block_pattern = re.compile(r'```python')
    end_block_pattern = re.compile(r'```', re.DOTALL)

    last_header = None
    in_code_block = False

    for line in string.split('\n'):
        header = header_pattern.match(line)
        if header is not None and not in_code_block:
            last_header = header.group(2)

        is_start_code_block = start_block_pattern.match(line) is not None and not in_code_block
        is_end_code_block = end_block_pattern.match(line) is not None and in_code_block
        if is_start_code_block:
            in_code_block = True
        elif is_end_code_block:
            in_code_block = False

        parsed_content[last_header] += line + '\n'

    return dict(parsed_content)


def main():
    """Read the documentation from /docs and put them into the environment header files."""
    envs_dir = os.path.join("..", "free_range_zoo/envs")
    docs_dir = os.path.join("source")

    for env_name in os.listdir(envs_dir):
        dir_path = os.path.join(envs_dir, env_name)

        if not os.path.isdir(dir_path) or env_name == '__pycache__':
            continue

        environment_documentation = os.path.join(docs_dir, 'environments', f'{env_name}.md')

        environment_runtime_file = os.path.join(dir_path, 'env', f'{env_name}.py')
        environment_readme = os.path.join(dir_path, 'README.md')

        with open(environment_documentation, encoding="utf-8") as file:
            contents = file.read()

        contents = _remove_front_matter(contents)
        contents = _parse_markdown(contents)

        header_blacklist_python = ['Usage', 'Parallel API', 'AEC API', 'Configuration', 'API']
        header_blacklist_markdown = ['Configuration', 'API']

        header_text = ''
        for key in contents:
            if key not in header_blacklist_python:
                header_text += f'{contents[key]}'

        _insert_docstring_into_python_file(environment_runtime_file, header_text)

        header_text = ''
        for key in contents:
            if key not in header_blacklist_markdown:
                header_text += f'{contents[key]}'

        _insert_docstring_into_markdown_file(environment_readme, header_text)


if __name__ == "__main__":
    main()
