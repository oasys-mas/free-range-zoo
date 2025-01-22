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
    new_docstring = f'"""\n{doc.strip()}\n"""\n'
    leading_docstring_pattern = re.compile(r'^\s*(("""|\'\'\')([\s\S]*?)\2)', re.DOTALL)

    with open(file_path, "r", encoding="utf-8") as file:
        file_text = file.read()

    match = leading_docstring_pattern.search(file_text)
    if match:
        start, end = match.span()
        file_text = file_text[:start] + new_docstring + '\n' + file_text[end:].lstrip('\n')
    else:
        file_text = new_docstring + file_text

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(file_text)


def _insert_docstring_into_markdown_file(file_path, doc):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(doc)


def _remove_front_matter(string):
    regex = re.compile(r"---\s*(\n|.)*?(---\s*\n)")
    match = regex.match(string)
    if match:
        string = string[len(match.group(0)):]

    start_block_pattern = re.compile(r'```{toctree}', re.DOTALL)
    end_block_pattern = re.compile(r'```', re.DOTALL)

    while True:
        start_match = start_block_pattern.search(string)
        if not start_match:
            break

        end_match = end_block_pattern.search(string, start_match.end())
        if end_match:
            string = string[:start_match.start()] + string[end_match.end():]
        else:
            break

    return string


def _parse_markdown(string):
    parsed_content = defaultdict(str)

    # Regex to find headers and code blocks
    header_pattern = re.compile(r'^(#+) (.+)', re.MULTILINE)
    start_block_pattern = re.compile(r'```python')
    end_block_pattern = re.compile(r'```', re.DOTALL)

    last_header = None
    in_code_block = False

    parsed_content = []
    section = ""
    for line in string.split('\n'):
        header = header_pattern.match(line)
        if header is not None and not in_code_block:
            section = section.rstrip('\n')
            if not section.endswith('\n'):
                section += '\n'

            parsed_content.append((last_header, section))
            section = ""
            last_header = header.group(2)

        is_start_code_block = start_block_pattern.match(line) is not None and not in_code_block
        is_end_code_block = end_block_pattern.match(line) is not None and in_code_block
        if is_start_code_block:
            in_code_block = True
        elif is_end_code_block:
            in_code_block = False

        section += line + '\n'

    parsed_content = parsed_content[1:]
    parsed_content.append((last_header, section))

    return parsed_content


def main():
    """Read the documentation from /docs and put them into the environment header files."""
    envs_dir = os.path.join("..", "free_range_zoo/envs")
    docs_dir = os.path.join("source")

    for env_name in os.listdir(envs_dir):
        dir_path = os.path.join(envs_dir, env_name)

        if not os.path.isdir(dir_path) or env_name == '__pycache__':
            continue

        environment_index = os.path.join(docs_dir, 'environments', env_name, 'index.md')
        environment_specification = os.path.join(docs_dir, 'environments', env_name, 'specification.md')

        environment_runtime_file = os.path.join(dir_path, 'env', f'{env_name}.py')
        environment_readme = os.path.join(dir_path, 'README.md')

        with open(environment_index, encoding="utf-8") as file:
            index_contents = file.read()

        index_contents = _remove_front_matter(index_contents)
        index_contents = _parse_markdown(index_contents)

        with open(environment_specification, encoding="utf-8") as file:
            specification_contents = file.read()

        specification_contents = _remove_front_matter(specification_contents)
        specification_contents = _parse_markdown(specification_contents)

        contents = index_contents + specification_contents

        header_blacklist_python = [
            'Usage',
            'Parallel API',
            'AEC API',
            'Configuration',
            'API',
        ]

        header_blacklist_markdown = [
            'Configuration',
            'API',
        ]

        header_text = ''
        for key, section in contents:
            if key not in header_blacklist_python:
                header_text += f'{section}'
        header_text.rstrip('\n')

        _insert_docstring_into_python_file(environment_runtime_file, header_text)

        header_text = ''
        for key, section in contents:
            if key not in header_blacklist_markdown:
                header_text += f'{section}'
        header_text.rstrip('\n')

        _insert_docstring_into_markdown_file(environment_readme, header_text)


if __name__ == "__main__":
    main()
