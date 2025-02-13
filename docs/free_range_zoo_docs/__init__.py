"""This module contains functions to generate and build the documentation and parsing."""
import subprocess
import os


def gen():
    """Generate the environment markdown files from environment docstrings."""
    from . import gen_envs_mds
    gen_envs_mds.main()


def sync():
    """Sync the environment markdown files with the environment docstrings."""
    from . import gen_envs_docstrings
    gen_envs_docstrings.main()


def build():
    """Build the documentation."""
    makefile_dir = None
    for root, dirs, files in os.walk(os.getcwd()):
        if '.venv' in root or '.git' in root or '__pycache__' in root:
            continue

        if "Makefile" in files:
            makefile_dir = root
            break

    process = subprocess.run(["make", "html"], cwd=makefile_dir)
    if process.returncode != 0:
        raise RuntimeError("Failed to build the documentation.")


def watch():
    """Build the documentation and watch for changes."""
    makefile_dir = None
    for root, dirs, files in os.walk(os.getcwd()):
        if '.venv' in root or '.git' in root or '__pycache__' in root:
            continue

        if "Makefile" in files:
            makefile_dir = root
            break

    subprocess.run(["sphinx-autobuild", "-b", "html", "./source", "build"], cwd=makefile_dir)
