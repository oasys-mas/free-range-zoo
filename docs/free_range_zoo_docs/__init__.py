"""This module contains functions to generate and build the documentation and parsing."""
import subprocess


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
    process = subprocess.run(["make", "html"])
    if process.returncode != 0:
        raise RuntimeError("Failed to build the documentation.")


def watch():
    """Build the documentation and watch for changes."""
    subprocess.run(["sphinx-autobuild", "-b", "html", "./source", "_build"])


def test():
    """Test the documentation."""
    # subprocess.run(["pytest", ".", "--markdown-docs", "-m", "markdown-docs"])
    raise NotImplementedError("This function is not implemented.")
