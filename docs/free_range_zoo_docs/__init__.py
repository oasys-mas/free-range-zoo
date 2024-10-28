"""This module contains functions to generate and build the documentation and parsing."""
import subprocess

from . import gen_envs_mds


def gen():
    """Generate the environment markdown files."""
    gen_envs_mds.main()


def build():
    """Build the documentation."""
    subprocess.run(["make", "dirhtml"])


def watch():
    """Build the documentation and watch for changes."""
    subprocess.run(["sphinx-autobuild", "-b", "dirhtml", ".", "_build"])


def test():
    """Test the documentation."""
    subprocess.run(["pytest", ".", "--markdown-docs", "-m", "markdown-docs"])
