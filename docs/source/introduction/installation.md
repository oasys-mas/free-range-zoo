# Free-Range-Zoo Installation Guide
 ![https://github.com/C4theBomb/free-range-rust]
 ![Free Range Zoo Logo (it is a goat!)](source/_static/img/darkgoat.png)

 In this guide we walk through installing [free-range-zoo](https://github.com/oasys-mas/free-range-zoo). For this we need a python virtual environment, here we use miniconda. The action/observation spaces in free-range-zoo rely on rust functions implemented in [free-range-rust](https://github.com/C4theBomb/free-range-rust) for efficency. Compiling these is part of the `poetry` install, so we will: 
 
1. Install gcc-13 / clang and [Rust](https://www.rust-lang.org/tools/install)
2. Clone  [free-range-zoo](https://github.com/oasys-mas/free-range-zoo)
3. Install [Miniconda](https://docs.anaconda.com/miniconda/install/)
4. Create a python **3.12** environment, we support no lower version
5. Install dependencies
6. Test


# Install gcc-13 / clang

-  check if you have a sufficient C compiler if `> gcc -version` -> `13.0` higher then skip the next install steps for gcc/clang.

>We set these paths so `maturin`, the rust<->python binder, uses the right version of gcc.


## Linux (GCC 13)

-  Install gcc
   - If in **Ubuntu** `sudo apt install gcc-13`
   - for **fedora/rhel/CentOS** `sudo dnf install gcc-toolset-13`
- Set your `gcc` and `g++` paths. Installing gcc with (2) will install `g++` as well.
   - `export CC="<path to your gcc v13 binary>"`
   - `export CXX="<path to your g++ v13 binary>`
 - Make sure the version is correct
   - `gcc -v` = `13.x`



## MacOS (Clang) -tested on apple silicon

- install clang 14/15
  - `brew install llvm@15` 
- Run the path override command that homebrew gives you when running this (this updates your `~/.zshrc`)
- Make sure the version is correct
  - `gcc -v` = `15.x`

## Windows (MinGW-w64)

- Install [Chocolatey](https://chocolatey.org/install)
- Install GCC/MinGW (in administrative powershell)
  - `choco install mingw` currently this gives the right version. You may have to specify version in the future.
- Add `mingw` to path. 
  - Search for `environment variables` in the windows search bar. Click on "environment variables" Then click on `path` in the list. Click `edit`. See if there is a entry for `mingw` If not find the `mingw64\mingw64\bin` folder and add it to `path` (typically this is in `C:\ProgramData`).
- Restart powershell
- Make sure the version is correct
-   `gcc -v` >= `13.x`
-   Reboot your machine. (Needed for rust install if you just installed chocolatey)

# Rust

## Linux / MacOS
-  Go to [Rust Install](https://www.rust-lang.org/tools/install) select your os, and run the curl command.

## Windows
- `choco install rust`
- Restart your machine.
- Open powershell.
- Make sure rust and cargo installed.
  - `rustc`
  - `cargo`

# Clone Free-Range-Zoo

...

# Install Miniconda

> ⚠ DO NOT USE SUDO HERE! it often breaks it.

- Run the installation script for [Miniconda](https://docs.anaconda.com/miniconda/install/), select your os. 


>If using mac make sure to pick apple/intel silicon in your selection.

- Install with defaults, and say `yes` to the conda init question.
- Restart your terminal
- You should see something like `(base) user@machine:...` 


>If you don't then double check that the `miniconda3` folder is in your home directory and that it was added to your path.

# Create Python env

10. Create a python 3.12 environment
    - `conda create -n three12 python=3.12`
11. Activate it
    - `conda activate three12`

# Install Dependencies

11. Install poetry, the dependency tool we use.
    - `pip install poetry`
12. Install all dependencies with poetry
    - `poetry install`

> Sometimes two packages give issues here `neptune` and `neptune-optuna`. If these fail to install manually install them with pip `pip install neptune neptune-optuna` then rerun the poetry install. 

# Test

To see if everything is working try running our unit tests. This will build a local copy of our docs page, see `free-range-zoo/docs/build/index.html`

13. `python -m unittest`